"""Declarative training profiles for reproducible X1 experiment launches."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union


DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "training" / "x1_profiles.json"
)
_ALLOWED_INITIALIZATIONS = {"scratch", "resume"}
_ALLOWED_DEFAULTS = {"seed", "num_envs", "max_iterations"}
_ALLOWED_REQUIRED_ARGS = {"load_run", "checkpoint"}


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    task: str
    initialization: str
    experiment_name: str
    defaults: Mapping[str, int]
    required_args: tuple[str, ...]

    @property
    def resume(self) -> bool:
        return self.initialization == "resume"


def _require_non_empty_string(value: Any, field: str, profile_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Profile '{profile_name}' must define non-empty '{field}'")
    return value


def _parse_profile(name: str, raw: Any) -> TrainingProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"Profile '{name}' must be a JSON object")

    initialization = _require_non_empty_string(
        raw.get("initialization"), "initialization", name
    )
    if initialization not in _ALLOWED_INITIALIZATIONS:
        raise ValueError(
            f"Profile '{name}' initialization must be one of "
            f"{sorted(_ALLOWED_INITIALIZATIONS)}"
        )

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Profile '{name}' defaults must be a JSON object")
    unknown_defaults = set(defaults) - _ALLOWED_DEFAULTS
    if unknown_defaults:
        raise ValueError(
            f"Profile '{name}' has unsupported defaults: {sorted(unknown_defaults)}"
        )
    for key, value in defaults.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"Profile '{name}' default '{key}' must be a positive integer"
            )

    required_args = raw.get("required_args", [])
    if not isinstance(required_args, list) or not all(
        isinstance(item, str) for item in required_args
    ):
        raise ValueError(f"Profile '{name}' required_args must be a string list")
    unknown_required = set(required_args) - _ALLOWED_REQUIRED_ARGS
    if unknown_required:
        raise ValueError(
            f"Profile '{name}' has unsupported required args: {sorted(unknown_required)}"
        )
    if initialization == "scratch" and required_args:
        raise ValueError(f"Scratch profile '{name}' cannot require resume arguments")

    return TrainingProfile(
        name=name,
        description=_require_non_empty_string(raw.get("description"), "description", name),
        task=_require_non_empty_string(raw.get("task"), "task", name),
        initialization=initialization,
        experiment_name=_require_non_empty_string(
            raw.get("experiment_name"), "experiment_name", name
        ),
        defaults=dict(defaults),
        required_args=tuple(required_args),
    )


def load_training_profiles(
    path: Optional[Union[Path, str]] = None,
) -> Dict[str, TrainingProfile]:
    profile_path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    with profile_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    if not isinstance(document, dict):
        raise ValueError(f"Training profile document must be an object: {profile_path}")
    if document.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported training profile schema in {profile_path}: "
            f"{document.get('schema_version')!r}"
        )
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError(f"No training profiles defined in {profile_path}")

    return {
        name: _parse_profile(name, raw)
        for name, raw in sorted(raw_profiles.items())
    }


def _validate_resume_source(args: Any, profile: TrainingProfile) -> None:
    missing = []
    for field in profile.required_args:
        value = getattr(args, field, None)
        invalid_checkpoint = field == "checkpoint" and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        )
        if value is None or value == "" or value == -1 or value == "-1" or invalid_checkpoint:
            missing.append(f"--{field}")
    if missing:
        raise ValueError(
            f"Training profile '{profile.name}' requires an explicit source: "
            + ", ".join(missing)
        )


def apply_training_profile(
    args: Any,
    profile_name: Optional[str] = None,
    path: Optional[Union[Path, str]] = None,
) -> Any:
    """Apply one training profile to parsed Isaac Gym CLI arguments.

    The profile owns the task and scratch/resume mode. Explicit CLI values for
    seed, environment count, iteration count, and experiment name override the
    profile defaults. Resume profiles require an explicit run and checkpoint so
    that a launch cannot silently load whichever checkpoint happens to be last.
    """

    selected_name = profile_name or getattr(args, "training_profile", None)
    if not selected_name:
        return args

    profiles = load_training_profiles(path)
    try:
        profile = profiles[selected_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training profile '{selected_name}'. Available profiles: "
            + ", ".join(profiles)
        ) from exc

    if not profile.resume and getattr(args, "resume", False):
        raise ValueError(
            f"Training profile '{profile.name}' starts from scratch and cannot "
            "be combined with --resume"
        )

    args.task = profile.task
    args.resume = profile.resume
    if getattr(args, "experiment_name", None) is None:
        args.experiment_name = profile.experiment_name
    for field, value in profile.defaults.items():
        if getattr(args, field, None) is None:
            setattr(args, field, value)

    if profile.resume:
        _validate_resume_source(args, profile)
    elif getattr(args, "load_run", None) is not None or getattr(
        args, "checkpoint", None
    ) is not None:
        raise ValueError(
            f"Training profile '{profile.name}' starts from scratch; remove "
            "--load_run and --checkpoint"
        )

    print(
        "[training-profile] "
        f"name={profile.name} task={args.task} initialization={profile.initialization} "
        f"experiment={args.experiment_name} seed={args.seed} "
        f"num_envs={args.num_envs} max_iterations={args.max_iterations}",
        flush=True,
    )
    return args


def format_training_profiles(profiles: Iterable[TrainingProfile]) -> str:
    lines = []
    for profile in profiles:
        lines.append(
            f"{profile.name}: task={profile.task}, "
            f"initialization={profile.initialization} - {profile.description}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help="Training profile JSON file",
    )
    args = parser.parse_args()
    profiles = load_training_profiles(args.config)
    print(format_training_profiles(profiles.values()))


if __name__ == "__main__":
    main()
