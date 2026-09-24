"""Read-only checkpoint probes for AMP failure analysis, no simulator claims."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.discriminator import AMPDiscriminator
from humanoid.amp.features import build_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    e = ScaledExperiment(ROOT)
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if model["amp_identity"] != e.identity():
        raise ValueError("Wrong experiment checkpoint")
    d = AMPDiscriminator(e.spec, e.mean, e.std)
    d.load_state_dict(model["amp_discriminator_state_dict"], strict=True)
    d.eval()
    q = torch.tensor(e.clip.joint_pos[1:], dtype=torch.float32)
    default = torch.tensor([[.4, .05, -.31, .49, -.21, 0., -.4, -.05, .31, .49, -.21, 0.]])
    def static(q):
        key = torch.tensor(e.kinematics.key_positions(q.numpy(), e.spec.joint_names), dtype=torch.float32)
        frame = build_features(e.spec, torch.zeros(len(q), 3), q, torch.zeros_like(q),
                               e.spec.joint_names, key, e.spec.body_names)
        return frame[:, None].expand(-1, 10, -1).clone()
    probes = {"demo": e.windows, "default_static": static(default),
              "reference_poses_held_static": static(q), "demo_time_reversed": e.windows.flip(1)}
    result = dict(checkpoint=str(args.checkpoint), completed_updates=model["completed_updates"],
                  experiment=e.identity(), probes={})
    with torch.no_grad():
        for name, windows in probes.items():
            scores = d(windows).flatten()
            reward = d.style_reward(windows, scale=2.)
            result["probes"][name] = dict(score_mean=float(scores.mean()), score_min=float(scores.min()),
                score_max=float(scores.max()), style_reward_mean=float(reward.mean()),
                zero_reward_fraction=float((reward == 0).float().mean()))
    result["normalizer"] = dict(mean=e.mean.tolist(), std=e.std.tolist(),
        tiny_std_feature_indices=torch.where(e.std <= .010001)[0].tolist(),
        default_zscore=((probes["default_static"][0, 0]-e.mean)/e.std).tolist())
    result["stationary_task_terms_per_second"] = dict(
        linear_tracking=2*np.exp(-.45**2*5), yaw_tracking=.5, upright=.5, velocity_stability=.2,
        note="Analytical ideal stationary pose before regularizers; not a measured rollout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "normalizer"}, indent=2))


if __name__ == "__main__":
    main()
