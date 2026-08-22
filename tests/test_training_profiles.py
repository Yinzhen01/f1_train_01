import unittest
from pathlib import Path
from types import SimpleNamespace

from humanoid.training_profiles import (
    apply_training_profile,
    load_training_profiles,
)


def make_args(**overrides):
    values = {
        "training_profile": None,
        "task": "ignored_default",
        "resume": False,
        "experiment_name": None,
        "load_run": None,
        "checkpoint": None,
        "seed": None,
        "num_envs": None,
        "max_iterations": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingProfilesTest(unittest.TestCase):
    def test_expected_profiles_are_available(self):
        profiles = load_training_profiles()
        self.assertEqual(
            set(profiles),
            {
                "no_dr_scratch",
                "stage1_from_no_dr",
                "full_dr_from_stage1",
                "full_dr_scratch",
                "retarget_walk_no_dr",
                "retarget_walk_resume",
                "retarget_walk_vx045_no_dr",
                "retarget_walk_vx045_conservative_no_dr",
                "retarget_walk_vx045_geometry_no_dr",
                "retarget_walk_native_geometry_no_dr",
                "retarget_walk_vx045_resume",
            },
        )

    def test_profile_tasks_are_registered(self):
        profiles = load_training_profiles()
        registry_source = (
            Path(__file__).resolve().parents[1] / "humanoid" / "envs" / "__init__.py"
        ).read_text(encoding="utf-8")

        for task in {profile.task for profile in profiles.values()}:
            self.assertIn(f'"{task}"', registry_source)

    def test_full_dr_scratch_reuses_full_dr_task(self):
        args = make_args(training_profile="full_dr_scratch")
        apply_training_profile(args)

        self.assertEqual(args.task, "x1_dh_stand_dr_full")
        self.assertFalse(args.resume)
        self.assertEqual(args.experiment_name, "x1_dh_stand_full_dr_baseline")
        self.assertEqual(args.seed, 5)
        self.assertEqual(args.num_envs, 4096)
        self.assertEqual(args.max_iterations, 6000)

    def test_retarget_walk_profile_uses_dedicated_task(self):
        args = make_args(training_profile="retarget_walk_no_dr")
        apply_training_profile(args)

        self.assertEqual(args.task, "x1_dh_stand_retarget_walk")
        self.assertFalse(args.resume)
        self.assertEqual(
            args.experiment_name,
            "x1_dh_stand_retarget_walk_periodic_contact",
        )
        self.assertEqual(args.seed, 5)
        self.assertEqual(args.num_envs, 4096)
        self.assertEqual(args.max_iterations, 3000)

    def test_retarget_walk_resume_requires_and_uses_explicit_checkpoint(self):
        args = make_args(
            training_profile="retarget_walk_resume",
            load_run="gm_resume",
            checkpoint=500,
        )
        apply_training_profile(args)

        self.assertEqual(args.task, "x1_dh_stand_retarget_walk")
        self.assertTrue(args.resume)
        self.assertEqual(args.load_run, "gm_resume")
        self.assertEqual(args.checkpoint, 500)
        self.assertEqual(args.max_iterations, 2500)

    def test_retarget_walk_vx045_resume_uses_time_scaled_task(self):
        args = make_args(
            training_profile="retarget_walk_vx045_resume",
            load_run="gm_resume",
            checkpoint=2000,
        )
        apply_training_profile(args)

        self.assertEqual(args.task, "x1_dh_stand_retarget_walk_vx045")
        self.assertEqual(
            args.experiment_name,
            "x1_dh_stand_retarget_walk_periodic_contact_vx045",
        )
        self.assertTrue(args.resume)
        self.assertEqual(args.max_iterations, 1500)

    def test_retarget_walk_geometry_profile_uses_dedicated_task(self):
        args = make_args(training_profile="retarget_walk_vx045_geometry_no_dr")
        apply_training_profile(args)

        self.assertEqual(
            args.task, "x1_dh_stand_retarget_walk_vx045_geometry"
        )
        self.assertFalse(args.resume)
        self.assertEqual(
            args.experiment_name,
            "x1_dh_stand_retarget_walk_periodic_contact_vx045_geometry",
        )
        self.assertEqual(args.max_iterations, 3000)

    def test_retarget_walk_native_geometry_profile_uses_native_rate_task(self):
        args = make_args(training_profile="retarget_walk_native_geometry_no_dr")
        apply_training_profile(args)

        self.assertEqual(
            args.task, "x1_dh_stand_retarget_walk_native_geometry"
        )
        self.assertFalse(args.resume)
        self.assertEqual(
            args.experiment_name,
            "x1_dh_stand_retarget_walk_periodic_contact_native_geometry",
        )
        self.assertEqual(args.max_iterations, 3000)

    def test_resume_profile_requires_explicit_source(self):
        args = make_args(training_profile="stage1_from_no_dr")
        with self.assertRaisesRegex(ValueError, "explicit source"):
            apply_training_profile(args)

    def test_resume_profile_sets_task_and_preserves_cli_override(self):
        args = make_args(
            training_profile="full_dr_from_stage1",
            load_run="stage1_source",
            checkpoint=6700,
            max_iterations=1200,
        )
        apply_training_profile(args)

        self.assertEqual(args.task, "x1_dh_stand_dr_full")
        self.assertTrue(args.resume)
        self.assertEqual(args.checkpoint, 6700)
        self.assertEqual(args.max_iterations, 1200)

    def test_scratch_profile_rejects_resume_arguments(self):
        args = make_args(
            training_profile="no_dr_scratch",
            load_run="unexpected",
            checkpoint=100,
        )
        with self.assertRaisesRegex(ValueError, "starts from scratch"):
            apply_training_profile(args)

    def test_unknown_profile_is_rejected(self):
        args = make_args(training_profile="not_a_profile")
        with self.assertRaisesRegex(ValueError, "Unknown training profile"):
            apply_training_profile(args)


if __name__ == "__main__":
    unittest.main()
