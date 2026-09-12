"""Approved formal phase experiment: original model8000, 4096 envs, 1000 updates."""
from isaacgym import gymapi  # Must precede the shared entry's torch import.
from humanoid.scripts.train_gmr_phase_smoke import main


if __name__ == "__main__":
    main(mode="formal")
