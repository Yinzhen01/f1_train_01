"""Resume the three interrupted cycle groups; unchanged rewards and dynamics."""
from humanoid.scripts.train_gmr_cycle import main


if __name__ == '__main__':
    main(restart=True)
