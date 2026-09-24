"""Explicit evaluation budgets; long validation never changes training horizon."""


def validate_evaluation_budget(num_envs, duration, extended_validation=False):
    if extended_validation:
        if (num_envs, duration) != (16, 60.):
            raise ValueError('Extended validation is exactly 16 environments x 60 seconds')
    elif num_envs not in (2, 16) or duration not in (1., 20.):
        raise ValueError('Unexpected independent evaluation budget')
    return True


def independent_mode_seeds(seed, extended_validation=False):
    # Preserve all existing 1s/20s evaluations. The extended protocol resets
    # each mode's RNG so earlier failures cannot change later RSI samples.
    return dict(standing=seed, reference=seed+100) if extended_validation else None
