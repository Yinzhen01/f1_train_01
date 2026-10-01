"""Deterministic target bandwidth, not observation noise or domain randomization."""
import math


def filter_alpha(cutoff_hz, dt):
    if not (math.isfinite(dt) and dt > 0. and math.isfinite(cutoff_hz) and 0. < cutoff_hz < .5/dt):
        raise ValueError('Invalid target filter rate')
    return 1.-math.exp(-2.*math.pi*cutoff_hz*dt)


def filter_target(raw, previous, alpha):
    return alpha*raw+(1.-alpha)*previous


def validate_filter_diagnostics(report, spec, steps, num_envs):
    if report.get('target_filter') != spec or report.get('filter_calls') != steps:
        raise ValueError('Missing actual controller filter calls/definition')
    expected = filter_alpha(spec['cutoff_hz'], spec['dt'])
    if not math.isclose(report.get('filter_alpha', float('nan')), expected, rel_tol=1e-6):
        raise ValueError('Wrong filter alpha')
    if report.get('filter_reset_count', 0) < 1:
        raise ValueError('Filter reset path not exercised')
    for key in ('filter_input_delta_squared_sum', 'filter_applied_delta_squared_sum', 'filter_residual_squared_sum'):
        if not math.isfinite(report.get(key, float('nan'))) or report[key] <= 0.:
            raise ValueError('Missing actual filter signal: '+key)
    if not (0 < report.get('filter_valid_intervals', 0) <= steps*num_envs):
        raise ValueError('Invalid filter sample count')
    return True
