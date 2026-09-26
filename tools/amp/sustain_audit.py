"""Only a bounded progress weight may differ from the exact heading2250 source."""
import copy

from humanoid.amp.sustain import sustain_contract
from tools.amp.progress_audit import validate_progress_updates


def validate_sustain_updates(rows, formal=False):
    # Reuse the AMP log-content checks, but require exactly2251..2260/2500.
    shifted = [dict(row, iteration=row['iteration']-250) for row in rows]
    return validate_progress_updates(shifted, formal)


def compare_environment(source, target, group, smoke=False):
    c = sustain_contract(group)
    after = copy.deepcopy(target)
    for cfg in (source, after):
        if (cfg['rewards']['direction_world_fraction'] != 0. or
            cfg['rewards']['scales']['refine_heading'] != -1.5):
            raise ValueError('Unexpected source or target direction definition')
    if (source['rewards']['scales']['recovery_progress'] != 2. or
        after['rewards']['scales']['recovery_progress'] != c['progress_scale']):
        raise ValueError('Wrong progress weight')
    after['rewards']['scales']['recovery_progress'] = 2.
    if smoke: after['env']['num_envs'] = source['env']['num_envs']
    if after != source: raise ValueError('Environment changed beyond progress weight and smoke size')
    return True
