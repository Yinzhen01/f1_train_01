"""CPU-side progress-frame artifact invariants; never simulator acceptance."""
import copy
import math

from humanoid.amp.progress import progress_contract


def validate_progress_updates(rows, formal=False):
    end = 2250 if formal else 2010
    if [r['iteration'] for r in rows] != list(range(2001, end+1)):
        raise ValueError('Not exactly the bounded updates from2000')
    for r in rows:
        if (not all(math.isfinite(v) for v in r.values()) or
            abs(r['weighted_style_reward']-r['style_reward']) >= 1e-8 or
            r['discriminator_bridge_gradient_penalty'] <= 0 or r['style_negative_fraction'] != 0):
            raise ValueError('Changed or inactive AMP bridge mechanics')
    return True


def compare_environment(source, target, group, smoke=False):
    progress_contract(group)
    after = copy.deepcopy(target)
    if after['rewards'].pop('progress_velocity_frame', None) != group:
        raise ValueError('Wrong actual progress frame')
    if smoke: after['env']['num_envs'] = source['env']['num_envs']
    if after != source:
        raise ValueError('Environment changed beyond explicit progress frame and smoke size')
    return True
