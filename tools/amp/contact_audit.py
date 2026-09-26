"""Shared matched-contact artifact checks, independent of Isaac Gym."""
import copy
import math

from humanoid.amp.contact import contact_contract


def validate_contact_updates(rows, formal=False):
    end = 2000 if formal else 1760
    if [r['iteration'] for r in rows] != list(range(1751, end+1)):
        raise ValueError('Not exactly the bounded updates from1750')
    for r in rows:
        if (not all(math.isfinite(v) for v in r.values()) or
            abs(r['weighted_style_reward']-r['style_reward']) >= 1e-8 or
            r['discriminator_bridge_gradient_penalty'] <= 0 or r['style_negative_fraction'] != 0):
            raise ValueError('Changed or inactive AMP bridge mechanics')
    return True


def compare_environment(source, target, group, smoke=False):
    c = contact_contract(group)
    after = copy.deepcopy(target)
    scales = after['rewards']['scales']
    if scales.pop('contact_force_tail', None) != c['contact_force_tail_scale']:
        raise ValueError('Wrong contact tail scale')
    if scales['refine_slip'] != c['refine_slip_scale']:
        raise ValueError('Wrong contact slip scale')
    scales['refine_slip'] = -2.
    if smoke: after['env']['num_envs'] = source['env']['num_envs']
    if after != source:
        raise ValueError('Environment changed beyond explicit contact scales and smoke size')
    return True
