"""Only allow the two planned changes relative to source tail2000."""
import copy
from humanoid.amp.direction import direction_contract
from tools.amp.progress_audit import validate_progress_updates as validate_direction_updates


def compare_environment(source, target, group, smoke=False):
    c = direction_contract(group)
    after = copy.deepcopy(target)
    if after['rewards'].pop('direction_world_fraction', None) != c['world_fraction']:
        raise ValueError('Wrong actual world mixture')
    if after['rewards']['scales']['refine_heading'] != c['heading_scale']:
        raise ValueError('Wrong actual heading scale')
    if source['rewards']['scales']['refine_heading'] != -.5:
        raise ValueError('Wrong source heading scale')
    after['rewards']['scales']['refine_heading'] = -.5
    if smoke: after['env']['num_envs'] = source['env']['num_envs']
    if after != source:
        raise ValueError('Environment changed beyond bounded direction and smoke size')
    return True
