"""Strict offline actor-head artifacts, NOT a resumed PPO checkpoint/certificate.

Only the final deterministic actor linear head may differ from the hash-bound
original110/8802500 source. Learning states are immutable *archives*, not new
Adam steps. Provenance/parity flags are recorded metadata; independent native
input/forward, split and rollout audits must still establish their truth.
"""
import copy
import hashlib
import io
import math
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import torch

from .mu_temporal import state_fingerprint


ARTIFACT_KIND = 'actor_head_offline_v2'
HEAD_SHAPES = {'actor.6.weight': (12, 128), 'actor.6.bias': (12,)}
ORIGINAL110_PARENT = MappingProxyType(dict(
    source_task='TASK_20260926_110',
    source_config='configs/amp/lafan_walk02_sustain_control.json',
    checkpoint_name='model_8802500.pt',
    file_sha256='07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31',
    model_state_sha256='c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc',
    iter=2499, completed_updates=2500, policy_tensor_count=33))
ARCHIVE_ROLE = 'source_learning_states_archived_not_resumed'
REQUIRED_SOURCE_KEYS = frozenset((
    'model_state_dict', 'optimizer_state_dict', 'es_optimizer_state_dict',
    'amp_discriminator_state_dict', 'amp_optimizer_state_dict',
    'amp_replay_state', 'amp_identity', 'iter', 'completed_updates'))
ARTIFACT_KEYS = frozenset((
    'artifact_kind', 'parent', 'model_state_dict', 'amp_identity',
    'candidate_model_state_sha256', 'source_archive', 'source_archive_sha256',
    'archive_role', 'provenance', 'effectiveness_verified', 'dr_unlocked'))


def _fail(message):
    raise ValueError('Offline actor-head artifact: '+message)


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        _fail('missing/extra '+name+' fields')


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        _fail('invalid '+name+' SHA256')


def _number(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or (integer and type(value) is not int)):
        _fail('invalid '+name)
    return value


def _raw_tensor(value):
    if value.layout != torch.strided:
        _fail('unsupported archive tensor layout')
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _tree_digest(value):
    """Type/shape/raw-byte tree hash, including signed zero and unknown metadata."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(b'tensor\0'+str(item.dtype).encode()+b'\0')
            digest.update(str(tuple(item.shape)).encode()+b'\0')
            raw = _raw_tensor(item)
            digest.update(str(len(raw)).encode()+b'\0'+raw)
        elif isinstance(item, dict):
            digest.update(('dict:'+type(item).__name__+'\0').encode())
            digest.update(str(len(item)).encode()+b'\0')
            # Preserve original mapping order too, rather than rewriting archives.
            for key, child in item.items():
                add(key)
                add(child)
            # torch state_dict OrderedDict carries version metadata here.
            if hasattr(item, '__dict__'):
                digest.update(b'mapping_attributes\0')
                add(vars(item))
            else:
                digest.update(b'no_mapping_attributes\0')
        elif isinstance(item, (list, tuple)):
            digest.update((type(item).__name__+':'+str(len(item))+'\0').encode())
            for child in item:
                add(child)
        elif type(item) is float:
            digest.update(b'float\0'+struct.pack('>d', item))
        elif type(item) is int:
            digest.update(('int:'+str(item)+'\0').encode())
        elif type(item) is bool:
            digest.update(b'true\0' if item else b'false\0')
        elif item is None:
            digest.update(b'none\0')
        elif isinstance(item, (str, bytes)):
            raw = item.encode('utf-8') if isinstance(item, str) else item
            digest.update((type(item).__name__+':'+str(len(raw))+'\0').encode()+raw)
        else:
            _fail('unsupported archived leaf type '+type(item).__name__)

    add(value)
    return digest.hexdigest()


def _same_tree(source, actual, path):
    """Reject any archive leaf dtype/shape/value/type change, not just weights."""
    if isinstance(source, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or source.dtype != actual.dtype or
                source.shape != actual.shape or _raw_tensor(source) != _raw_tensor(actual)):
            _fail('changed archived tensor '+path)
    elif isinstance(source, dict):
        if (type(source) is not type(actual) or list(source) != list(actual)):
            _fail('changed archived mapping '+path)
        for key, value in source.items():
            _same_tree(value, actual[key], path+'.'+str(key))
        if hasattr(source, '__dict__') or hasattr(actual, '__dict__'):
            _same_tree(vars(source) if hasattr(source, '__dict__') else {},
                       vars(actual) if hasattr(actual, '__dict__') else {}, path+'.@attributes')
    elif isinstance(source, (list, tuple)):
        if type(source) is not type(actual) or len(source) != len(actual):
            _fail('changed archived sequence '+path)
        for index, value in enumerate(source):
            _same_tree(value, actual[index], path+'.'+str(index))
    elif type(source) is not type(actual) or _tree_digest(source) != _tree_digest(actual):
        _fail('changed archived leaf '+path)


def _validate_model(model, name):
    if (not isinstance(model, dict) or len(model) != 33 or
            any(not isinstance(key, str) for key in model)):
        _fail(name+' must contain exactly 33 named policy tensors')
    for key, tensor in model.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or
                not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
            _fail('nonfinite/nonfloating '+name+' policy tensor '+key)
    for key, shape in HEAD_SHAPES.items():
        if key not in model or tuple(model[key].shape) != shape:
            _fail('wrong '+name+' actor-head shape '+key)


def _load_parent(source_path, expected_parent):
    """Hash and decode the SAME bytes: no claimed hash or path/load TOCTOU."""
    if not isinstance(expected_parent, Mapping) or set(expected_parent) != set(ORIGINAL110_PARENT):
        _fail('invalid expected parent binding')
    binding = dict(expected_parent)
    for key in ('file_sha256', 'model_state_sha256'):
        _sha(binding[key], 'parent '+key)
    for key, expected in (('iter', 2499), ('completed_updates', 2500), ('policy_tensor_count', 33)):
        if type(binding[key]) is not int or binding[key] != expected:
            _fail('invalid parent '+key)
    for key in ('source_task', 'source_config', 'checkpoint_name'):
        if not isinstance(binding[key], str) or not binding[key]:
            _fail('missing parent '+key)
    blob = Path(source_path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != binding['file_sha256']:
        _fail('parent file SHA256 mismatch')
    try:
        source = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError('Offline actor-head artifact: cannot decode parent checkpoint') from error
    if not isinstance(source, dict) or not REQUIRED_SOURCE_KEYS <= set(source):
        _fail('missing parent policy/learning-state archive')
    for key in ('iter', 'completed_updates'):
        if type(source[key]) is not int or source[key] != binding[key]:
            _fail('parent source '+key+' mismatch')
    _validate_model(source['model_state_dict'], 'parent')
    if state_fingerprint(source['model_state_dict']) != binding['model_state_sha256']:
        _fail('parent model-state fingerprint mismatch')
    for key in REQUIRED_SOURCE_KEYS-{'iter', 'completed_updates', 'model_state_dict'}:
        if not isinstance(source[key], dict) or not source[key]:
            _fail('empty/malformed parent '+key)
    _tree_digest(source)  # Reject unsupported unknown archive leaves explicitly.
    return source, binding


def _validate_provenance(provenance):
    fields = ('code_commit', 'implementation_fingerprint', 'mode', 'solver', 'budget',
              'cohorts', 'source_parity', 'report_sha256', 'offline_admitted',
              'effectiveness_verified', 'dr_unlocked')
    _exact_keys(provenance, fields, 'provenance')
    if (not isinstance(provenance['code_commit'], str) or
            re.fullmatch('[0-9a-f]{40}', provenance['code_commit']) is None):
        _fail('invalid solver code_commit')
    for key in ('implementation_fingerprint', 'report_sha256'):
        _sha(provenance[key], key)
    if provenance['mode'] not in ('smoke', 'formal'):
        _fail('invalid offline mode')
    if type(provenance['offline_admitted']) is not bool:
        _fail('invalid offline admission metadata')
    for key in ('effectiveness_verified', 'dr_unlocked'):
        if provenance[key] is not False:
            _fail('offline metadata cannot claim '+key)
    solver = provenance['solver']
    constants = dict(kind='quadratic_direction_per_axis_v2', temporal_weight=1., ridge=1e-6,
                     filter_window=21, action_scale=.5, dtype='float64', solve_count=1,
                     direction_scope='per_output_axis_train_only')
    _exact_keys(solver, tuple(constants)+('direction_scale_per_joint',), 'solver')
    for key, expected in constants.items():
        actual = solver[key]
        if isinstance(expected, (int, float)):
            _number(actual, 'solver '+key, integer=type(expected) is int)
        if actual != expected:
            _fail('changed solver '+key)
    scales = solver['direction_scale_per_joint']
    if (type(scales) is not list or len(scales) != 12
            or any(type(scale) is not float or not math.isfinite(scale)
                   or not 0. <= scale <= 1. for scale in scales)):
        _fail('direction scales must be exactly 12 finite floats in [0,1]')
    num_envs, duration_s = (4, 2) if provenance['mode'] == 'smoke' else (8, 20)
    budget = provenance['budget']
    _exact_keys(budget, ('num_envs', 'duration_s', 'fit_episode_count'), 'budget')
    for key, expected in (('num_envs', num_envs), ('duration_s', duration_s),
                          ('fit_episode_count', num_envs)):
        if _number(budget[key], 'budget '+key, integer=key != 'duration_s') != expected:
            _fail('offline mode/budget mismatch '+key)
    cohorts = provenance['cohorts']
    _exact_keys(cohorts, ('train', 'validation'), 'cohorts')
    for split, seed in (('train', 306), ('validation', 506)):
        cohort = cohorts[split]
        _exact_keys(cohort, ('sha256', 'seed', 'num_envs', 'duration_s', 'episode_ids'), split+' cohort')
        _sha(cohort['sha256'], split+' cohort')
        for key, expected in (('seed', seed), ('num_envs', num_envs), ('duration_s', duration_s)):
            if _number(cohort[key], split+' '+key, integer=key != 'duration_s') != expected:
                _fail('changed '+split+' cohort '+key)
        ids = cohort['episode_ids']
        expected_ids = ['head_cohort_v2/%s/seed-%d/env-%d/episode-0' % (split, seed, i)
                        for i in range(num_envs)]
        if type(ids) is not list or ids != expected_ids:
            _fail('wrong/duplicate/sealed/source-regression '+split+' episode IDs')
    if cohorts['train']['sha256'] == cohorts['validation']['sha256']:
        _fail('training/validation cohort hashes overlap')
    if set(cohorts['train']['episode_ids']) & set(cohorts['validation']['episode_ids']):
        _fail('training/validation episode IDs overlap')
    parity = provenance['source_parity']
    _exact_keys(parity, ('full_obs_verified', 'hidden_verified', 'action_verified',
                        'max_abs_error'), 'source parity')
    for key in ('full_obs_verified', 'hidden_verified', 'action_verified'):
        if parity[key] is not True:
            _fail('missing recorded source parity '+key)
    if not 0 <= _number(parity['max_abs_error'], 'source parity error') < .001:
        _fail('recorded source parity error exceeds bound')


def _validate(artifact, source, binding):
    _exact_keys(artifact, ARTIFACT_KEYS, 'artifact')
    if artifact['artifact_kind'] != ARTIFACT_KIND or artifact['archive_role'] != ARCHIVE_ROLE:
        _fail('not an offline head / learning-state archive')
    _same_tree(binding, artifact['parent'], 'parent')
    for key in ('effectiveness_verified', 'dr_unlocked'):
        if artifact[key] is not False:
            _fail('offline artifact cannot claim '+key)
    _validate_provenance(artifact['provenance'])
    _same_tree(source, artifact['source_archive'], 'source_archive')
    _same_tree(source['amp_identity'], artifact['amp_identity'], 'amp_identity')
    _sha(artifact['source_archive_sha256'], 'source archive')
    if artifact['source_archive_sha256'] != _tree_digest(source):
        _fail('source archive fingerprint mismatch')
    model, original = artifact['model_state_dict'], source['model_state_dict']
    _validate_model(model, 'candidate')
    if type(model) is not type(original) or list(model) != list(original):
        _fail('changed candidate policy mapping/keys/order')
    _same_tree(vars(original) if hasattr(original, '__dict__') else {},
               vars(model) if hasattr(model, '__dict__') else {}, 'model_state_dict.@attributes')
    for key in original:
        if key in HEAD_SHAPES:
            if model[key].dtype != original[key].dtype:
                _fail('changed candidate head dtype '+key)
        else:
            _same_tree(original[key], model[key], 'model_state_dict.'+key)
    _sha(artifact['candidate_model_state_sha256'], 'candidate model state')
    if artifact['candidate_model_state_sha256'] != state_fingerprint(model):
        _fail('candidate model-state fingerprint mismatch')
    return dict(artifact_kind=ARTIFACT_KIND, parent_file_sha256=binding['file_sha256'],
                parent_model_state_sha256=binding['model_state_sha256'],
                candidate_model_state_sha256=artifact['candidate_model_state_sha256'],
                unchanged_policy_tensor_count=31, editable_policy_tensor_count=2,
                source_archive_sha256=artifact['source_archive_sha256'],
                source_learning_states_archived=True, learning_states_resumed=False,
                provenance_is_metadata_not_independent_forward_evidence=True,
                effectiveness_verified=False, dr_unlocked=False)


def assemble_head_artifact(source_path, head_state, *, provenance,
                           expected_parent=ORIGINAL110_PARENT):
    """Assemble only two edited tensors; never write a model or run a forward."""
    source, binding = _load_parent(source_path, expected_parent)
    _exact_keys(head_state, HEAD_SHAPES, 'head state')
    model = copy.deepcopy(source['model_state_dict'])
    for key, shape in HEAD_SHAPES.items():
        value = head_state[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
                tuple(value.shape) != shape or
                value.dtype != model[key].dtype or not bool(torch.isfinite(value).all())):
            _fail('invalid fitted head '+key)
        model[key] = value.detach().cpu().clone()
    artifact = dict(artifact_kind=ARTIFACT_KIND, parent=binding,
                    model_state_dict=model, amp_identity=copy.deepcopy(source['amp_identity']),
                    candidate_model_state_sha256=state_fingerprint(model),
                    source_archive=copy.deepcopy(source), source_archive_sha256=_tree_digest(source),
                    archive_role=ARCHIVE_ROLE, provenance=copy.deepcopy(provenance),
                    effectiveness_verified=False, dr_unlocked=False)
    _validate(artifact, source, binding)
    return artifact


def validate_head_artifact(artifact, source_path, *, expected_parent=ORIGINAL110_PARENT):
    """Independently reload the bound parent and reject every unallowed mutation.

    ``expected_parent`` exists for explicitly synthetic fixtures/independent
    callers. Production admission must retain the ORIGINAL110_PARENT default.
    No old smoke/formal certificate, PPO update counter or native optimizer
    resume state is accepted at the artifact's top level.
    """
    source, binding = _load_parent(source_path, expected_parent)
    return _validate(artifact, source, binding)
