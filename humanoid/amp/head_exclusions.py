"""Exact source110 SHA256 exclusion inputs; not model/rollout/independence proof.

NumPy-only import keeps the native Isaac Gym-before-torch ABI intact. Readers
must receive the independently pinned file hash from the native driver, never
an arbitrary user-provided admission binding. Every digest retains all 256 bits
and all associated episode labels, including legitimate source collisions.
"""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np


SCHEMA = 'source110_exclusion_index_v1'
KINDS = ('initial_state', 'initial_full_observation', 'ready_full_history')
SOURCE_ENDPOINT_SHA = 'de77d4a0c85cd581ae9dc9418d3fbce144730749b9f746f93b9671ec122a833b'
SOURCE_CHECKPOINT_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'
SOURCE_MODEL_SHA = 'c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc'
SOURCE_TASK = 'TASK_20260926_110'
EPISODE_LABELS = tuple('source110/%s/env-%d/episode-0' % (mode, i)
                       for mode in ('standing', 'reference') for i in range(16))
PROTOCOL = 'collector_source_exclusions_exact_cpu_reconstruction_v1'
LIMITATION = ('Exact hash exclusion of source-only CPU reconstructed inputs is not '
              'bitwise GPU input parity, statistical independence, semantic novelty, '
              'training effectiveness, a model checkpoint, or a deployment certificate.')


def _fail(message):
    raise ValueError('Source110 exclusion index: '+message)


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        _fail('invalid '+name+' SHA256')


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        _fail('missing/extra '+name+' fields')


def _identity(actual, expected):
    if type(actual) is not dict or type(expected) is not dict or actual != expected:
        _fail('original source identity mismatch')
    _keys(actual, ('experiment', 'config_sha256', 'motion_sha256',
                   'urdf_lf_sha256', 'feature_fingerprint'), 'source identity')
    if actual['experiment'] != 'f1_amp_walk02_sustain_control':
        _fail('not the original source experiment')
    for key in set(actual)-{'experiment'}:
        _sha(actual[key], 'identity '+key)


def _validate_proof(proof):
    required = ('sha256', 'source_task', 'regression_episode_ids', 'initial_state',
                'initial_full_observation', 'ready_full_history',
                'reconstruction_not_raw_history', 'limitation',
                'source_environment', 'source_runtime')
    _keys(proof, required, 'source reconstruction proof')
    if (proof['sha256'] != SOURCE_ENDPOINT_SHA or proof['source_task'] != SOURCE_TASK or
            proof['regression_episode_ids'] != list(EPISODE_LABELS) or
            proof['reconstruction_not_raw_history'] is not True):
        _fail('wrong source regression provenance/labels')
    for key in KINDS+('limitation',):
        if not isinstance(proof[key], str) or not proof[key]:
            _fail('missing source reconstruction semantics '+key)
    environment, runtime = proof['source_environment'], proof['source_runtime']
    if type(environment) is not dict or not environment or type(runtime) is not dict:
        _fail('missing original environment/runtime metadata')
    if not {'dof_properties', 'p_gains', 'd_gains', 'control_dt', 'physics_dt'} <= set(runtime):
        _fail('missing original runtime readback')
    for key, expected in (('control_dt', .01), ('physics_dt', .001)):
        value = runtime[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-10)):
            _fail('wrong original source timing metadata')


def _validate_index(index):
    _keys(index, KINDS, 'three exclusion classes')
    counts = {}
    positions = {label: i for i, label in enumerate(EPISODE_LABELS)}
    for kind in KINDS:
        if type(index[kind]) is not dict or not index[kind]:
            _fail('missing '+kind+' entries')
        label_counts = {label: 0 for label in EPISODE_LABELS}
        for digest, labels in index[kind].items():
            _sha(digest, kind+' digest')
            if (type(labels) is not list or not labels or
                    any(type(label) is not str or label not in positions for label in labels)):
                _fail('wrong '+kind+' label association')
            ids = [positions[label] for label in labels]
            if ids != sorted(set(ids)):
                _fail('duplicate/noncanonical '+kind+' label associations')
            for label in labels:
                label_counts[label] += 1
        if (any(count < 1 for count in label_counts.values()) or
                (kind != 'ready_full_history' and any(count != 1 for count in label_counts.values())) or
                (kind == 'ready_full_history' and any(count > 5935 for count in label_counts.values()))):
            _fail('missing/excess source32 '+kind+' coverage')
        counts[kind] = dict(unique_hashes=len(index[kind]),
                            label_associations=sum(label_counts.values()))
    return counts


def _metadata(index, proof, identity, builder_sha):
    _identity(identity, identity)
    _sha(builder_sha, 'builder collector source')
    _validate_proof(proof)
    return dict(schema=SCHEMA, hash_bits=256, digest_encoding='uint8[N,32]',
        source_task=SOURCE_TASK, source_endpoint_sha256=SOURCE_ENDPOINT_SHA,
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA, source_model_state_sha256=SOURCE_MODEL_SHA,
        identity=copy.deepcopy(identity), building_protocol=PROTOCOL,
        builder=dict(module='humanoid.scripts.collect_amp_head_cohort.source_exclusions',
                     collector_file_sha256=builder_sha), counts=_validate_index(index),
        source_regression=copy.deepcopy(proof), limitation=LIMITATION,
        effectiveness_verified=False, dr_unlocked=False)


def write_exclusion_index(path, index, proof, *, identity, builder_sha):
    """Exclusively write derived fingerprints, never overwrite source or output."""
    metadata = _metadata(index, proof, identity, builder_sha)
    arrays = dict(episode_labels=np.asarray(EPISODE_LABELS, dtype=np.str_))
    positions = {label: i for i, label in enumerate(EPISODE_LABELS)}
    for kind in KINDS:
        digests = sorted(index[kind])
        arrays[kind+'_sha256'] = np.asarray(
            [list(bytes.fromhex(digest)) for digest in digests], dtype=np.uint8)
        offsets, labels = [0], []
        for digest in digests:
            labels.extend(positions[label] for label in index[kind][digest])
            offsets.append(len(labels))
        arrays[kind+'_label_offsets'] = np.asarray(offsets, dtype=np.int64)
        arrays[kind+'_label_indices'] = np.asarray(labels, dtype=np.uint8)
    arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False), dtype=np.str_)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail('duplicate metadata key '+key)
        result[key] = value
    return result


def _invalid_json_number(value):
    _fail('nonfinite metadata number '+value)


def load_exclusion_index(path, *, expected_file_sha, identity):
    """Load only pinned allow_pickle=False exact256bit source-derived input."""
    _sha(expected_file_sha, 'expected file')
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected_file_sha:
        _fail('file SHA256 differs from independent native-driver binding')
    expected_keys = {'metadata_json', 'episode_labels'} | {
        kind+suffix for kind in KINDS for suffix in ('_sha256', '_label_offsets', '_label_indices')}
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            if len(archive.files) != len(expected_keys) or set(archive.files) != expected_keys:
                _fail('missing/extra NPZ arrays')
            arrays = {key: archive[key].copy() for key in archive.files}
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError('Source110 exclusion index: invalid non-pickle NPZ') from error
    encoded, labels = arrays['metadata_json'], arrays['episode_labels']
    if (encoded.shape != () or encoded.dtype.kind != 'U' or labels.shape != (32,) or
            labels.dtype.kind != 'U' or labels.tolist() != list(EPISODE_LABELS)):
        _fail('wrong metadata encoding or duplicate source episode labels')
    metadata = json.loads(encoded.item(), object_pairs_hook=_json_object,
                          parse_constant=_invalid_json_number)
    _keys(metadata, ('schema', 'hash_bits', 'digest_encoding', 'source_task',
        'source_endpoint_sha256', 'source_checkpoint_sha256', 'source_model_state_sha256',
        'identity', 'building_protocol', 'builder', 'counts', 'source_regression',
        'limitation', 'effectiveness_verified', 'dr_unlocked'), 'metadata')
    if (metadata['schema'] != SCHEMA or type(metadata['hash_bits']) is not int or
            metadata['hash_bits'] != 256 or metadata['digest_encoding'] != 'uint8[N,32]' or
            metadata['source_task'] != SOURCE_TASK or metadata['source_endpoint_sha256'] != SOURCE_ENDPOINT_SHA or
            metadata['source_checkpoint_sha256'] != SOURCE_CHECKPOINT_SHA or
            metadata['source_model_state_sha256'] != SOURCE_MODEL_SHA or
            metadata['building_protocol'] != PROTOCOL or metadata['limitation'] != LIMITATION or
            metadata['effectiveness_verified'] is not False or metadata['dr_unlocked'] is not False):
        _fail('changed source/protocol/exact256bit metadata')
    _identity(metadata['identity'], identity)
    _keys(metadata['builder'], ('module', 'collector_file_sha256'), 'builder')
    if metadata['builder']['module'] != 'humanoid.scripts.collect_amp_head_cohort.source_exclusions':
        _fail('wrong CPU reconstruction builder')
    _sha(metadata['builder']['collector_file_sha256'], 'builder collector source')
    _validate_proof(metadata['source_regression'])
    index = {}
    for kind in KINDS:
        hashes, offsets, ids = (arrays[kind+suffix] for suffix in
                                ('_sha256', '_label_offsets', '_label_indices'))
        if (hashes.dtype != np.uint8 or hashes.ndim != 2 or hashes.shape[1] != 32 or
                not len(hashes) or offsets.dtype != np.int64 or offsets.shape != (len(hashes)+1,) or
                ids.dtype != np.uint8 or ids.ndim != 1 or offsets[0] != 0 or
                offsets[-1] != len(ids) or np.any(np.diff(offsets) <= 0) or
                np.any(ids >= len(EPISODE_LABELS))):
            _fail('malformed exact256bit/CSR '+kind+' arrays')
        digests = [row.tobytes().hex() for row in hashes]
        if digests != sorted(set(digests)):
            _fail('duplicate/noncanonical '+kind+' digest rows')
        index[kind] = {digest: [EPISODE_LABELS[int(i)] for i in ids[offsets[n]:offsets[n+1]]]
                       for n, digest in enumerate(digests)}
    counts = _validate_index(index)
    _keys(metadata['counts'], KINDS, 'counts')
    for kind in KINDS:
        _keys(metadata['counts'][kind], ('unique_hashes', 'label_associations'), kind+' counts')
        if any(type(value) is not int or value < 1 for value in metadata['counts'][kind].values()):
            _fail('noninteger '+kind+' counts')
    if metadata['counts'] != counts:
        _fail('declared counts do not match exact digest/label arrays')
    regression = copy.deepcopy(metadata['source_regression'])
    proof = dict(source_environment=regression.pop('source_environment'),
                 source_runtime=regression.pop('source_runtime'), source_regression=regression)
    proof['derived_index'] = dict(path=str(Path(path)), schema=SCHEMA,
        sha256=expected_file_sha, bytes=len(blob), hash_bits=256, counts=counts,
        building_protocol=PROTOCOL, builder=copy.deepcopy(metadata['builder']))
    return index, proof
