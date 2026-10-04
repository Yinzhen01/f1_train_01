"""Synthetic full256bit/source-label contract tests, not native evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from humanoid.amp.head_exclusions import (
    EPISODE_LABELS, KINDS, SCHEMA, SOURCE_ENDPOINT_SHA, SOURCE_TASK,
    load_exclusion_index, write_exclusion_index,
)


def synthetic_identity():
    return dict(experiment='f1_amp_walk02_sustain_control', config_sha256='1'*64,
                motion_sha256='2'*64, urdf_lf_sha256='3'*64, feature_fingerprint='4'*64)


def synthetic_index():
    index = {kind: {} for kind in KINDS}
    for kind in KINDS:
        for label in EPISODE_LABELS:
            for tick in range(2 if kind == 'ready_full_history' else 1):
                digest = hashlib.sha256(('%s/%s/%d' % (kind, label, tick)).encode()).hexdigest()
                index[kind][digest] = [label]
    return index


def synthetic_proof():
    return dict(sha256=SOURCE_ENDPOINT_SHA, source_task=SOURCE_TASK,
        regression_episode_ids=list(EPISODE_LABELS),
        initial_state='synthetic shape/dtype/value initial state',
        initial_full_observation='synthetic CPU zero65 warmup single',
        ready_full_history='synthetic CPU66 consecutive single observations',
        reconstruction_not_raw_history=True, limitation='synthetic metadata, not native evidence',
        source_environment=dict(noise=dict(add_noise=False), domain_rand={}),
        source_runtime=dict(dof_properties=dict(effort=[1.]), p_gains=[2.], d_gains=[3.],
                            control_dt=.01, physics_dt=.001))


class HeadExclusionIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'synthetic_exclusions.npz'
        self.index, self.proof, self.identity = synthetic_index(), synthetic_proof(), synthetic_identity()
        self.sha = write_exclusion_index(self.path, self.index, self.proof,
                                         identity=self.identity, builder_sha='a'*64)

    def load(self, sha=None, identity=None):
        return load_exclusion_index(self.path, expected_file_sha=self.sha if sha is None else sha,
                                    identity=self.identity if identity is None else identity)

    def rewrite(self, change):
        with np.load(self.path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        change(arrays)
        with self.path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def mutate_meta(self, change):
        def update(arrays):
            metadata = json.loads(arrays['metadata_json'].item())
            change(metadata)
            arrays['metadata_json'] = np.asarray(json.dumps(metadata), dtype=np.str_)
        return self.rewrite(update)

    def test_exact_roundtrip_complete_source_labels_and_runtime_metadata(self):
        loaded, proof = self.load()
        self.assertEqual(loaded, self.index)
        expected_regression = {key: value for key, value in self.proof.items()
                               if key not in ('source_environment', 'source_runtime')}
        self.assertEqual(proof['source_regression'], expected_regression)
        self.assertEqual(proof['source_environment'], self.proof['source_environment'])
        self.assertEqual(proof['source_runtime'], self.proof['source_runtime'])
        self.assertEqual(proof['source_regression']['sha256'], SOURCE_ENDPOINT_SHA)
        self.assertEqual(proof['derived_index']['sha256'], self.sha)
        self.assertNotEqual(self.sha, SOURCE_ENDPOINT_SHA)
        self.assertEqual(proof['derived_index']['schema'], SCHEMA)
        self.assertEqual(proof['derived_index']['hash_bits'], 256)
        with np.load(self.path, allow_pickle=False) as archive:
            for kind in KINDS:
                self.assertEqual(archive[kind+'_sha256'].dtype, np.uint8)
                self.assertEqual(archive[kind+'_sha256'].shape[1], 32)
            self.assertTrue(all(archive[key].dtype.kind != 'O' for key in archive.files))

    def test_all256_bits_preserved_and_legal_shared_source_digest_keeps_labels(self):
        index = copy.deepcopy(self.index)
        ready = index['ready_full_history']
        keys = list(ready)[:2]
        for suffix, key in zip(('00', '01'), keys):
            ready['0'*62+suffix] = ready.pop(key)
        initial = index['initial_state']
        keys = list(initial)[:2]
        initial[keys[0]].extend(initial.pop(keys[1]))
        path = Path(self.temp.name)/'synthetic_shared.npz'
        sha = write_exclusion_index(path, index, self.proof, identity=self.identity, builder_sha='a'*64)
        loaded, _ = load_exclusion_index(path, expected_file_sha=sha, identity=self.identity)
        self.assertEqual(loaded, index)
        self.assertIn('0'*64, loaded['ready_full_history'])
        self.assertIn('0'*62+'01', loaded['ready_full_history'])
        self.assertEqual(len(loaded['initial_state'][keys[0]]), 2)

    def test_independent_hash_mandatory_and_tampering_cannot_self_bind(self):
        with self.assertRaises(TypeError):
            load_exclusion_index(self.path, identity=self.identity)
        for bad in (None, '0'*64, 'short', 'A'*64):
            with self.assertRaises(ValueError):
                load_exclusion_index(self.path, expected_file_sha=bad, identity=self.identity)
        def flip(arrays):
            arrays['ready_full_history_sha256'][0, 31] ^= np.uint8(1)
        self.rewrite(flip)
        with self.assertRaisesRegex(ValueError, 'file SHA256'):
            self.load()  # Native driver MUST keep the original pinned hash.

    def test_source_endpoint_policy_identity_schema_and_false_gates_bound(self):
        changes = [lambda m: m.__setitem__('schema', 'bloom_v1'),
                   lambda m: m.__setitem__('hash_bits', 128),
                   lambda m: m.__setitem__('source_endpoint_sha256', '0'*64),
                   lambda m: m.__setitem__('source_checkpoint_sha256', '0'*64),
                   lambda m: m.__setitem__('source_model_state_sha256', '0'*64),
                   lambda m: m.__setitem__('building_protocol', 'GPU_raw'),
                   lambda m: m.__setitem__('effectiveness_verified', True),
                   lambda m: m.__setitem__('dr_unlocked', 0)]
        original = self.path.read_bytes()
        for change in changes:
            self.path.write_bytes(original)  # Runtime synthetic fixture only.
            sha = self.mutate_meta(change)
            with self.assertRaises(ValueError):
                self.load(sha=sha)
        self.path.write_bytes(original)
        identity = dict(self.identity, config_sha256='0'*64)
        with self.assertRaises(ValueError):
            self.load(identity=identity)

    def test_missing_or_extra_array_fields_rejected_even_rehashed(self):
        original = self.path.read_bytes()
        with np.load(self.path, allow_pickle=False) as archive:
            names = list(archive.files)
        for key in names:
            self.path.write_bytes(original)
            sha = self.rewrite(lambda arrays: arrays.pop(key))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(sha=sha)
        self.path.write_bytes(original)
        sha = self.rewrite(lambda arrays: arrays.__setitem__('raw_observations', np.zeros((1, 3102))))
        with self.assertRaises(ValueError):
            self.load(sha=sha)

    def test_missing_any_metadata_and_source_proof_field_rejected(self):
        original = self.path.read_bytes()
        with np.load(self.path, allow_pickle=False) as archive:
            metadata = json.loads(archive['metadata_json'].item())
        for key in metadata:
            self.path.write_bytes(original)
            sha = self.mutate_meta(lambda m: m.pop(key))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(sha=sha)
        for key in metadata['source_regression']:
            self.path.write_bytes(original)
            sha = self.mutate_meta(lambda m: m['source_regression'].pop(key))
            with self.subTest(proof=key), self.assertRaises(ValueError):
                self.load(sha=sha)

    def test_truncated_wrong_dtype_or_shape_hash_arrays_rejected(self):
        original = self.path.read_bytes()
        for kind in KINDS:
            for change in ('truncate', 'dtype', 'reshape', 'empty'):
                self.path.write_bytes(original)
                key = kind+'_sha256'
                def mutate(arrays):
                    value = arrays[key]
                    arrays[key] = (value[:, :16] if change == 'truncate' else
                                   value.astype(np.int16) if change == 'dtype' else
                                   value.reshape(-1) if change == 'reshape' else value[:0])
                sha = self.rewrite(mutate)
                with self.subTest(kind=kind, change=change), self.assertRaises(ValueError):
                    self.load(sha=sha)

    def test_duplicate_hashes_labels_and_malformed_CSR_rejected(self):
        original = self.path.read_bytes()
        changes = [lambda a: a['episode_labels'].__setitem__(1, a['episode_labels'][0]),
                   lambda a: a['initial_state_sha256'].__setitem__(1, a['initial_state_sha256'][0]),
                   lambda a: a['ready_full_history_label_offsets'].__setitem__(1, 0),
                   lambda a: a['ready_full_history_label_offsets'].__setitem__(-1, 1),
                   lambda a: a['ready_full_history_label_indices'].__setitem__(0, 32),
                   lambda a: a.__setitem__('initial_state_label_offsets',
                                          a['initial_state_label_offsets'].astype(np.int32)),
                   lambda a: a.__setitem__('initial_full_observation_label_indices',
                                          a['initial_full_observation_label_indices'].astype(np.int64))]
        for change in changes:
            self.path.write_bytes(original)
            sha = self.rewrite(change)
            with self.assertRaises(ValueError):
                self.load(sha=sha)
        self.path.write_bytes(original)
        sha = self.mutate_meta(lambda m: m['source_regression']['regression_episode_ids'].__setitem__(
            1, m['source_regression']['regression_episode_ids'][0]))
        with self.assertRaises(ValueError):
            self.load(sha=sha)

    def test_wrong_counts_nonfinite_metadata_and_duplicate_JSON_keys_rejected(self):
        original = self.path.read_bytes()
        sha = self.mutate_meta(lambda m: m['counts']['ready_full_history'].__setitem__('unique_hashes', 1))
        with self.assertRaises(ValueError):
            self.load(sha=sha)
        self.path.write_bytes(original)
        sha = self.mutate_meta(lambda m: m['counts']['initial_state'].__setitem__('unique_hashes', 32.))
        with self.assertRaises(ValueError):
            self.load(sha=sha)
        self.path.write_bytes(original)
        sha = self.mutate_meta(lambda m: m['source_regression']['source_runtime'].__setitem__('control_dt', float('nan')))
        with self.assertRaises(ValueError):
            self.load(sha=sha)
        self.path.write_bytes(original)
        def duplicate(arrays):
            text = arrays['metadata_json'].item()
            arrays['metadata_json'] = np.asarray(text[:-1]+',"schema":"'+SCHEMA+'"}', dtype=np.str_)
        sha = self.rewrite(duplicate)
        with self.assertRaises(ValueError):
            self.load(sha=sha)

    def test_object_arrays_cannot_trigger_pickle_or_accept_raw_source(self):
        sha = self.rewrite(lambda a: a.__setitem__('ready_full_history_sha256',
                                                 np.asarray([object()], dtype=object)))
        with self.assertRaises(ValueError):
            self.load(sha=sha)
        self.path.write_bytes(b'not a numpy index, not a source checkpoint')
        with self.assertRaises(ValueError):
            self.load(sha=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_writer_refuses_overwrite_and_bad_duplicate_missing_label_hash_inputs(self):
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_exclusion_index(self.path, self.index, self.proof,
                                  identity=self.identity, builder_sha='a'*64)
        self.assertEqual(self.path.read_bytes(), original)
        for change in ('short_hash', 'duplicate_label', 'missing_coverage', 'bad_label'):
            index = copy.deepcopy(self.index)
            digest = next(iter(index['initial_state']))
            if change == 'short_hash':
                index['initial_state']['a'*32] = index['initial_state'].pop(digest)
            elif change == 'duplicate_label':
                index['initial_state'][digest] *= 2
            elif change == 'missing_coverage':
                del index['initial_state'][digest]
            else:
                index['initial_state'][digest] = ['head_cohort_v2/train/seed-306/env-0/episode-0']
            with self.subTest(change=change), self.assertRaises(ValueError):
                write_exclusion_index(Path(self.temp.name)/(change+'.npz'), index, self.proof,
                                      identity=self.identity, builder_sha='a'*64)

    def test_native_import_does_not_import_torch_or_isaacgym(self):
        code = ('import sys; import humanoid.amp.head_exclusions; '
                'assert "torch" not in sys.modules; assert "isaacgym" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-c', code], check=True,
                       cwd=str(Path(__file__).resolve().parents[1]), capture_output=True)


if __name__ == '__main__':
    unittest.main()
