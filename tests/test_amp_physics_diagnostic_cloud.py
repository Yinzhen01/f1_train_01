"""Independent cloud binding rejects stale, missing and corrupt execution evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from humanoid.amp.physics_diagnostic import GROUPS, SOURCE_CONFIG, SOURCE_SHA, diagnostic_contract
from tools.amp import verify_physics_diagnostic_cloud as audit


COMMIT, TASK, FINGERPRINT = 'a'*40, 'TASK_20261002_099', 'b'*64


def fixture():
    records, hashes, artifacts = {}, {}, {}
    for group in GROUPS:
        record = dict(code_commit=COMMIT, checkpoint_sha256=SOURCE_SHA,
            physics_diagnostic=diagnostic_contract(group), implementation_fingerprint=FINGERPRINT,
            identity={'experiment': 'source'}, num_envs=2, duration_s=1., diagnostic_only=True,
            policy_deterministic=True, dr_unlocked=False, effectiveness_verified=False)
        digest = hashlib.sha256(group.encode()).hexdigest()
        records[group], hashes[group] = record, digest
        artifacts[group] = dict(file='model_%d.pt' % audit.ARTIFACT_IDS[group], sha256=digest, manifest=record)
    manifest = dict(mode='smoke', code_commit=COMMIT, source_checkpoint_sha256=SOURCE_SHA,
        source_completed_updates=2500, source_task='TASK_20260926_110', sdk_task='f1_amp_physics_diagnostic',
        source_config=SOURCE_CONFIG,
        num_envs=2, duration_s=1., seed=5, groups=list(GROUPS), group_artifacts=artifacts,
        identity={'experiment': 'source'}, no_training=True, policy_updates=0, discriminator_updates=0,
        estimator_updates=0, effectiveness_verified=False, dr_unlocked=False)
    completion = dict(complete=True, mode='smoke', groups=list(GROUPS),
        source_checkpoint_sha256=SOURCE_SHA, source_completed_updates=2500,
        group_artifacts_verified=True, all_finite=True, no_training=True,
        policy_updates=0, discriminator_updates=0, estimator_updates=0,
        effectiveness_verified=False, dr_unlocked=False)
    certificate = dict(verified=True, no_training=True, all_finite=True, raw_verified=True,
        readback_verified=True, reset_inputs_exact=True, code_commit=COMMIT,
        implementation_fingerprint=FINGERPRINT, source_checkpoint_sha256=SOURCE_SHA,
        num_envs=2, duration_s=1., groups=list(GROUPS),
        group_artifacts={g:dict(sha256=hashes[g]) for g in GROUPS},
        cloud_execution_verified=False, effectiveness_verified=False, dr_unlocked=False)
    status = dict(success=True, data=dict(taskBaseInfo=dict(taskId=TASK, taskStatus=5,
        userId=4409, goodsId='ESKU000001', imageVersion='V000124')))
    start = copy.deepcopy(manifest); start['group_artifacts'] = {}
    text = audit.START+json.dumps(start)+'\n'
    for group in GROUPS:
        text += audit.GROUP+json.dumps(dict(group=group, artifact=artifacts[group]['file'], sha256=hashes[group]))+'\n'
    text += audit.COMPLETE+json.dumps(completion)+'\n[SDK][INFO] uploaded successfully\n'
    return certificate, manifest, completion, status, text, records, hashes


def verify(values):
    return audit.bind_evidence(*values, COMMIT, TASK, FINGERPRINT)


class PhysicsDiagnosticCloudTests(unittest.TestCase):
    def test_valid_evidence_is_bound_without_mutating_input_or_promoting_effectiveness(self):
        values = fixture(); before = copy.deepcopy(values)
        result = verify(values)
        self.assertTrue(result['cloud_execution_verified'])
        self.assertEqual(result['cloud_task_id'], TASK)
        self.assertFalse(result['effectiveness_verified'])
        self.assertFalse(result['dr_unlocked'])
        self.assertEqual(values, before)
        self.assertTrue(result['cloud_evidence']['logs_clean'])

    def test_wrong_task_status_user_machine_image_and_unsuccessful_api_rejected(self):
        for field, value in (('taskId', 'TASK_20261002_100'), ('taskStatus', 3), ('userId', 999),
                             ('goodsId', 'ESKU000002'), ('imageVersion', 'V000001')):
            values = list(fixture()); values[3]['data']['taskBaseInfo'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): verify(values)
        values = list(fixture()); values[3]['success'] = False
        with self.assertRaises(ValueError): verify(values)

    def test_changed_identity_training_and_stale_raw_certificate_fail_closed(self):
        for which, field, value in ((0, 'cloud_execution_verified', True), (0, 'raw_verified', False),
            (0, 'implementation_fingerprint', 'c'*64), (1, 'code_commit', 'c'*40),
            (1, 'mode', 'full'), (1, 'seed', 6), (2, 'complete', False),
            (2, 'source_checkpoint_sha256', 'c'*64), (1, 'policy_updates', 1),
            (2, 'estimator_updates', 1), (1, 'dr_unlocked', True), (2, 'effectiveness_verified', True)):
            values = list(fixture()); values[which][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): verify(values)

    def test_artifact_hash_name_record_and_group_membership_must_agree(self):
        for mutation in ('hash', 'certificate_hash', 'file', 'record', 'missing', 'groups'):
            values = list(fixture())
            if mutation == 'hash': values[6]['original'] = 'c'*64
            elif mutation == 'certificate_hash': values[0]['group_artifacts']['original']['sha256'] = 'c'*64
            elif mutation == 'file': values[1]['group_artifacts']['original']['file'] = '../source.pt'
            elif mutation == 'record': values[5]['original'] = dict(values[5]['original'], code_commit='c'*40)
            elif mutation == 'missing': values[6].pop('selfoff')
            else: values[1]['groups'] = ['original', 'original', 'selfoff']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): verify(values)

    def test_markers_require_exact_objects_order_and_no_duplicates(self):
        values = fixture()
        for text in (values[4].replace(audit.START, '[missing] '),
                     values[4]+audit.COMPLETE+json.dumps(values[2]),
                     values[4].replace('"artifact": "model_8802501.pt"', '"artifact": "wrong.pt"'),
                     values[4].replace(audit.GROUP, '[missing] ', 1),
                     values[4].replace('"complete": true', '"complete": false'),
                     audit.COMPLETE+json.dumps(values[2])+values[4].split(audit.COMPLETE)[0]):
            bad = list(fixture()); bad[4] = text
            with self.subTest(text=text[:80]), self.assertRaises(ValueError): verify(bad)
        # SDK concatenation is accepted without trimming or editing the log.
        good = list(fixture()); good[4] = good[4].replace('\n', '[SDK] following output\n')
        self.assertTrue(verify(good)['cloud_execution_verified'])

    def test_training_oom_non_sdk_traceback_and_failed_wrapper_rejected(self):
        for tail in ('[f1-amp-update] {"iteration":1}\n', 'Learning iteration 1/10\n',
            'CUDA out of memory\n', 'Nonfinite policy\n', 'RuntimeError: solver failed\n',
            '[SDK][ERROR] upload failed\n', 'RL task is failed.\n', 'OSError: disk failure\n',
            'ConnectionResetError: [Errno 104] Connection reset by peer\n',
            'Traceback (most recent call last):\n  File "/repo/train.py", line 1\n    train()\nRuntimeError: failed\n',
            'Traceback (most recent call last):\ntruncated\n'):
            values = list(fixture()); values[4] += tail
            with self.subTest(tail=tail), self.assertRaises(ValueError): verify(values)
        sdk = ('Traceback (most recent call last):\n'
               '  File "/env/site-packages/pika/adapters/utils/io_services_utils.py", line 110\n'
               '    callback()\nConnectionResetError: [Errno 104] Connection reset by peer\n')
        values = list(fixture()); values[4] += sdk
        result = verify(values)
        self.assertEqual(result['cloud_evidence']['sdk_connection_reset_tracebacks'], 1)
        self.assertFalse(result['cloud_evidence']['logs_clean'])

    def test_binary_is_rejected_by_default_and_only_completed_upload_tail_can_escape(self):
        values = fixture(); raw = values[4].encode()+b'\0\xb0\x05'
        diagnostics = {}
        escaped = audit.decode_post_diagnostic_log(raw, diagnostics)
        self.assertIn(r'\x00\xb0\x05', escaped)
        self.assertTrue(diagnostics['escaped_post_completion_binary'])
        bad = list(values); bad[4] = escaped+'\nRuntimeError: still visible\n'
        with self.assertRaises(ValueError): verify(bad)
        for data in (b'\0'+values[4].encode(), raw.replace(b'uploaded successfully', b'upload pending'),
                     raw.replace(b'"policy_updates": 0', b'"policy_updates": 1', 1),
                     raw.replace(b'"mode": "smoke"', b'"mode": "full"', 1)):
            with self.assertRaises((ValueError, UnicodeDecodeError)): audit.decode_post_diagnostic_log(data, {})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cloud.log'; path.write_bytes(raw)
            with self.assertRaises((ValueError, UnicodeDecodeError)): audit.read_log(path)
            self.assertEqual(audit.read_log(path, True), escaped)
            path = Path(folder)/'cloud.json'
            path.write_text(json.dumps(dict(success=True, data=values[4])), encoding='utf-8')
            self.assertEqual(audit.read_log(path), values[4])

    def test_main_writes_new_certificate_only_after_all_evidence_and_never_overwrites(self):
        values = fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cert, batch, status, logs, output = [folder/name for name in
                ('raw.json', 'model_amp_manifest.pt', 'status.json', 'cloud.log', 'bound.json')]
            cert.write_text(json.dumps(values[0]), encoding='utf-8')
            status.write_text(json.dumps(values[3]), encoding='utf-8')
            logs.write_text(values[4], encoding='utf-8')
            batch.write_bytes(b'batch fixture')
            for group in GROUPS: (folder/values[1]['group_artifacts'][group]['file']).write_bytes(group.encode())
            packed = dict(manifest_json=json.dumps(values[1]), certificate_json=json.dumps(values[2]))
            argv = ['verify', '--certificate', str(cert), '--batch', str(batch), '--folder', str(folder),
                '--status', str(status), '--logs', str(logs), '--output', str(output),
                '--expected-commit', COMMIT, '--task-id', TASK]
            with patch.object(audit.sys, 'argv', argv), patch.object(torch, 'load', return_value=packed), \
                    patch.object(audit, 'load_bundle', side_effect=[(values[5][g], {}) for g in GROUPS]), \
                    patch.object(audit, 'implementation_fingerprint', return_value=FINGERPRINT), \
                    patch('builtins.print'):
                audit.main()
                with self.assertRaisesRegex(ValueError, 'already exists'): audit.main()
            result = json.loads(output.read_text())
            self.assertTrue(result['cloud_execution_verified'])
            self.assertEqual(result['cloud_evidence']['batch_sha256'], hashlib.sha256(batch.read_bytes()).hexdigest())
            self.assertFalse(json.loads(cert.read_text())['cloud_execution_verified'])


if __name__ == '__main__':
    unittest.main()
