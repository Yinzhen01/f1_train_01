"""Mocked ancestry recovery protocol, not native, cloud, or physical evidence.

No git process is actually started: every response below is synthetic.  These
checks ensure the bounded shallow-history repair never bypasses provenance or
fetches the certificate's commit directly, and keeps remote credentials private.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'humanoid/scripts/run_amp_head_smooth.py'
COMMIT = 'a' * 40
HEAD = 'b' * 40
MERGE = ('git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD')
OBJECT = ('git', 'rev-parse', '--verify', '--quiet', COMMIT + '^{commit}')
SHALLOW = ('git', 'rev-parse', '--is-shallow-repository')
CURRENT = ('git', 'rev-parse', 'HEAD')
REMOTE = ('git', 'remote', 'get-url', 'origin')
TOP = ('git', 'rev-parse', '--show-toplevel')
TRACKED = ('git', 'status', '--porcelain', '--untracked-files=no')
UNTRACKED = ('git', 'ls-files', '--others', '--exclude-standard', '--',
             'humanoid', 'configs', 'resources')
FETCH = ('git', 'fetch', '--no-tags', '--deepen=64', 'origin', HEAD)


def load_driver():
    spec = importlib.util.spec_from_file_location('head_ancestry_protocol', DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def response(returncode=0, stdout=b'', stderr=b''):
    return (returncode, stdout, stderr)


class SmokeAncestryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        # A missing implementation must not be disguised as a rejected case.
        if not callable(getattr(cls.driver, 'ensure_smoke_ancestry', None)):
            raise AssertionError('The ancestry helper must exist before running these tests')

    def invoke(self, *, merge=None, changes=None, commit=COMMIT, failure=None, sequences=None):
        """Return actual mock calls/result or an exception with captured streams."""
        routes = {
            OBJECT: response(1),
            SHALLOW: response(0, b'true\n'),
            CURRENT: response(0, (HEAD + '\n').encode()),
            REMOTE: response(0, b'https://github.com/Yinzhen01/f1_train_01.git\n'),
            TOP: response(0, (str(ROOT) + '\n').encode()),
            TRACKED: response(),
            UNTRACKED: response(),
            FETCH: response(),
        }
        routes.update(changes or {})
        sequences = {command: list(rows) for command, rows in (sequences or {}).items()}
        merges = list(merge if merge is not None else [response(128), response()])
        out, err = io.StringIO(), io.StringIO()

        def run(args, **kwargs):
            command = tuple(args)
            if failure is not None and command == failure[0]:
                raise failure[1]
            if command == MERGE:
                if not merges:
                    raise AssertionError('The original ancestry command was retried unexpectedly')
                rc, stdout, stderr = merges.pop(0)
            else:
                if command not in routes:
                    raise AssertionError('Unexpected mocked ancestry command: ' + repr(command))
                if command in sequences:
                    if not sequences[command]:
                        raise AssertionError('An ancestry metadata command was retried unexpectedly')
                    rc, stdout, stderr = sequences[command].pop(0)
                else:
                    rc, stdout, stderr = routes[command]
            return subprocess.CompletedProcess(args, rc, stdout, stderr)

        with patch.object(self.driver.subprocess, 'run', side_effect=run) as mocked, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                result = self.driver.ensure_smoke_ancestry(ROOT, commit)
                caught = None
            except Exception as exception:
                result, caught = None, exception
        return result, caught, mocked.call_args_list, out.getvalue(), err.getvalue()

    def assert_rejected(self, **kwargs):
        result, caught, calls, out, err = self.invoke(**kwargs)
        self.assertIsNone(result)
        self.assertIsInstance(caught, ValueError)
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        return caught, calls

    def commands(self, calls):
        return [tuple(call.args[0]) for call in calls]

    def fetches(self, calls):
        return [call for call in calls if tuple(call.args[0])[:2] == ('git', 'fetch')]

    def assert_captured(self, call):
        self.assertEqual(call.kwargs['cwd'], str(ROOT))
        self.assertIs(call.kwargs['stdout'], subprocess.PIPE)
        self.assertIs(call.kwargs['stderr'], subprocess.PIPE)
        self.assertIs(call.kwargs['check'], False)
        self.assertEqual(call.kwargs['timeout'], 60)
        self.assertEqual(call.kwargs['env']['GIT_TERMINAL_PROMPT'], '0')
        self.assertEqual(call.kwargs['env']['GIT_ALLOW_PROTOCOL'], 'https')

    def test_verified_ancestor_returns_without_fetch_or_remote_probe(self):
        result, caught, calls, out, err = self.invoke(merge=[response()])
        self.assertIsNone(caught)
        self.assertEqual(result, dict(ancestor_verified=True, history_refreshed=False,
            required_commit=COMMIT, head_commit=HEAD))
        commands = self.commands(calls)
        self.assertEqual(commands[0], MERGE)
        self.assertNotIn(FETCH, commands)
        self.assertNotIn(REMOTE, commands)
        self.assertNotIn(SHALLOW, commands)
        self.assertEqual(out + err, '')
        for call in calls:
            self.assert_captured(call)

    def test_commit_must_be_exact_lowercase_40_hex_before_any_process(self):
        for commit in (None, b'a' * 40, '', 'HEAD', 'a' * 39, 'a' * 41,
                       'A' * 40, 'g' * 40, COMMIT + '\n', '--help'):
            with self.subTest(commit=commit):
                _, calls = self.assert_rejected(commit=commit)
                self.assertEqual(calls, [])

    def test_real_nonancestor_rejects_without_repair_even_if_shallow(self):
        _, calls = self.assert_rejected(merge=[response(1)])
        self.assertEqual(self.commands(calls), [MERGE])

    def test_unknown_ancestry_errors_do_not_attempt_fetch(self):
        for rc in (2, 127, 129, -9):
            with self.subTest(rc=rc):
                _, calls = self.assert_rejected(merge=[response(rc)])
                self.assertEqual(self.commands(calls), [MERGE])

    def test_only_rc128_and_unresolved_object_can_enter_shallow_repair(self):
        for object_response in (response(0, COMMIT.encode()), response(1, COMMIT.encode()),
                                response(2), response(128)):
            with self.subTest(object_response=object_response):
                _, calls = self.assert_rejected(changes={OBJECT: object_response})
                self.assertEqual(self.commands(calls)[0], MERGE)
                self.assertIn(OBJECT, self.commands(calls))
                self.assertEqual(self.fetches(calls), [])

    def test_shallow_repository_probe_must_succeed_and_be_exact_true(self):
        for shallow in (response(0, b'false\n'), response(0, b'True\n'),
                        response(0, b'true false\n'), response(0), response(1, b'true\n')):
            with self.subTest(shallow=shallow):
                _, calls = self.assert_rejected(changes={SHALLOW: shallow})
                self.assertEqual(self.fetches(calls), [])

    def test_head_probe_requires_successful_exact_40_hex_binding(self):
        for head in (response(1, HEAD.encode()), response(), response(0, b'HEAD\n'),
                     response(0, b'B' * 40), response(0, b'b' * 39),
                     response(0, (HEAD + '\n' + HEAD).encode())):
            with self.subTest(head=head):
                _, calls = self.assert_rejected(changes={CURRENT: head})
                self.assertEqual(self.fetches(calls), [])

    def test_single_bounded_fetch_targets_current_head_not_certificate(self):
        result, caught, calls, out, err = self.invoke()
        self.assertIsNone(caught)
        self.assertEqual(result, dict(ancestor_verified=True, history_refreshed=True,
            required_commit=COMMIT, head_commit=HEAD))
        commands = self.commands(calls)
        self.assertEqual(commands[0], MERGE)
        self.assertEqual(commands[-1], MERGE)
        self.assertEqual(commands.count(MERGE), 2)
        fetches = self.fetches(calls)
        self.assertEqual(len(fetches), 1)
        self.assertEqual(tuple(fetches[0].args[0]), FETCH)
        self.assertNotEqual(FETCH[-1], COMMIT)
        self.assertEqual(fetches[0].kwargs['timeout'], 60)
        self.assertEqual(fetches[0].kwargs['env']['GIT_TERMINAL_PROMPT'], '0')
        self.assertEqual(fetches[0].kwargs['env']['GIT_ALLOW_PROTOCOL'], 'https')
        self.assertEqual(out + err, '')
        for call in calls:
            self.assert_captured(call)
        self.assertTrue(all(command[:2] not in (('git', 'checkout'), ('git', 'reset'),
            ('git', 'update-ref'), ('git', 'pull'), ('git', 'merge')) for command in commands))

    def test_compatible_fetch_keeps_exact_single_bounded_command_without_optional_flag(self):
        # Synthetic command regression, not a fabricated Git 2.25/native run.
        result, caught, calls, out, err = self.invoke()
        self.assertIsNone(caught)
        self.assertTrue(result['ancestor_verified'])
        fetches = self.fetches(calls)
        self.assertEqual(len(fetches), 1)
        self.assertEqual(tuple(fetches[0].args[0]),
            ('git', 'fetch', '--no-tags', '--deepen=64', 'origin', HEAD))
        self.assertNotIn('--no-write-fetch-head', fetches[0].args[0])
        self.assertNotEqual(fetches[0].args[0][-1], COMMIT)
        self.assert_captured(fetches[0])
        self.assertEqual(self.commands(calls)[-1], MERGE)
        self.assertEqual(out + err, '')

    def test_only_trusted_https_origin_path_allows_repair(self):
        bad_urls = (
            'http://github.com/Yinzhen01/f1_train_01.git',
            'ssh://github.com/Yinzhen01/f1_train_01.git',
            'git@github.com:Yinzhen01/f1_train_01.git',
            'file:///tmp/f1_train_01.git',
            'https://github.com.evil.invalid/Yinzhen01/f1_train_01.git',
            'https://github.com/freedof/f1_train_01.git',
            'https://github.com/Yinzhen01/another_repo.git',
            'https://github.com/Yinzhen01/f1_train_01.git/',
            'https://github.com:444/Yinzhen01/f1_train_01.git',
            'https://github.com/Yinzhen01/f1_train_01.git?token=secret',
            'https://github.com/Yinzhen01/f1_train_01.git#fragment',
            'https://github.com/Yinzhen01/f1_train_01.git\nhttps://evil.invalid',
            'https://github.com:bad/Yinzhen01/f1_train_01.git',
            'https://github.\ncom/Yinzhen01/f1_train_01.git',
            'https://git\thub.com/Yinzhen01/f1_train_01.git',
        )
        for url in bad_urls:
            with self.subTest(url=url):
                _, calls = self.assert_rejected(changes={REMOTE: response(0, url.encode())})
                self.assertEqual(self.fetches(calls), [])

    def test_optional_git_suffix_and_userinfo_are_accepted_without_output(self):
        for url in ('https://github.com/Yinzhen01/f1_train_01',
                    'https://github.com:443/Yinzhen01/f1_train_01.git',
                    'https://Yinzhen01:synthetic-secret@github.com/Yinzhen01/f1_train_01.git'):
            with self.subTest(url=url):
                result, caught, calls, out, err = self.invoke(
                    changes={REMOTE: response(0, (url + '\n').encode())})
                self.assertIsNone(caught)
                self.assertTrue(result['ancestor_verified'])
                self.assertEqual(out + err, '')
                self.assertEqual(len(self.fetches(calls)), 1)

    def test_remote_probe_error_never_fetches_or_discloses_output(self):
        secret = b'https://token:synthetic-secret@github.com/Yinzhen01/f1_train_01.git'
        caught, calls = self.assert_rejected(changes={REMOTE: response(1, secret, secret)})
        self.assertNotIn('synthetic-secret', str(caught))
        self.assertEqual(self.fetches(calls), [])

    def test_fetch_failure_is_terminal_without_retry_or_diagnostic_leak(self):
        secret = b'https://token:synthetic-secret@github.com/Yinzhen01/f1_train_01.git'
        for rc in (1, 128, 129, -9):
            with self.subTest(rc=rc):
                caught, calls = self.assert_rejected(changes={FETCH: response(rc, secret, secret)})
                self.assertNotIn('synthetic-secret', str(caught))
                self.assertEqual(len(self.fetches(calls)), 1)
                self.assertEqual(self.commands(calls).count(MERGE), 1)

    def test_after_fetch_original_ancestry_must_actually_succeed(self):
        for rc in (1, 2, 128):
            with self.subTest(rc=rc):
                _, calls = self.assert_rejected(merge=[response(128), response(rc)])
                self.assertEqual(len(self.fetches(calls)), 1)
                self.assertEqual(self.commands(calls)[-1], MERGE)
                self.assertEqual(self.commands(calls).count(MERGE), 2)

    def test_subprocess_exceptions_are_sanitized_and_do_not_retry(self):
        secret = 'https://token:synthetic-secret@github.com/Yinzhen01/f1_train_01.git'
        for command in (MERGE, OBJECT, SHALLOW, CURRENT, REMOTE, TOP, TRACKED, UNTRACKED, FETCH):
            failures = (
                OSError(secret),
                subprocess.TimeoutExpired([secret], 60, output=secret.encode(), stderr=secret.encode()),
            )
            for failure in failures:
                with self.subTest(command=command, kind=type(failure).__name__):
                    caught, calls = self.assert_rejected(failure=(command, failure))
                    self.assertNotIn('synthetic-secret', str(caught))
                    self.assertLessEqual(len(self.fetches(calls)), 1)

    def test_pre_fetch_root_head_or_cleanliness_mismatch_never_fetches(self):
        changes = (
            {TOP: response(1, str(ROOT).encode())},
            {TOP: response(0, str(ROOT.parent).encode())},
            {TRACKED: response(0, b' M humanoid/scripts/run_amp_head_smooth.py\n')},
            {TRACKED: response(1)},
            {UNTRACKED: response(0, b'humanoid/unregistered.py\n')},
            {UNTRACKED: response(1)},
        )
        for change in changes:
            with self.subTest(change=change):
                _, calls = self.assert_rejected(changes=change)
                self.assertEqual(self.fetches(calls), [])
        _, calls = self.assert_rejected(sequences={CURRENT: [response(0, HEAD.encode()),
            response(0, ('c' * 40).encode())]})
        self.assertEqual(self.fetches(calls), [])

    def test_post_fetch_checkout_mutation_rejects_without_second_fetch(self):
        sequence_cases = (
            {TOP: [response(0, str(ROOT).encode()), response(0, str(ROOT.parent).encode())]},
            {CURRENT: [response(0, HEAD.encode()), response(0, HEAD.encode()),
                       response(0, ('c' * 40).encode())]},
            {TRACKED: [response(), response(0, b' M tracked.py\n')]},
            {UNTRACKED: [response(), response(0, b'configs/unregistered.json\n')]},
        )
        for sequences in sequence_cases:
            with self.subTest(sequences=sequences):
                _, calls = self.assert_rejected(sequences=sequences)
                self.assertEqual(len(self.fetches(calls)), 1)
                self.assertEqual(self.commands(calls).count(MERGE), 1)

    def test_git_readback_requires_actual_int_and_byte_streams(self):
        for bad_response in (response(True), response(0, 'true'), response(0, b'', 'stderr')):
            with self.subTest(bad_response=bad_response):
                _, calls = self.assert_rejected(merge=[bad_response])
                self.assertEqual(self.commands(calls), [MERGE])


class CertificateAncestryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()

    def certificate(self):
        return dict(schema='head_smooth_native_smoke_audit_v1', mode='smoke',
            identity={'synthetic': True}, implementation_fingerprint='e' * 64,
            native_verified=True, cohort_arrays_verified=True, source_forward_verified=True,
            head_archive_verified=True, native_recorder_verified=True,
            train_seed=305, validation_seed=505, num_envs=4, duration_s=2, solver_runs=1,
            formal_admission=False, source_checkpoint_sha256=self.driver.SOURCE_SHA,
            source_model_state_sha256=self.driver.SOURCE_MODEL_SHA,
            platform_terminal_status='5', effectiveness_verified=False, dr_unlocked=False,
            platform_task_id='TASK_20261004_999', code_commit=COMMIT,
            cloud_log_sha256='1' * 64, train_sha256='2' * 64, validation_sha256='3' * 64,
            head_sha256='4' * 64, policy_rollout_sha256='5' * 64, audit_report_sha256='6' * 64)

    def validate(self, value):
        return self.driver.validate_smoke_certificate(value, identity={'synthetic': True},
            fingerprint='e' * 64, repo=ROOT)

    def test_missing_commit_legacy128_calls_helper_and_mandatory_original_check(self):
        proof = dict(ancestor_verified=True, history_refreshed=True,
            required_commit=COMMIT, head_commit=HEAD)
        with patch.object(self.driver.subprocess, 'check_call', side_effect=[
                subprocess.CalledProcessError(128, list(MERGE)), 0]) as ancestry, \
                patch.object(self.driver, 'ensure_smoke_ancestry', return_value=proof) as helper, \
                patch.object(self.driver, '_marker') as marker:
            self.assertIsNone(self.validate(self.certificate()))
        self.assertEqual([tuple(call.args[0]) for call in ancestry.call_args_list], [MERGE, MERGE])
        helper.assert_called_once_with(ROOT, COMMIT)
        marker.assert_called_once_with('history', proof)

    def test_post_repair_legacy_check_failure_remains_rejected(self):
        with patch.object(self.driver.subprocess, 'check_call', side_effect=[
                subprocess.CalledProcessError(128, list(MERGE)),
                subprocess.CalledProcessError(1, list(MERGE))]) as ancestry, \
                patch.object(self.driver, 'ensure_smoke_ancestry', return_value={}) as helper, \
                patch.object(self.driver, '_marker') as marker, \
                self.assertRaises(subprocess.CalledProcessError):
            self.validate(self.certificate())
        self.assertEqual(ancestry.call_count, 2)
        helper.assert_called_once_with(ROOT, COMMIT)
        marker.assert_not_called()

    def test_non128_legacy_failure_never_calls_history_helper(self):
        for rc in (1, 2, 129):
            with self.subTest(rc=rc), patch.object(self.driver.subprocess, 'check_call',
                    side_effect=subprocess.CalledProcessError(rc, list(MERGE))) as ancestry, \
                    patch.object(self.driver, 'ensure_smoke_ancestry') as helper, \
                    self.assertRaises(subprocess.CalledProcessError):
                self.validate(self.certificate())
            self.assertEqual(ancestry.call_count, 1)
            helper.assert_not_called()

    def test_invalid_artifact_hashes_never_touch_git_or_prepare_history(self):
        for key in ('cloud_log_sha256', 'train_sha256', 'validation_sha256',
                    'head_sha256', 'policy_rollout_sha256', 'audit_report_sha256'):
            for bad_value in ('bad', '0' * 64):
                value = self.certificate()
                value[key] = bad_value
                with self.subTest(key=key, bad_value=bad_value), \
                        patch.object(self.driver.subprocess, 'check_call') as ancestry, \
                        patch.object(self.driver, 'ensure_smoke_ancestry') as helper, \
                        self.assertRaises(ValueError):
                    self.validate(value)
                ancestry.assert_not_called()
                helper.assert_not_called()


if __name__ == '__main__':
    unittest.main()
