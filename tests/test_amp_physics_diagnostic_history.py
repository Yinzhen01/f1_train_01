"""History bootstrap changes only Git ancestry and preserves the native diagnostic."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.amp import prepare_physics_diagnostic_history as bootstrap


HEAD, SMOKE = 'a'*40, 'b'*40
REPO = Path(__file__).resolve().parents[1]
ARGV = ['--mode=full', '--expected-commit='+HEAD,
        '--task=f1_amp_physics_diagnostic', '--smoke-certificate='+str(bootstrap.CERTIFICATE)]


class GitFixture:
    def __init__(self, ancestor=True, shallow=False, fetch_success=True,
                 proved_after_fetch=True, changed_head=False, dirty=False, untracked=False):
        self.ancestor = ancestor
        self.shallow = shallow
        self.fetch_success = fetch_success
        self.proved_after_fetch = proved_after_fetch
        self.changed_head = changed_head
        self.dirty = dirty
        self.untracked = untracked
        self.fetched = False
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[0] != 'git':
            return subprocess.CompletedProcess(command, 0)
        arguments = command[1:]
        output, code = '', 0
        if arguments == ['rev-parse', '--show-toplevel']:
            output = str(REPO)
        elif arguments == ['rev-parse', 'HEAD']:
            output = 'c'*40 if self.fetched and self.changed_head else HEAD
        elif arguments == ['rev-parse', '--is-shallow-repository']:
            output = 'true' if self.shallow else 'false'
        elif arguments[:2] == ['merge-base', '--is-ancestor']:
            code = 0 if (self.proved_after_fetch if self.fetched else self.ancestor) else 128
        elif arguments[0] == 'fetch':
            self.fetched = True
            code = 0 if self.fetch_success else 1
        elif arguments[0] == 'status':
            output = ' M humanoid/source.py' if self.dirty else ''
        elif arguments[0] == 'ls-files':
            output = 'configs/untracked.json' if self.untracked else ''
        else:
            raise AssertionError('Unexpected Git command')
        return subprocess.CompletedProcess(command, code, output, 'secret transport detail')


class DiagnosticHistoryTests(unittest.TestCase):
    def prepare(self, fixture, certificate=None, argv=None):
        args = bootstrap.parse_request(ARGV if argv is None else argv)
        with patch.object(bootstrap.subprocess, 'run', side_effect=fixture), \
                patch.object(Path, 'read_text', return_value=json.dumps(
                    dict(code_commit=SMOKE) if certificate is None else certificate)):
            return bootstrap.prepare_history(REPO, args)

    def test_valid_ancestor_never_fetches(self):
        fixture = GitFixture()
        proof = self.prepare(fixture)
        self.assertEqual(proof, dict(expected_commit=HEAD, smoke_commit=SMOKE,
                                     fetched_history=False, ancestor_verified=True))
        self.assertFalse(any(command[1] == 'fetch' for command, _ in fixture.calls))

    def test_full_repository_nonancestor_rejected_without_fetch(self):
        fixture = GitFixture(ancestor=False)
        with self.assertRaisesRegex(ValueError, 'not an ancestor'):
            self.prepare(fixture)
        self.assertFalse(fixture.fetched)

    def test_initial_root_or_head_mismatch_rejected_before_fetch(self):
        for arguments, wrong_value in ((['rev-parse', '--show-toplevel'], str(REPO/'other')),
                                       (['rev-parse', 'HEAD'], 'c'*40)):
            fixture = GitFixture(ancestor=False, shallow=True)

            def mismatched(command, **kwargs):
                result = fixture(command, **kwargs)
                if command[1:] == arguments:
                    result.stdout = wrong_value
                return result

            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, 'root or HEAD'):
                self.prepare(mismatched)
            self.assertFalse(fixture.fetched)

    def test_shallow_fetch_is_fixed_and_rechecks_head_cleanliness_and_ancestry(self):
        fixture = GitFixture(ancestor=False, shallow=True)
        proof = self.prepare(fixture)
        self.assertTrue(proof['fetched_history'])
        fetches = [(command, options) for command, options in fixture.calls if command[1] == 'fetch']
        self.assertEqual(len(fetches), 1)
        command, options = fetches[0]
        self.assertEqual(command, ['git', 'fetch', '--no-tags', '--depth=64', bootstrap.REMOTE, HEAD])
        self.assertEqual(options['env']['GIT_TERMINAL_PROMPT'], '0')
        self.assertEqual(options['timeout'], 90)
        self.assertTrue(options['capture_output'])
        self.assertFalse(options['check'])
        self.assertEqual(sum(command[1:] == ['rev-parse', 'HEAD'] for command, _ in fixture.calls), 2)
        self.assertEqual(sum(command[1:3] == ['merge-base', '--is-ancestor']
                             for command, _ in fixture.calls), 2)
        self.assertTrue(any(command[1] == 'status' for command, _ in fixture.calls))
        self.assertTrue(any(command[1] == 'ls-files' for command, _ in fixture.calls))
        self.assertFalse(any(command[1] in ('checkout', 'reset') for command, _ in fixture.calls))

    def test_fetch_failure_never_exposes_transport_output(self):
        fixture = GitFixture(ancestor=False, shallow=True, fetch_success=False)
        with self.assertRaisesRegex(ValueError, 'Git operation failed') as caught:
            self.prepare(fixture)
        self.assertNotIn('secret', str(caught.exception))

    def test_still_missing_ancestry_and_changed_head_are_rejected(self):
        for options in (dict(proved_after_fetch=False), dict(changed_head=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(GitFixture(ancestor=False, shallow=True, **options))

    def test_dirty_or_untracked_source_is_rejected(self):
        for options in (dict(dirty=True), dict(untracked=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(GitFixture(**options))

    def test_invalid_commit_groups_certificate_and_cli_rejected(self):
        for argv in (['--mode=full', '--expected-commit=abc', '--smoke-certificate='+str(bootstrap.CERTIFICATE)],
                     ARGV+['--groups=original,original'], ARGV+['--groups=unknown'],
                     ARGV+['--groups= original']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                bootstrap.parse_request(argv)
        for argv in (ARGV+['--mode=smoke'], ARGV+['--task=another_task'],
                     ARGV+['--unexpected=true'], ARGV[:-1]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                bootstrap.parse_request(argv)
        for certificate in ({}, dict(code_commit='abc'), dict(code_commit=17)):
            with self.subTest(certificate=certificate), self.assertRaises(ValueError):
                self.prepare(GitFixture(), certificate)
        with self.assertRaisesRegex(ValueError, 'Only the committed'):
            self.prepare(GitFixture(), argv=ARGV[:-1]+['--smoke-certificate=other.json'])

    def test_git_timeout_rejected_without_transport_output(self):
        with patch.object(bootstrap.subprocess, 'run', side_effect=subprocess.TimeoutExpired(
                'git', 90, output='secret', stderr='secret')):
            with self.assertRaisesRegex(ValueError, 'timed out') as caught:
                bootstrap.git(REPO, 'fetch')
        self.assertNotIn('secret', str(caught.exception))

    def test_main_preserves_original_argv_and_native_failure(self):
        fixture = GitFixture(ancestor=False, shallow=True)
        output = io.StringIO()
        with patch.object(bootstrap, 'ROOT', REPO), \
                patch.object(bootstrap.subprocess, 'run', side_effect=fixture), \
                patch.object(Path, 'read_text', return_value=json.dumps(dict(code_commit=SMOKE))), \
                contextlib.redirect_stdout(output):
            bootstrap.main(ARGV)
        command, options = fixture.calls[-1]
        self.assertEqual(command, [sys.executable, str(REPO/'humanoid/scripts/diagnose_amp_physics.py')]+ARGV)
        self.assertEqual(options, dict(cwd=str(REPO), check=True))
        marker = output.getvalue()
        self.assertTrue(marker.startswith('[amp-physics-history-bootstrap] '))
        self.assertNotIn('secret', marker)
        self.assertNotIn('https://', marker)
        with patch.object(bootstrap, 'prepare_history', return_value={}), \
                patch.object(bootstrap.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'native')), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            bootstrap.main(ARGV)


if __name__ == '__main__':
    unittest.main()
