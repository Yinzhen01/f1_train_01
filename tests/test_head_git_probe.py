"""Synthetic Git/transport/certificate mocks only; no cloud or native evidence.

The fresh-interpreter test exercises stdlib imports, not Git or a simulator.
No test invokes a real Git process, fetch, policy, fit, or rollout.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.amp import probe_head_git_history as probe


HEAD = 'b'*40
CURRENT = ('rev-parse', 'HEAD')
TOP = ('rev-parse', '--show-toplevel')
TRACKED = ('status', '--porcelain', '--untracked-files=no')
UNTRACKED = ('ls-files', '--others', '--exclude-standard', '--', 'humanoid', 'configs', 'resources')
MERGE = ('merge-base', '--is-ancestor', probe.SMOKE_COMMIT, 'HEAD')
FETCH = ('fetch', '--no-tags', '--deepen=64', 'origin', HEAD)
REMOTE = ('remote', 'get-url', 'origin')
SENTINEL = 'synthetic-secret-never-output'


def row(rc=0, out=b'', err=b''):
    return rc, out, err


def synthetic_certificate():
    value = dict(schema='head_smooth_native_smoke_audit_v1', mode='smoke',
        identity=dict(experiment='f1_amp_walk02_sustain_control', config_sha256='a'*64,
            motion_sha256='b'*64, urdf_lf_sha256='c'*64, feature_fingerprint='d'*64),
        implementation_fingerprint=probe.FINGERPRINT, native_verified=True,
        cohort_arrays_verified=True, source_forward_verified=True, head_archive_verified=True,
        native_recorder_verified=True, train_seed=305, validation_seed=505, num_envs=4,
        duration_s=2, solver_runs=1, formal_admission=False,
        source_checkpoint_sha256=probe.SOURCE_SHA, source_model_state_sha256=probe.SOURCE_MODEL_SHA,
        platform_task_id='TASK_20261004_087', platform_terminal_status='5',
        code_commit=probe.SMOKE_COMMIT, logs_clean=False, effectiveness_verified=False, dr_unlocked=False)
    for i,key in enumerate(('cloud_log_sha256','train_sha256','validation_sha256','head_sha256',
            'audit_report_sha256','report_artifact_sha256','platform_info_sha256','policy_rollout_sha256'),1):
        value[key]=str(i)*64
    return value


class HeadGitProbeTests(unittest.TestCase):
    def certificate_bytes(self, value=None, raw=None):
        raw = json.dumps(synthetic_certificate() if value is None else value).encode() if raw is None else raw
        with patch.object(probe, 'CERTIFICATE_SHA', hashlib.sha256(raw).hexdigest()), \
                patch.object(Path, 'read_bytes', return_value=raw):
            return probe._certificate()

    def invoke(self, *, initial=128, final=0, changes=None, sequences=None, expected=HEAD,
               failure=None, helper_error=None, main=False, real_fingerprint=False):
        driver = probe._load_driver()
        original = driver._head_history_git
        routes = {
            CURRENT: row(0,(HEAD+'\n').encode()), TOP: row(0,(str(probe.ROOT)+'\n').encode()),
            TRACKED: row(), UNTRACKED: row(), ('--version',): row(0,b'git version 2.17.1\n'),
            ('fetch','-h'): row(129,b'',b'usage: git fetch --deepen <n>\n'),
            ('rev-parse','--verify','--quiet',probe.SMOKE_COMMIT+'^{commit}'): row(1),
            ('rev-parse','--is-shallow-repository'): row(0,b'true\n'),
            REMOTE: row(0,b'https://github.com/Yinzhen01/f1_train_01.git\n'), FETCH: row(),
        }
        routes.update(changes or {})
        sequence = {k:list(v) for k,v in (sequences or {}).items()}
        merges = [row(initial),row(final)]
        def run(args, **kwargs):
            command=tuple(args[1:])
            if failure is not None and command==failure[0]:
                raise failure[1]
            if command==MERGE:
                response=merges.pop(0)
            elif command in sequence:
                response=sequence[command].pop(0)
            else:
                if command not in routes:
                    raise AssertionError('Unexpected synthetic Git command')
                response=routes[command]
            rc,out,err=response
            return subprocess.CompletedProcess(args,rc,out,err)
        out,err=io.StringIO(),io.StringIO()
        wrapped_helper=driver.ensure_smoke_ancestry
        helper_mock = patch.object(driver,'ensure_smoke_ancestry',
            side_effect=helper_error if helper_error is not None else wrapped_helper)
        fingerprint_mock = contextlib.nullcontext() if real_fingerprint else \
            patch.object(probe,'_runtime_fingerprint',return_value=probe.FINGERPRINT)
        with patch.object(probe.subprocess,'run',side_effect=run) as git, \
                patch.object(probe,'_certificate',return_value={'synthetic_only':True}), \
                fingerprint_mock, \
                patch.object(probe,'_load_driver',return_value=driver), helper_mock as helper, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                result=probe.main(['--expected-commit',expected]) if main else probe.run_probe(expected)
                error=None
            except Exception as caught:
                result=None; error=caught
        self.assertIs(driver._head_history_git,original)
        self.assertNotIn(SENTINEL,out.getvalue()+err.getvalue())
        self.assertNotIn('https://',out.getvalue()+err.getvalue())
        return result,error,git.call_args_list,helper.call_count,out.getvalue(),err.getvalue()

    def commands(self,calls):
        return [tuple(call.args[0][1:]) for call in calls]

    def transport(self,output):
        return [json.loads(line.split('] ',1)[1]) for line in output.splitlines()
                if line.startswith('[head-git-probe-transport] ')]

    def test_verified_existing_ancestor_does_not_fetch(self):
        result,error,calls,count,out,err=self.invoke(initial=0)
        self.assertIsNone(error); self.assertTrue(result['helper_succeeded'])
        self.assertEqual(count,1); self.assertEqual(result['fetch_count'],0)
        self.assertNotIn(FETCH,self.commands(calls)); self.assertNotIn(REMOTE,self.commands(calls))
        self.assertFalse(result['formal_admission']); self.assertFalse(result['effectiveness_verified'])
        self.assertFalse(result['dr_unlocked']); self.assertFalse(result['training_started'])
        self.assertEqual(result['solver_runs'],0); self.assertEqual(err,'')

    def test_original_helper_single_fetch_and_original_return_bytes(self):
        result,error,calls,count,out,err=self.invoke()
        self.assertIsNone(error); self.assertTrue(result['ancestor_verified'])
        self.assertTrue(result['history_refreshed']); self.assertEqual(count,1)
        self.assertEqual(self.commands(calls).count(FETCH),1)
        self.assertEqual(self.commands(calls).count(MERGE),2)
        self.assertEqual(result['fetch_count'],1)
        for call in calls:
            self.assertEqual(call.kwargs['timeout'],60)
            self.assertIs(call.kwargs['stdout'],subprocess.PIPE)
            self.assertIs(call.kwargs['stderr'],subprocess.PIPE)
            self.assertEqual(call.kwargs['env']['GIT_TERMINAL_PROMPT'],'0')
            self.assertEqual(call.kwargs['env']['GIT_ALLOW_PROTOCOL'],'https')
        self.assertEqual(len(self.transport(out)),1)

    def test_fetch_unknown_option_classification_not_assumed_from_version(self):
        raw=("error: unknown option 'no-write-fetch-head'\nhttps://token:"+SENTINEL+'@github.com/').encode()
        result,error,calls,count,out,err=self.invoke(changes={FETCH:row(129,b'',raw)})
        self.assertIsNone(error); self.assertFalse(result['helper_succeeded'])
        self.assertEqual(result['helper_error'],'fetch_failed'); self.assertEqual(count,1)
        self.assertEqual(self.commands(calls).count(FETCH),1)
        event=self.transport(out)[0]
        self.assertTrue(event['unsupported_no_write_fetch_head']); self.assertFalse(event['unknown'])
        self.assertEqual(event['stderr_bytes'],len(raw)); self.assertEqual(event['returncode'],129)
        self.assertFalse(result['formal_admission']); self.assertTrue(result['diagnostic_completed'])

    def test_unknown_fetch_error_preserved_as_unknown_without_retry(self):
        result,error,calls,_,out,_=self.invoke(changes={FETCH:row(128,b'',SENTINEL.encode())})
        self.assertIsNone(error); self.assertEqual(result['helper_error'],'fetch_failed')
        self.assertTrue(self.transport(out)[0]['unknown'])
        self.assertEqual(self.commands(calls).count(FETCH),1)

    def test_fixed_bool_classifier_has_multiple_pattern_presence(self):
        cases={
            'unsupported_deepen':b"error: unrecognized option 'deepen'",
            'auth_missing':b'could not read Username: terminal prompts disabled',
            'auth_failed':b'Authentication failed', 'dns':b'Could not resolve host',
            'tls':b'SSL certificate problem', 'timeout':b'Operation timed out',
            'missing_remote_object':b'not our ref',
        }
        for key,raw in cases.items():
            with self.subTest(key=key):
                flags=probe.classify_stderr(raw,128)
                self.assertTrue(flags[key]); self.assertFalse(flags['unknown'])
                self.assertTrue(all(type(value) is bool for value in flags.values()))
        flags=probe.classify_stderr(b'Authentication failed; SSL error',128)
        self.assertTrue(flags['auth_failed']); self.assertTrue(flags['tls'])

    def test_classifier_rejects_coerced_types(self):
        for stderr,code in (('bad',1),(b'bad',True),(b'bad',1.0)):
            with self.subTest(code_type=type(code).__name__), self.assertRaises(probe.ProbeRejected):
                probe.classify_stderr(stderr,code)

    def test_git_help_is_metadata_not_a_network_fetch_or_compatibility_pass(self):
        result,error,calls,_,out,_=self.invoke(initial=0)
        self.assertIsNone(error)
        caps=[json.loads(line.split('] ',1)[1]) for line in out.splitlines()
              if line.startswith('[head-git-probe-capabilities] ')][0]
        self.assertEqual(caps['git_version'],'2.17.1'); self.assertFalse(caps['help_no_write_fetch_head'])
        self.assertTrue(caps['help_deepen']); self.assertEqual(result['fetch_count'],0)
        self.assertEqual(self.commands(calls).count(('fetch','-h')),1)

    def test_bracketed_negatable_git_help_option_is_recognized(self):
        _,error,_,_,out,_=self.invoke(initial=0,changes={('fetch','-h'):
            row(129,b'',b'--[no-]write-fetch-head --deepen <n>')})
        self.assertIsNone(error)
        self.assertIn('"help_no_write_fetch_head": true',out)

    def test_unparseable_version_never_echoes_the_actual_output(self):
        _,error,_,_,out,_=self.invoke(initial=0,changes={('--version',):
            row(0,('git version '+SENTINEL).encode())})
        self.assertIsNone(error); self.assertIn('"git_version": null',out)

    def test_bad_expected_commit_before_any_git(self):
        for expected in ('HEAD','A'*40,'a'*39,'a'*40+'\n','--help'):
            with self.subTest(expected_type=type(expected).__name__):
                result,error,calls,count,_,_=self.invoke(expected=expected)
                self.assertIsNone(result); self.assertIsInstance(error,probe.ProbeRejected)
                self.assertEqual(calls,[]); self.assertEqual(count,0)

    def test_historical_probe_rejects_changed_runtime_fingerprint_before_helper_or_fetch(self):
        # Only Git/certificate responses are synthetic.  The pure hash reads
        # current files, so the original087 probe must reject the newer driver.
        result,error,calls,count,out,err=self.invoke(real_fingerprint=True)
        self.assertIsNone(result)
        self.assertIsInstance(error,probe.ProbeRejected)
        self.assertEqual(error.args,('fingerprint_mismatch',))
        self.assertEqual(count,0)
        self.assertNotIn(FETCH,self.commands(calls))
        self.assertNotIn(('--version',),self.commands(calls))
        self.assertEqual(out+err,'')

    def test_dirty_or_wrong_checkout_has_no_helper_or_fetch(self):
        cases=({TOP:row(0,str(probe.ROOT.parent).encode())},{CURRENT:row(0,b'a'*40)},
            {TRACKED:row(0,b' M docs/state.md')},{UNTRACKED:row(0,b'configs/unknown.json')},
            {TRACKED:row(1)})
        for changes in cases:
            with self.subTest(changed_command=list(changes)[0]):
                _,error,calls,count,_,_=self.invoke(changes=changes)
                self.assertIsInstance(error,probe.ProbeRejected); self.assertEqual(count,0)
                self.assertNotIn(FETCH,self.commands(calls))

    def test_post_checkout_change_is_rejected_even_if_ancestor_fastpath(self):
        _,error,calls,count,_,_=self.invoke(initial=0,sequences={CURRENT:
            [row(0,HEAD.encode()),row(0,HEAD.encode()),row(0,b'c'*40)]})
        self.assertIsInstance(error,probe.ProbeRejected); self.assertEqual(count,1)
        self.assertNotIn(FETCH,self.commands(calls))

    def test_post_fetch_dirty_tracked_or_critical_untracked_rejects_without_retry(self):
        for command in (TRACKED,UNTRACKED):
            with self.subTest(command=command):
                _,error,calls,count,_,_=self.invoke(sequences={command:
                    [row(),row(),row(),row(0,b' M critical-file')]})
                self.assertIsInstance(error,probe.ProbeRejected); self.assertEqual(count,1)
                self.assertEqual(self.commands(calls).count(FETCH),1)

    def test_missing_or_nonshallow_or_nonancestor_never_fetch(self):
        for changes,initial in (({},1),({('rev-parse','--is-shallow-repository'):row(0,b'false')},128),
                ({('rev-parse','--verify','--quiet',probe.SMOKE_COMMIT+'^{commit}'):row(0,b'yes')},128)):
            result,error,calls,count,_,_=self.invoke(changes=changes,initial=initial)
            self.assertIsNone(error); self.assertFalse(result['ancestor_verified'])
            self.assertEqual(count,1); self.assertNotIn(FETCH,self.commands(calls))

    def test_unsafe_origin_blocks_fetch_without_echo(self):
        urls=('https://token:'+SENTINEL+'@evil.invalid/Yinzhen01/f1_train_01.git',
            'https://github.com/Yinzhen01/other.git',
            'https://github.com/Yinzhen01/f1_train_01.git?token='+SENTINEL)
        for url in urls:
            result,error,calls,count,out,_=self.invoke(changes={REMOTE:row(0,url.encode())})
            self.assertIsNone(error); self.assertEqual(result['helper_error'],'untrusted_origin')
            self.assertEqual(count,1); self.assertNotIn(FETCH,self.commands(calls))

    def test_timeout_exception_is_fixed_enum_and_runner_restored(self):
        failure=subprocess.TimeoutExpired(['git',SENTINEL],60,output=SENTINEL.encode(),stderr=SENTINEL.encode())
        result,error,calls,count,_,_=self.invoke(failure=(FETCH,failure))
        self.assertIsNone(error); self.assertEqual(result['helper_error'],'git_unavailable_or_timeout')
        self.assertEqual(result['fetch_count'],1); self.assertEqual(count,1)
        self.assertEqual(self.commands(calls).count(FETCH),1)

    def test_arbitrary_helper_exception_is_not_printed_or_preserved(self):
        for failure in (ValueError(SENTINEL),RuntimeError(SENTINEL),probe.ProbeRejected(SENTINEL)):
            result,error,calls,count,out,err=self.invoke(helper_error=failure)
            self.assertIsNone(error); self.assertFalse(result['helper_succeeded']); self.assertEqual(count,1)
            self.assertNotIn(SENTINEL,json.dumps(result)); self.assertEqual(err,'')

    def test_main_diagnostic_success_is_not_training_or_effect_admission(self):
        result,error,_,_,out,_=self.invoke(main=True,changes={FETCH:row(128,b'',b'unknown')})
        self.assertIsNone(error); self.assertEqual(result,0)
        self.assertIn('"diagnostic_completed": true',out)
        self.assertIn('"helper_succeeded": false',out)
        self.assertIn('"dr_unlocked": false',out)

    def test_cli_unknown_or_missing_argument_has_no_echo_or_git(self):
        for argv in ([],['--url','https://token:'+SENTINEL+'@host/'],['--expected-com',HEAD]):
            out,err=io.StringIO(),io.StringIO()
            with patch.object(probe.subprocess,'run') as git,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                self.assertEqual(probe.main(argv),2)
            git.assert_not_called(); self.assertNotIn(SENTINEL,out.getvalue()+err.getvalue())
            self.assertEqual(err.getvalue(),'')

    def test_synthetic_certificate_fixture_is_explicitly_not_production_hash(self):
        value=self.certificate_bytes()
        self.assertEqual(len(value),31)
        self.assertNotEqual(hashlib.sha256(json.dumps(value).encode()).hexdigest(),probe.CERTIFICATE_SHA)

    def test_certificate_wrong_types_fields_or_fixed_binding_rejected(self):
        changes=({'train_seed':True},{'duration_s':2.0},{'identity':[]},{'extra':False},
            {'source_checkpoint_sha256':'f'*64},{'code_commit':'e'*40},{'implementation_fingerprint':'e'*64},
            {'schema':'wrong'},{'platform_task_id':'TASK_20261004_090'},{'logs_clean':True},
            {'train_sha256':'2'*64,'validation_sha256':'2'*64},{'head_sha256':'0'*64})
        for change in changes:
            value=synthetic_certificate(); value.update(change)
            with self.subTest(changed_key=tuple(change)),self.assertRaises(probe.ProbeRejected):
                self.certificate_bytes(value)

    def test_certificate_duplicate_and_nonfinite_json_are_rejected(self):
        raw=json.dumps(synthetic_certificate())
        for bad in ('{"schema":"other",'+raw[1:],raw.replace('"train_seed": 305','"train_seed": NaN')):
            with self.assertRaises(probe.ProbeRejected):
                self.certificate_bytes(raw=bad.encode())

    def test_actual_certificate_bytes_pin_cannot_be_bypassed_by_good_shape(self):
        with patch.object(Path,'read_bytes',return_value=json.dumps(synthetic_certificate()).encode()), \
                self.assertRaises(probe.ProbeRejected):
            probe._certificate()

    def test_certificate_symlink_escape_is_rejected_before_read(self):
        with patch.object(Path,'resolve',return_value=probe.ROOT.parent), \
                patch.object(Path,'read_bytes') as read, self.assertRaises(probe.ProbeRejected):
            probe._certificate()
        read.assert_not_called()

    def test_fresh_historical_fp_rejection_does_not_import_torch_or_isaac(self):
        code="""import builtins,sys
before=set(sys.modules)
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','isaacgym'):
        raise AssertionError('Forbidden native/policy dependency')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from tools.amp import probe_head_git_history as p
p._load_driver()
try:
    p._runtime_fingerprint()
except p.ProbeRejected as error:
    assert error.args==('fingerprint_mismatch',)
else:
    raise AssertionError('Historical probe must reject the newer runtime FP')
assert not any(name.split('.')[0] in ('torch','isaacgym') for name in set(sys.modules)-before)
"""
        # This launches only a stdlib import/hash interpreter, never Git/native.
        result=subprocess.run([sys.executable,'-c',code],cwd=str(probe.ROOT),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20,check=False)
        self.assertEqual(result.returncode,0,result.stderr.decode('utf-8'))
        self.assertEqual(result.stdout,b'')


if __name__ == '__main__':
    unittest.main()
