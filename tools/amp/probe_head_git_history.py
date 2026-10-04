"""One bounded Git-history diagnosis, not native training or an admission.

Transport streams stay in transient memory only.  The original runtime helper
decides whether a fetch is permitted; this tool never retries or supplies a
fallback remote.  All emitted records contain fixed labels and primitives.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = 'resources/amp_admission/head_native_smoke_TASK_20261004_087.json'
CERTIFICATE_SHA = 'c21a7dcde9b82c4d41d399b93f2943bd550183439aa362973bb1e9df78cbad60'
SMOKE_COMMIT = '683fd5bd92b25de8d04780a2f5704a1a98eb25ce'
FINGERPRINT = '71073f3f8b26a77cdc5a14fc06078528942c87ffb8b135419f1eda9ac62021dd'
SOURCE_SHA = '07c0f5b0a0b50fe57b9c42fe5743efad1d4bda9ce806c64be88ea01193446f31'
SOURCE_MODEL_SHA = 'c06fbd97e32a483d46a2738014215de2721ad7db4b2b86547645ab72d4c555bc'
CERTIFICATE_FIELDS = frozenset(('schema', 'mode', 'identity', 'implementation_fingerprint',
    'native_verified', 'cohort_arrays_verified', 'source_forward_verified', 'head_archive_verified',
    'native_recorder_verified', 'train_seed', 'validation_seed', 'num_envs', 'duration_s',
    'solver_runs', 'formal_admission', 'source_checkpoint_sha256', 'source_model_state_sha256',
    'platform_task_id', 'platform_terminal_status', 'code_commit', 'cloud_log_sha256',
    'train_sha256', 'validation_sha256', 'head_sha256', 'audit_report_sha256',
    'report_artifact_sha256', 'platform_info_sha256', 'policy_rollout_sha256', 'logs_clean',
    'effectiveness_verified', 'dr_unlocked'))
HELPER_ERRORS = {
    'Invalid head history required commit': 'invalid_required_commit',
    'Head smoke ancestry is not established': 'ancestry_not_established',
    'Invalid head history checkout HEAD': 'invalid_checkout_head',
    'Required head smoke commit is not confirmed missing': 'commit_not_confirmed_missing',
    'Missing head smoke commit outside a verified shallow checkout': 'not_verified_shallow',
    'Invalid head history checkout root': 'invalid_checkout_root',
    'Head history checkout changed or is not clean': 'checkout_changed_or_dirty',
    'Head history origin is not the registered HTTPS repository': 'untrusted_origin',
    'Head history refresh failed; no retry or diagnostic echo': 'fetch_failed',
    'Head smoke ancestry remains unproved after one history refresh': 'ancestry_unproved',
    'Head history operation unavailable or timed out': 'git_unavailable_or_timeout',
    'Invalid head history Git readback': 'invalid_git_readback',
}
FIXED_ERRORS = frozenset(HELPER_ERRORS.values()) | frozenset((
    'invalid_marker', 'invalid_cli', 'invalid_expected_commit', 'checkout_guard_failed',
    'certificate_binding_failed', 'fingerprint_unavailable', 'fingerprint_mismatch',
    'fetch_budget_violation', 'invalid_helper_proof', 'unknown_helper_error', 'unknown_probe_error'))


class ProbeRejected(ValueError):
    """Only internally supplied fixed error codes, never external diagnostics."""


def _error_code(error, *, helper=False):
    value = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else None
    if helper:
        return HELPER_ERRORS.get(value, 'unknown_helper_error')
    return value if value in FIXED_ERRORS else 'unknown_probe_error'


def _marker(name, value):
    if any(type(v) not in (str, int, bool, type(None)) for v in value.values()):
        raise ProbeRejected('invalid_marker')
    print('[head-git-probe-'+name+'] '+json.dumps(value, allow_nan=False), flush=True)


def _git_read(*arguments):
    """Local checkout preconditions only; no caller-selected network route."""
    environment = os.environ.copy()
    environment['GIT_TERMINAL_PROMPT'] = '0'
    environment['GIT_ALLOW_PROTOCOL'] = 'https'
    try:
        row = subprocess.run(['git']+list(arguments), cwd=str(ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ProbeRejected('git_unavailable_or_timeout') from None
    if (type(row.returncode) is not int or type(row.stdout) is not bytes
            or type(row.stderr) is not bytes):
        raise ProbeRejected('invalid_git_readback')
    return row


def _checkout(expected):
    rows = [_git_read('rev-parse', '--show-toplevel'), _git_read('rev-parse', 'HEAD'),
        _git_read('status', '--porcelain', '--untracked-files=no'),
        _git_read('ls-files', '--others', '--exclude-standard', '--',
                  'humanoid', 'configs', 'resources')]
    try:
        root = Path(rows[0].stdout.decode('utf-8').strip()).resolve()
    except (UnicodeError, OSError, ValueError):
        raise ProbeRejected('checkout_guard_failed') from None
    if (any(row.returncode != 0 for row in rows) or root != ROOT
            or rows[1].stdout.strip() != expected.encode('ascii')
            or rows[2].stdout.strip() or rows[3].stdout.strip()):
        raise ProbeRejected('checkout_guard_failed')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ProbeRejected('certificate_binding_failed')
        result[key] = value
    return result


def _nonfinite(value):
    raise ProbeRejected('certificate_binding_failed')


def _certificate():
    path = ROOT/CERTIFICATE
    if path.resolve() != path or not path.is_file():
        raise ProbeRejected('certificate_binding_failed')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CERTIFICATE_SHA:
        raise ProbeRejected('certificate_binding_failed')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, ValueError):
        raise ProbeRejected('certificate_binding_failed') from None
    expected = dict(schema='head_smooth_native_smoke_audit_v1', mode='smoke',
        implementation_fingerprint=FINGERPRINT, native_verified=True, cohort_arrays_verified=True,
        source_forward_verified=True, head_archive_verified=True, native_recorder_verified=True,
        train_seed=305, validation_seed=505, num_envs=4, duration_s=2, solver_runs=1,
        formal_admission=False, source_checkpoint_sha256=SOURCE_SHA,
        source_model_state_sha256=SOURCE_MODEL_SHA, platform_task_id='TASK_20261004_087',
        platform_terminal_status='5', code_commit=SMOKE_COMMIT, logs_clean=False,
        effectiveness_verified=False, dr_unlocked=False)
    if type(value) is not dict or set(value) != CERTIFICATE_FIELDS:
        raise ProbeRejected('certificate_binding_failed')
    for key, item in expected.items():
        if type(value[key]) is not type(item) or value[key] != item:
            raise ProbeRejected('certificate_binding_failed')
    identity = value['identity']
    identity_keys = {'experiment', 'config_sha256', 'motion_sha256',
                     'urdf_lf_sha256', 'feature_fingerprint'}
    if (type(identity) is not dict or set(identity) != identity_keys
            or identity['experiment'] != 'f1_amp_walk02_sustain_control'
            or any(type(v) is not str for v in identity.values())):
        raise ProbeRejected('certificate_binding_failed')
    for key in identity_keys-{'experiment'}:
        if re.fullmatch('[0-9a-f]{64}', identity[key]) is None or identity[key] == '0'*64:
            raise ProbeRejected('certificate_binding_failed')
    for key in ('cloud_log_sha256', 'train_sha256', 'validation_sha256', 'head_sha256',
                'audit_report_sha256', 'report_artifact_sha256', 'platform_info_sha256',
                'policy_rollout_sha256'):
        if (type(value[key]) is not str or re.fullmatch('[0-9a-f]{64}', value[key]) is None
                or value[key] == '0'*64):
            raise ProbeRejected('certificate_binding_failed')
    if value['train_sha256'] == value['validation_sha256']:
        raise ProbeRejected('certificate_binding_failed')
    return value


def _runtime_fingerprint():
    """Execute only the existing pure hash function, not its Torch-owning module."""
    source = ROOT/'humanoid/amp/scaled_experiment.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'implementation_fingerprint']
    if len(functions) != 1 or functions[0].decorator_list or any(
            isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(functions[0])):
        raise ProbeRejected('fingerprint_unavailable')
    namespace = {'Path': Path, 'hashlib': hashlib,
                 '__builtins__': {'list': list, 'sorted': sorted}}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(source), 'exec'), namespace)
    value = namespace['implementation_fingerprint'](ROOT)
    if type(value) is not str or value != FINGERPRINT:
        raise ProbeRejected('fingerprint_mismatch')
    return value


def _load_driver():
    path = ROOT/'humanoid/scripts/run_amp_head_smooth.py'
    spec = importlib.util.spec_from_file_location('head_git_probe_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_stderr(stderr, returncode):
    """Pattern presence only: never a verified causal diagnosis or raw text."""
    if type(stderr) is not bytes or type(returncode) is not int:
        raise ProbeRejected('invalid_git_readback')
    unsupported = rb'(?:unknown|unrecognized|unsupported|invalid)\s+(?:option|switch)'
    flags = dict(
        unsupported_no_write_fetch_head=bool(re.search(unsupported+rb'[^\r\n]*no-write-fetch-head', stderr, re.I)),
        unsupported_deepen=bool(re.search(unsupported+rb'[^\r\n]*deepen', stderr, re.I)),
        auth_missing=bool(re.search(rb'could not read (?:username|password)|terminal prompts disabled|no credentials', stderr, re.I)),
        auth_failed=bool(re.search(rb'authentication failed|invalid username or password|invalid credentials|access denied|http[^\r\n]*(?:401|403)', stderr, re.I)),
        dns=bool(re.search(rb'could not resolve (?:host|proxy)|name or service not known|temporary failure in name resolution', stderr, re.I)),
        tls=bool(re.search(rb'ssl|tls|certificate verify failed|certificate problem', stderr, re.I)),
        timeout=bool(re.search(rb'timed out|timeout|time out', stderr, re.I)),
        missing_remote_object=bool(re.search(rb'not our ref|couldn.t find remote ref|unadvertised object|no such ref', stderr, re.I)))
    flags['unknown'] = returncode != 0 and not any(flags.values())
    return flags


def run_probe(expected_commit, *, emit=None):
    emit = _marker if emit is None else emit
    if type(expected_commit) is not str or re.fullmatch('[0-9a-f]{40}', expected_commit) is None:
        raise ProbeRejected('invalid_expected_commit')
    _checkout(expected_commit)
    _certificate()
    _runtime_fingerprint()
    driver = _load_driver()
    original = driver._head_history_git
    fetch_count = 0

    def wrapped(repo, *arguments):
        nonlocal fetch_count
        is_fetch = arguments and arguments[0] == 'fetch' and '-h' not in arguments
        if is_fetch:
            fetch_count += 1
            if fetch_count != 1:
                raise ProbeRejected('fetch_budget_violation')
        row = original(repo, *arguments)
        if is_fetch:
            emit('transport', dict(returncode=row.returncode, stdout_bytes=len(row.stdout),
                stderr_bytes=len(row.stderr), **classify_stderr(row.stderr, row.returncode)))
        return row

    emit('start', dict(schema='head_git_history_probe_v1', expected_commit=expected_commit,
        certificate_sha256=CERTIFICATE_SHA, runtime_fingerprint=FINGERPRINT,
        certificate_bound=True, training_started=False, solver_runs=0,
        effectiveness_verified=False, formal_admission=False, dr_unlocked=False))
    driver._head_history_git = wrapped
    proof = None
    error = None
    try:
        version = wrapped(ROOT, '--version')
        match = re.fullmatch(rb'git version ([0-9]+(?:\.[0-9]+){1,3}(?:\.windows\.[0-9]+)?)',
                             version.stdout.strip()) if version.returncode == 0 else None
        help_row = wrapped(ROOT, 'fetch', '-h')
        help_bytes = help_row.stdout+help_row.stderr
        emit('capabilities', dict(git_version_valid=match is not None,
            git_version=match.group(1).decode('ascii') if match else None,
            version_returncode=version.returncode, help_returncode=help_row.returncode,
            help_no_write_fetch_head=(b'--no-write-fetch-head' in help_bytes
                                     or b'--[no-]write-fetch-head' in help_bytes),
            help_deepen=b'--deepen' in help_bytes))
        proof = driver.ensure_smoke_ancestry(ROOT, SMOKE_COMMIT)
        if (type(proof) is not dict or set(proof) != {'ancestor_verified', 'history_refreshed',
                'required_commit', 'head_commit'} or proof['ancestor_verified'] is not True
                or type(proof['history_refreshed']) is not bool
                or proof['required_commit'] != SMOKE_COMMIT
                or proof['head_commit'] != expected_commit):
            proof = None
            error = 'invalid_helper_proof'
    except ProbeRejected as caught:
        error = _error_code(caught)
    except ValueError as caught:
        error = _error_code(caught, helper=True)
    except Exception:
        error = 'unknown_helper_error'
    finally:
        driver._head_history_git = original
    _checkout(expected_commit)
    return dict(schema='head_git_history_probe_v1', diagnostic_completed=True,
        helper_succeeded=proof is not None and error is None, helper_error=error,
        ancestor_verified=proof is not None and error is None,
        history_refreshed=proof['history_refreshed'] if proof is not None and error is None else False,
        fetch_count=fetch_count, checkout_unchanged=True, certificate_bound=True,
        runtime_fingerprint_bound=True, training_started=False, solver_runs=0,
        ppo_updates_added=0, offline_admitted=False, formal_admission=False,
        effectiveness_verified=False, dr_unlocked=False)


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ProbeRejected('invalid_cli')


def main(argv=None):
    try:
        parser = SafeParser(allow_abbrev=False)
        parser.add_argument('--expected-commit', required=True)
        args = parser.parse_args(argv)
        result = run_probe(args.expected_commit)
        status = 0
    except ProbeRejected as error:
        # All ProbeRejected instances are created internally from fixed labels.
        result = dict(schema='head_git_history_probe_v1', diagnostic_completed=False,
            helper_error=_error_code(error), training_started=False, solver_runs=0,
            offline_admitted=False, formal_admission=False,
            effectiveness_verified=False, dr_unlocked=False)
        status = 2
    except Exception:
        result = dict(schema='head_git_history_probe_v1', diagnostic_completed=False,
            helper_error='unknown_probe_error', training_started=False, solver_runs=0,
            offline_admitted=False, formal_admission=False,
            effectiveness_verified=False, dr_unlocked=False)
        status = 2
    _marker('complete', result)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
