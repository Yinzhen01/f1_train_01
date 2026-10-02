"""Restore shallow-clone ancestry before the unchanged frozen-policy diagnostic."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
REMOTE = 'https://github.com/Yinzhen01/f1_train_01.git'
CERTIFICATE = Path('docs/validation/physics_diagnostic_cloud_smoke.json')
GROUPS = ('original', 'velocity1', 'selfoff')


def parse_request(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', choices=('full',), required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--task', choices=('f1_amp_physics_diagnostic',),
                        default='f1_amp_physics_diagnostic')
    parser.add_argument('--groups', default=','.join(GROUPS))
    parser.add_argument('--smoke-certificate', type=Path, required=True)
    args = parser.parse_args(argv)
    groups = args.groups.split(',')
    if (re.fullmatch('[0-9a-f]{40}', args.expected_commit) is None or
            not groups or len(set(groups)) != len(groups) or
            any(group not in GROUPS for group in groups)):
        raise ValueError('Invalid full diagnostic commit or groups')
    return args


def git(repo, *arguments, require_success=True):
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    try:
        result = subprocess.run(['git']+list(arguments), cwd=str(repo), env=env,
            capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError('Diagnostic history Git operation unavailable or timed out') from None
    if require_success and result.returncode != 0:
        # Git transport output can contain authentication data; never echo it.
        raise ValueError('Diagnostic history Git operation failed')
    return result


def checkout_identity(repo, expected_commit):
    root = Path(git(repo, 'rev-parse', '--show-toplevel').stdout.strip()).resolve()
    head = git(repo, 'rev-parse', 'HEAD').stdout.strip()
    if root != repo or head != expected_commit:
        raise ValueError('Diagnostic history checkout root or HEAD changed')


def prepare_history(repo, args):
    repo = Path(repo).resolve()
    checkout_identity(repo, args.expected_commit)
    certificate = args.smoke_certificate
    if not certificate.is_absolute():
        certificate = repo/certificate
    if certificate.resolve() != (repo/CERTIFICATE).resolve():
        raise ValueError('Only the committed physics diagnostic smoke certificate is allowed')
    try:
        value = json.loads(certificate.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise ValueError('Diagnostic smoke certificate is missing or invalid') from None
    smoke_commit = value.get('code_commit') if isinstance(value, dict) else None
    if not isinstance(smoke_commit, str) or re.fullmatch('[0-9a-f]{40}', smoke_commit) is None:
        raise ValueError('Diagnostic smoke commit must be a complete SHA')
    ancestry = ('merge-base', '--is-ancestor', smoke_commit, args.expected_commit)
    fetched = False
    if git(repo, *ancestry, require_success=False).returncode != 0:
        shallow = git(repo, 'rev-parse', '--is-shallow-repository').stdout.strip()
        if shallow != 'true':
            raise ValueError('Smoke commit is not an ancestor of the full diagnostic checkout')
        git(repo, 'fetch', '--no-tags', '--depth=64', REMOTE, args.expected_commit)
        fetched = True
    checkout_identity(repo, args.expected_commit)
    if git(repo, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        raise ValueError('Full diagnostic checkout has tracked changes')
    if git(repo, 'ls-files', '--others', '--exclude-standard', '--',
           'humanoid', 'configs', 'resources').stdout.strip():
        raise ValueError('Full diagnostic checkout has untracked source or configuration')
    if git(repo, *ancestry, require_success=False).returncode != 0:
        raise ValueError('Smoke ancestry is still unproved after diagnostic history preparation')
    return dict(expected_commit=args.expected_commit, smoke_commit=smoke_commit,
                fetched_history=fetched, ancestor_verified=True)


def main(argv=None):
    original_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_request(original_argv)
    proof = prepare_history(ROOT, args)
    print('[amp-physics-history-bootstrap] '+json.dumps(proof), flush=True)
    subprocess.run([sys.executable, str(ROOT/'humanoid/scripts/diagnose_amp_physics.py')]
                   +original_argv, cwd=str(ROOT), check=True)


if __name__ == '__main__':
    main()
