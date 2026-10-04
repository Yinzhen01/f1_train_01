"""Read-only Gradmotion hardware diagnostic, never a fitting/simulation stage.

This tool neither relaxes run_amp_head_smooth's hardware guard nor creates a
smoke certificate. Platform SKU metadata must be checked separately against the
actual task API; a printed expected SKU is not proof of allocated hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def read_hardware(torch, platform_name):
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    devices = []
    if available:
        for index in range(count):
            name = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            devices.append(dict(index=index, name=name,
                total_memory_bytes=int(props.total_memory),
                compute_capability=[int(props.major), int(props.minor)]))
    name_matches = bool(len(devices) == 1 and
                        '4090D' in devices[0]['name'].replace(' ', ''))
    checks = dict(linux=platform_name == 'linux', cuda_available=available,
                  single_cuda_device=count == 1, name_contains_4090d=name_matches)
    return dict(platform=platform_name, torch_version=str(torch.__version__),
        torch_cuda_version=torch.version.cuda, cuda_device_count=count,
        devices=devices, driver_guard_checks=checks,
        current_driver_guard_would_pass=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.expected_commit):
        raise ValueError('Expected an exact published commit')
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=str(repo)).strip()
    if actual != args.expected_commit or dirty:
        raise ValueError('Exact published clean checkout is required')
    subprocess.check_call(['git', 'ls-files', '--error-unmatch',
                           'tools/amp/probe_native_hardware.py'],
                          cwd=str(repo), stdout=subprocess.DEVNULL)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
        '--', 'tools/amp', 'humanoid', 'configs', 'resources'], cwd=str(repo)).strip()
    if untracked:
        raise ValueError('Untracked native/probe code is not allowed')
    # Preserve the Isaac Gym ABI ordering, even though no simulator is created.
    from isaacgym import gymapi  # noqa: F401
    import torch
    hardware = read_hardware(torch, sys.platform)
    report = dict(schema='amp_native_hardware_readback_v1', code_commit=actual,
        probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        hardware=hardware, simulation_created=False, policy_executed=False,
        fitting_executed=False, smoke_admission=False, effectiveness_verified=False,
        dr_unlocked=False)
    print('[amp-native-hardware-readback] '+json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
