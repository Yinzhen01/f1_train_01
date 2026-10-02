"""Bind a raw-data smoke audit to completed Gradmotion execution evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.physics_diagnostic import GROUPS, SOURCE_CONFIG, SOURCE_SHA, diagnostic_contract
from humanoid.amp.scaled_experiment import implementation_fingerprint
from tools.amp.inspect_rollout import load_bundle
from tools.amp.verify_signal_formal import inspect_tracebacks

ARTIFACT_IDS = {'original': 8802500, 'velocity1': 8802501, 'selfoff': 8802502}
START = '[amp-physics-diagnostic-start] '
GROUP = '[amp-physics-diagnostic-group-complete] '
COMPLETE = '[amp-physics-diagnostic-complete] '
HISTORY = '[amp-physics-reset-history] '
UPDATE_FIELDS = ('policy_updates', 'discriminator_updates', 'estimator_updates')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def marker_records(text, marker):
    """SDK output can share a line with a marker; parse JSON, not line tails."""
    records = []
    for match in re.finditer(re.escape(marker), text):
        try:
            value, size = json.JSONDecoder().raw_decode(text[match.end():])
        except json.JSONDecodeError as error:
            raise ValueError('Invalid diagnostic marker JSON: '+marker) from error
        require(isinstance(value, dict), 'Diagnostic marker must contain an object')
        records.append((match.start(), match.end()+size, value))
    return records


def validate_log_records(text, manifest, completion):
    starts, groups, ends = (marker_records(text, marker) for marker in (START, GROUP, COMPLETE))
    require(len(starts) == len(ends) == 1 and len(groups) == len(GROUPS),
            'Missing or duplicate frozen-diagnostic execution markers')
    initial = copy.deepcopy(manifest)
    initial['group_artifacts'] = {}
    require(starts[0][2] == initial, 'Cloud start marker differs from downloaded batch')
    require(ends[0][2] == completion, 'Cloud completion differs from downloaded batch')
    require(starts[0][1] <= groups[0][0] and groups[-1][1] <= ends[0][0],
            'Diagnostic completion markers are out of order')
    require([row[2].get('group') for row in groups] == manifest['groups'],
            'Cloud group completion order or membership changed')
    for row in groups:
        group = row[2]['group']
        artifact = manifest['group_artifacts'][group]
        require(row[2] == dict(group=group, artifact=artifact['file'], sha256=artifact['sha256']),
                'Cloud group completion artifact differs from downloaded batch')
    return ends[0][1]


def validate_history_records(text):
    """Only proved zero-history JSON payloads are exempted from error words."""
    records = marker_records(text, HISTORY)
    groups = marker_records(text, GROUP)
    starts = marker_records(text, START)
    require(len(records) == 6 and len(groups) == 3 and len(starts) == 1,
            'Missing or duplicate native reset-history probes')
    proof = []
    for index, (start, end, value) in enumerate(records):
        group_index, mode_index = divmod(index, 2)
        expected_mode = ('standing', 'reference')[mode_index]
        lower = starts[0][1] if group_index == 0 else groups[group_index-1][1]
        require(lower <= start < end <= groups[group_index][0] and
                set(value) == {'mode', 'histories'} and value['mode'] == expected_mode and
                isinstance(value['histories'], dict) and
                set(value['histories']) == {'obs_history', 'critic_history'},
                'Native reset-history group/mode order or schema changed')
        for name, counts in value['histories'].items():
            require(isinstance(counts, dict) and
                    set(counts) == {'elements', 'nonzero', 'nonfinite', 'negative_zero'} and
                    all(isinstance(count, int) and not isinstance(count, bool) and count >= 0
                        for count in counts.values()), 'Invalid native reset-history count schema')
            require(counts['nonzero'] == counts['nonfinite'] == 0 and
                    counts['negative_zero'] <= counts['elements'],
                    'Native reset history is nonzero, nonfinite or inconsistent')
        proof.append(dict(group=groups[group_index][2]['group'], **value))
    # Preserve character offsets and all surrounding errors. Raw evidence and
    # its hash are unchanged; only these six validated JSON objects are masked.
    scanned = text
    for start, end, _ in reversed(records):
        payload = start+len(HISTORY)
        scanned = scanned[:payload]+' '*(end-payload)+scanned[end:]
    return scanned, dict(probes_verified=6, modes_per_group=['standing', 'reference'],
        groups=list(GROUPS), records=proof,
        total_history_elements=sum(counts['elements'] for row in proof
            for counts in row['histories'].values()),
        total_negative_zero_elements=sum(counts['negative_zero'] for row in proof
            for counts in row['histories'].values()))


def validate_wrapper_warnings(text, completion_end, task_id):
    """Mask only the proved pre-native metadata errors and post-native banner."""
    starts, groups = (marker_records(text, marker) for marker in (START, GROUP))
    start = starts[0][0]
    warnings, spans = [], []
    metadata_error = ('ERROR: Can not execute `setup.py` since setuptools is not available '
                      'in the build environment.')
    if metadata_error in text:
        lines = [
            'Processing ./gradmotion.tar.gz',
            'Preparing metadata (setup.py): started',
            "Preparing metadata (setup.py): finished with status 'error'",
            'error: subprocess-exited-with-error', '',
            '× python setup.py egg_info did not run successfully.', '│ exit code: 1',
            '╰─> [3 lines of output]', None, 'warnings.warn(', metadata_error,
            '[end of output]', '',
            'note: This error originates from a subprocess, and is likely not a problem with pip.',
            'error: metadata-generation-failed', '',
            '× Encountered error while generating package metadata.', '╰─> See above for output.', '',
            'note: This is an issue with the package mentioned above, not pip.',
            'hint: See above for details.']
        pattern = r'(?m)^'+r'\r?\n'.join(r'[ \t]*'+(re.escape(line) if line is not None else
            r'/opt/conda/lib/python3\.13/site-packages/_distutils_hack/__init__\.py:53: '
            r'UserWarning: [^\r\n]+') for line in lines)+r'[ \t]*\r?$'
        blocks = list(re.finditer(pattern, text))
        require(len(blocks) == 1 and text.count(metadata_error) == 1 and blocks[0].end() < start,
                'Unproved, duplicate or native metadata installation failure')
        block = blocks[0]
        recovery = text[block.end():start]
        recovered = (r'(?m)^Successfully installed gradmotion-1\.0\.10\r?$',
            r'(?m)^Successfully installed humanoid\r?$',
            r"(?m)^Importing module 'gym_38' \([^\r\n]+/gym_38\.so\)\r?$",
            r'(?m)^PyTorch version 2\.4\.1\r?$')
        require(all(re.search(proof, recovery) for proof in recovered) and
                '/opt/conda/envs/pointfoot_legged_gym/lib/python3.8/site-packages/torch/' in recovery,
                'Metadata failure has no subsequent Python 3.8 native runtime proof')
        error_lines = (lines[3], metadata_error, lines[14])
        for line in error_lines:
            matches = list(re.finditer(r'(?m)^[ \t]*'+re.escape(line)+r'[ \t]*\r?$', text))
            require(len(matches) == 1 and block.start() <= matches[0].start() < matches[0].end() <= block.end(),
                    'Duplicate or unscoped metadata installation error')
            spans.append((matches[0].start(), matches[0].end()))
        warnings.append(dict(kind='pre_native_python313_metadata_failure',
            scope='before_diagnostic_start', character_scope=[block.start(), block.end()],
            masked_character_scopes=[list(span) for span in spans],
            setup_metadata_exit_code=1, error=metadata_error,
            package='gradmotion.tar.gz', python_path='/opt/conda/lib/python3.13',
            subsequent_installation_verified=['gradmotion-1.0.10', 'humanoid'],
            native_runtime_verified=dict(python='3.8', gym_binding='gym_38', torch='2.4.1')))
    failures = list(re.finditer(r'(?m)^RL task is failed\.\r?$', text))
    if failures or 'RL task is failed.' in text or 'Task status is completed, cannot terminate' in text or \
            re.search(r'"taskStatus"\s*:\s*"6"', text):
        require(len(failures) == 1, 'Missing or duplicate exact wrapper failure banner')
        failure = failures[0]
        tail = re.fullmatch(r'RL task is failed\.\r?\n(?P<status>\{[^\r\n]+\})\r?\n'
            r'(?P<response>\{[^\r\n]+\})Start to backup logs\.\r?\n'
            r"cp: missing destination file operand after '/mnt/rl_tfevents_logs'\r?\n"
            r"Try 'cp --help' for more information\.\r?\nEnd to backup logs\.\r?\n"
            r'RL task is successful\.[ \t\r\n]*', text[failure.start():])
        require(tail is not None and completion_end < failure.start(),
                'Unproved or native wrapper failure sequence')
        try:
            attempted, response = (json.loads(tail.group(name)) for name in ('status', 'response'))
        except json.JSONDecodeError as error:
            raise ValueError('Invalid wrapper status-conflict JSON') from error
        require(attempted == dict(taskStatus='6', taskId=task_id) and
                set(response) == {'code', 'msg', 'msgEn', 'success', 'time'} and
                response['code'] == 200 and response['success'] is True and
                response['msg'] == '任务状态已完成，不能终止' and
                response['msgEn'] == 'Task status is completed, cannot terminate',
                'Wrapper status conflict identity or completed-task rejection changed')
        exits = list(re.finditer(r'(?m)^【SDK】训练进程状态码：([0-9]+)\r?$', text))
        completed = list(re.finditer(r'(?m)^\[SDK\]\[INFO\] [0-9 :\-]+ '
            r'Task\(([^)]+)\) status updated to: Completed , ret:(True|False)\r?$', text))
        require(len(exits) == len(completed) == 1 and exits[0].group(1) == '0' and
                completed[0].group(1) == task_id and completed[0].group(2) == 'True' and
                completion_end < exits[0].start() < exits[0].end() < completed[0].start() <
                completed[0].end() < failure.start(), 'Wrapper failure lacks successful native SDK exit/completion')
        uploads = []
        for filename, lower in [(row[2]['artifact'], row[1]) for row in groups]+[
                ('model_amp_manifest.pt', completion_end)]:
            matches = list(re.finditer(r'(?m)^\[SDK\]\[INFO\] [0-9 :\-]+ PT file \('
                +re.escape(filename)+r'\) uploaded successfully\r?$', text))
            require(matches and all(lower <= match.start() < match.end() < exits[0].start()
                    for match in matches), 'Wrapper failure lacks all four ordered successful artifact uploads')
            uploads.append(dict(file=filename, character_scopes=[[match.start(), match.end()] for match in matches]))
        spans.append((failure.start(), failure.end()))
        warnings.append(dict(kind='post_native_wrapper_status_conflict', scope='after_diagnostic_completion',
            character_scope=[failure.start(), len(text)], masked_character_scopes=[[failure.start(), failure.end()]],
            sdk_child_exit_code=0, sdk_completed_task_id=task_id, sdk_completed_ret=True,
            sdk_exit_character_scope=[exits[0].start(), exits[0].end()],
            sdk_completed_character_scope=[completed[0].start(), completed[0].end()],
            uploaded_artifacts=uploads, attempted_status_update=attempted,
            completed_task_rejection=response, final_wrapper_success_verified=True,
            backup_error="cp: missing destination file operand after '/mnt/rl_tfevents_logs'",
            note='Status 6 was attempted after native exit 0 and SDK Completed, then rejected; wrapper cause unverified.'))
    scanned = text
    for lower, upper in sorted(spans, reverse=True):
        scanned = scanned[:lower]+' '*(upper-lower)+scanned[upper:]
    return scanned, warnings


def decode_post_diagnostic_log(raw, diagnostics):
    """Opt-in escaping preserves every byte after a proved diagnostic upload tail."""
    positions = [i for i, value in enumerate(raw) if value < 32 and value not in (9, 10, 13, 27)]
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError as error:
        positions.append(error.start)
    if not positions:
        return raw.decode('utf-8')
    first = min(positions)
    prefix = raw[:first].decode('utf-8')
    starts, groups, ends = (marker_records(prefix, marker) for marker in (START, GROUP, COMPLETE))
    require(len(starts) == len(ends) == 1 and len(groups) == 3,
            'Binary log precedes complete frozen-diagnostic records')
    start, finish = starts[0][2], ends[0][2]
    require(start.get('mode') == finish.get('mode') == 'smoke' and
            start.get('no_training') is True and finish.get('no_training') is True and
            finish.get('complete') is True and finish.get('all_finite') is True and
            all(start.get(key) == finish.get(key) == 0 for key in UPDATE_FIELDS),
            'Binary log has no completed finite diagnostic-only smoke')
    require({row[2].get('group') for row in groups} == set(GROUPS) and
            'uploaded successfully' in prefix[ends[0][1]:],
            'Binary data precedes completed three-group SDK upload tail')
    text = raw.decode('utf-8', errors='backslashreplace')
    text = ''.join('\\x%02x' % ord(value) if
        (ord(value) < 32 and value not in '\t\r\n\x1b') or ord(value) == 127 else value
        for value in text)
    diagnostics.update(escaped_post_completion_binary=True, first_binary_byte_offset=first,
        raw_log_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw),
        nul_count=raw.count(b'\0'),
        note='Raw log retained; post-completion bytes escaped, never removed.')
    return text


def read_log(path, allow_post_completion_binary=False, diagnostics=None):
    path = Path(path)
    diagnostics = {} if diagnostics is None else diagnostics
    raw = path.read_bytes()
    require(path.suffix in ('.log', '.json'), 'Expected .log or gm .json log envelope')
    try:
        text = raw.decode('utf-8')
        require(not any(ord(c) < 32 and c not in '\t\r\n\x1b' for c in text),
                'Unclean binary/control bytes in log')
    except (UnicodeDecodeError, ValueError):
        if path.suffix != '.log' or not allow_post_completion_binary:
            raise
        text = decode_post_diagnostic_log(raw, diagnostics)
    if path.suffix == '.json':
        envelope = json.loads(text)
        require(envelope.get('success') is True and isinstance(envelope.get('data'), str),
                'Invalid gm log envelope')
        text = envelope['data'].replace('\\n', '\n')
        require(not any(ord(c) < 32 and c not in '\t\r\n\x1b' for c in text),
                'Unclean control bytes in gm log data')
    diagnostics.update(raw_log_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw))
    return text


def inspect_log_errors(text, completion_end):
    """Scan every unmasked error, allowing only the existing exact SDK reset."""
    try:
        traces = inspect_tracebacks(text)
    except AssertionError as error:
        raise ValueError('Non-allowlisted or truncated cloud traceback') from error
    require(not re.search(r'\[f1-amp-update\]|Learning iteration|\[f1-amp-complete\]', text),
            'Frozen diagnosis unexpectedly contains training updates')
    require(not re.search(r'CUDA out of memory|\bOOM\b|\bNonfinite\b|\bNaN\b|\bInf\b|'
                         r'RuntimeError:|ValueError:|AssertionError:|TypeError:|KeyError:|'
                         r'IndexError:|ImportError:|ModuleNotFoundError:|FileNotFoundError:', text,
                         flags=re.IGNORECASE), 'Cloud execution contains an error or nonfinite value')
    # An error banner is not a diagnostic success, even when later upload messages exist.
    require('RL task is failed.' not in text, 'Cloud wrapper reported failure')
    for line in text.splitlines():
        classes = re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)):', line)
        for name in classes:
            sdk_reset = (traces > 0 and
                ((name == 'ConnectionResetError' and
                  line == 'ConnectionResetError: [Errno 104] Connection reset by peer') or
                 (name == 'StreamLostError' and line.startswith(('ERROR:pika.', 'INFO:pika.',
                     'ERROR:coldplay.get_rabbitmq:')) and
                  "ConnectionResetError(104, 'Connection reset by peer')" in line)))
            require(sdk_reset, 'Non-allowlisted cloud exception line')
        if re.search(r'\[ERROR\]|^\s*ERROR:', line, flags=re.IGNORECASE):
            sdk_reset = (traces > 0 and line.startswith((
                'ERROR:pika.adapters.base_connection:', 'ERROR:pika.adapters.blocking_connection:',
                'ERROR:coldplay.get_rabbitmq:')) and 'StreamLostError:' in line and
                "ConnectionResetError(104, 'Connection reset by peer')" in line)
            require(sdk_reset, 'Non-allowlisted cloud error log line')
    return dict(sdk_connection_reset_tracebacks=traces, logs_clean=traces == 0,
                completion_character_offset=completion_end)


def bind_evidence(certificate, manifest, completion, status, text, record_manifests,
                  artifact_hashes, expected_commit, task_id, fingerprint):
    require(re.fullmatch(r'[0-9a-f]{40}', expected_commit) is not None and
            re.fullmatch(r'TASK_[0-9]{8}_[0-9]+', task_id) is not None,
            'Invalid expected commit or cloud task identity')
    require(status.get('success') is True, 'Cloud task status request did not succeed')
    task = status.get('data', {}).get('taskBaseInfo', {})
    require(task.get('taskId') == task_id and str(task.get('taskStatus')) == '5' and
            str(task.get('userId')) == '4409' and task.get('goodsId') == 'ESKU000001' and
            task.get('imageVersion') == 'V000124', 'Cloud task identity/status/resource mismatch')
    require(manifest.get('mode') == completion.get('mode') == 'smoke' and
            manifest.get('code_commit') == certificate.get('code_commit') == expected_commit and
            manifest.get('source_checkpoint_sha256') == completion.get('source_checkpoint_sha256') ==
            certificate.get('source_checkpoint_sha256') == SOURCE_SHA and
            manifest.get('source_completed_updates') == completion.get('source_completed_updates') == 2500 and
            manifest.get('source_task') == 'TASK_20260926_110' and
            manifest.get('source_config') == SOURCE_CONFIG and
            manifest.get('sdk_task') == 'f1_amp_physics_diagnostic' and
            manifest.get('num_envs') == certificate.get('num_envs') == 2 and
            manifest.get('duration_s') == certificate.get('duration_s') == 1. and
            manifest.get('seed') == 5, 'Downloaded smoke identity/budget mismatch')
    require(manifest.get('groups') == completion.get('groups') == certificate.get('groups') == list(GROUPS),
            'Smoke must cover exactly the three registered groups in order')
    require(all(manifest.get(key) == completion.get(key) == 0 for key in UPDATE_FIELDS),
            'Frozen diagnosis updated a policy, estimator or discriminator')
    require(all(value.get('no_training') is True and value.get('effectiveness_verified') is False and
                value.get('dr_unlocked') is False for value in (manifest, completion, certificate)),
            'Diagnosis training/effectiveness/DR boundary changed')
    require(completion.get('complete') is True and completion.get('all_finite') is True and
            completion.get('group_artifacts_verified') is True, 'Batch completion evidence missing')
    require(all(certificate.get(key) is True for key in ('verified', 'all_finite', 'raw_verified',
            'readback_verified', 'reset_inputs_exact')) and
            certificate.get('cloud_execution_verified') is False and
            certificate.get('implementation_fingerprint') == fingerprint,
            'Supplied raw-data smoke audit is incomplete, already bound or stale')
    require(set(manifest.get('group_artifacts', {})) == set(certificate.get('group_artifacts', {})) ==
            set(record_manifests) == set(artifact_hashes) == set(GROUPS), 'Missing group artifact evidence')
    for group in GROUPS:
        artifact = manifest['group_artifacts'][group]
        require(artifact.get('file') == 'model_%d.pt' % ARTIFACT_IDS[group] and
                artifact.get('sha256') == certificate['group_artifacts'][group].get('sha256') ==
                artifact_hashes[group] and re.fullmatch(r'[0-9a-f]{64}', artifact_hashes[group]) is not None,
                'Downloaded group checksum differs from cloud batch/raw-data audit')
        record = record_manifests[group]
        require(artifact.get('manifest') == record and record.get('code_commit') == expected_commit and
                record.get('checkpoint_sha256') == SOURCE_SHA and
                record.get('physics_diagnostic') == diagnostic_contract(group) and
                record.get('implementation_fingerprint') == fingerprint and
                record.get('identity') == manifest.get('identity') and
                record.get('num_envs') == 2 and record.get('duration_s') == 1. and
                record.get('diagnostic_only') is True and record.get('policy_deterministic') is True and
                record.get('dr_unlocked') is False and record.get('effectiveness_verified') is False,
                'Downloaded record differs from frozen smoke batch')
    end = validate_log_records(text, manifest, completion)
    scanned, history_proof = validate_history_records(text)
    scanned, wrapper_warnings = validate_wrapper_warnings(scanned, end, task_id)
    diagnostics = inspect_log_errors(scanned, end)
    diagnostics['logs_clean'] = diagnostics['logs_clean'] and not wrapper_warnings
    result = copy.deepcopy(certificate)
    result.update(cloud_execution_verified=True, cloud_task_id=task_id,
                  cloud_evidence=dict(user_id='4409', platform_terminal_status='5',
                      goods_id='ESKU000001', image_version='V000124',
                      native_reset_history=history_proof, non_native_warnings=wrapper_warnings, **diagnostics))
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('certificate', 'batch', 'folder', 'status', 'logs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('expected-commit', 'task-id'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--allow-post-completion-binary', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Cloud-bound certificate output already exists')
    import torch
    packed = torch.load(args.batch, weights_only=True, map_location='cpu')
    manifest, completion = (json.loads(packed[key]) for key in ('manifest_json', 'certificate_json'))
    records, hashes = {}, {}
    for group in GROUPS:
        path = args.folder/('model_%d.pt' % ARTIFACT_IDS[group])
        hashes[group] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[group], _ = load_bundle(path)
    diagnostics = {}
    text = read_log(args.logs, args.allow_post_completion_binary, diagnostics)
    result = bind_evidence(json.loads(args.certificate.read_text(encoding='utf-8')), manifest, completion,
        json.loads(args.status.read_text(encoding='utf-8')), text, records, hashes,
        args.expected_commit, args.task_id, implementation_fingerprint(ROOT))
    result['cloud_evidence'].update(diagnostics, batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest(),
        input_certificate_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        status_sha256=hashlib.sha256(args.status.read_bytes()).hexdigest())
    if diagnostics.get('escaped_post_completion_binary'):
        result['cloud_evidence']['logs_clean'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), cloud_task_id=args.task_id,
        cloud_execution_verified=True, effectiveness_verified=False, dr_unlocked=False)), flush=True)


if __name__ == '__main__':
    main()
