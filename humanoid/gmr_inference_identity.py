"""Verified finite-clip checkpoints; refuse cross-experiment playback."""

PROFILES = {
    'x1_gmr_29dof': {
        'source_task': 'TASK_20260911_112',
        'training_commit': '9688e5ae4d0548ec5b4d8ba49cb57b9983168671',
        'checkpoint': 5000,
        'num_actions': 29,
        'checkpoint_sha256': 'e528971d23a2d8d693389eaf857f927a120814e94a955234de68eb84ad92d7cf',
        'motion_sha256': '08730ae2264bc4598332a1f0e5adb572442662192e3bad564870350f0969c3b6',
        'urdf_lf_sha256': '79a51ab2e8e111c0cd0e1167b203ef7df02cdcc504a52ae7a117e07f06d03be0',
    },
    'x1_gmr_clip': {
        'source_task': 'TASK_20260910_187',
        'training_commit': '181f8034e62105f4138b90ebb560f63c0d1640a4',
        'checkpoint_sha256': 'cf5b3c4267dd874eada01ba5b1dd7076cc6f0b852d2222101c21c6c548abc476',
        'motion_sha256': 'a3ae41e54909455d477cefa662bdcbbee8a1852564ef547ebeb455f275a0be02',
    },
    'x1_gmr_upright': {
        'source_task': 'TASK_20260911_008',
        'training_commit': '9829d1d1ee88d94dd8db414406a1b700ea12c6fe',
        'checkpoint_sha256': '5e89389e9eb3392759a78ff6295c2a395d3af0c8855c49499bf57fa8e7a8c5f9',
        'motion_sha256': 'd625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb',
    },
}


def validate_identity(task, source_task, checkpoint_sha256, motion_sha256, checkpoint=None):
    if task not in PROFILES:
        raise ValueError('Unsupported finite GMR inference task: ' + task)
    expected = PROFILES[task]
    if checkpoint is not None and checkpoint != expected.get('checkpoint', 5000):
        raise ValueError('GMR inference identity mismatch: checkpoint number')
    actual = dict(source_task=source_task, checkpoint_sha256=checkpoint_sha256.lower(),
                  motion_sha256=motion_sha256.lower())
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError('GMR inference identity mismatch: ' + key)
    return dict(expected, task=task)
