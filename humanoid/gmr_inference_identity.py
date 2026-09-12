"""Verified finite-clip checkpoints; refuse cross-experiment playback."""

PROFILES = {
    'x1_gmr_swing': {
        'source_task': 'TASK_20260911_171',
        'training_commit': 'a92dfe6346bcd1e853a1c923fec33667e9b601bb',
        'checkpoint': 8000,
        'num_actions': 12,
        'checkpoint_sha256': '4c865c178a12a4796013b6b138972becb875301d7198543a6909c9a76ef7fe50',
        'motion_sha256': 'd625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb',
        'urdf_lf_sha256': '9fb3d6efa623576a964aef9a3e2db2f96ca2296a85f2e0ade6bfe0ecf043ba15',
    },
    'x1_gmr_smooth': {
        'source_task': 'TASK_20260911_116',
        'training_commit': '21788f6da04cc578925343a09a77aab5e8a10511',
        'checkpoint': 7000,
        'num_actions': 12,
        'checkpoint_sha256': 'ccaf51ecb069e6ecb80ee9fe07eda4a63dab627e4592d15b4240b5d024701857',
        'motion_sha256': 'd625efc73972b4f4952587ac6391543b6d747b404184703361a394e0b6bca4fb',
        'urdf_lf_sha256': '9fb3d6efa623576a964aef9a3e2db2f96ca2296a85f2e0ade6bfe0ecf043ba15',
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

# Completed, downloaded and strictly validated model9000 acceleration sweep.
PROFILES.update({
    'x1_gmr_accel_' + group: dict(
        PROFILES['x1_gmr_swing'], source_task=source_task, checkpoint=9000,
        training_commit='eccd14eed21ffe980c90abd7dc740d043e01fcaf',
        checkpoint_sha256=digest)
    for group, source_task, digest in (
        ('1x', 'TASK_20260912_033', 'd258477d65787c7b89740eb4df4261b04e2fa1b204fdb9e64400a71ab595bdae'),
        ('3x', 'TASK_20260912_034', '576d5cc511def5318a6c41d4b8e1aadd6dda56b2986f2f2f58ad806a32a9cbb7'),
        ('10x', 'TASK_20260912_035', '22fb7520a219a265c39b627e8c453136adc57bed019ab4dccff4614f0a58bb37'),
    )
})


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
