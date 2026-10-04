"""CPU readbacks and synthetic protocol fixtures, never native GPU evidence."""
import copy
import unittest

import torch

from humanoid.scripts.collect_amp_head_cohort import (
    source_forward_context, validate_source_forward_context)


def cpu_readback():
    policy = torch.nn.Linear(3, 2).eval()
    observation = torch.zeros(4, 3)
    with torch.no_grad():
        value = source_forward_context(policy, observation)
    return policy, observation, value


def native_fixture():
    """Deliberately synthetic CUDA-shaped metadata; no native/device claim."""
    _, _, value = cpu_readback()
    value.update(device_type='cuda', device_index=0, torch_version='fixture-only',
        torch_cuda_version='12.1', cudnn_version=90100, cudnn_enabled=True,
        cudnn_deterministic=True, cudnn_benchmark=False)
    return value


class SourceForwardContextTests(unittest.TestCase):
    def test_actual_cpu_context_is_read_only_and_cannot_authenticate_native_gpu(self):
        policy, observation, first = cpu_readback()
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            second = source_forward_context(policy, observation)
        self.assertEqual(first, second)
        self.assertEqual(first['device_type'], 'cpu')
        self.assertIsNone(first['device_index'])
        self.assertFalse(first['grad_enabled'])
        self.assertFalse(first['policy_training'])
        self.assertTrue(validate_source_forward_context(first))
        with self.assertRaises(ValueError):
            validate_source_forward_context(first, native=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, policy.state_dict()[key]))

    def test_only_existing_source_native_numerical_protocol_is_accepted(self):
        self.assertTrue(validate_source_forward_context(native_fixture(), native=True))
        for changes in (dict(device_type='cpu', device_index=None), dict(device_index=1),
                dict(cudnn_enabled=False), dict(cudnn_deterministic=False),
                dict(cudnn_benchmark=True), dict(torch_cuda_version=None), dict(cudnn_version=None)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_source_forward_context(dict(native_fixture(), **changes), native=True)

    def test_strict_keysets_and_true_types_not_coercion_or_equal_bool_integers(self):
        original = native_fixture()
        for key in original:
            missing = dict(original)
            del missing[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_source_forward_context(missing)
        with self.assertRaises(ValueError):
            validate_source_forward_context(dict(original, extra='not-permitted'))
        for key, value in original.items():
            wrong = 1 if type(value) is bool else True
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_source_forward_context(dict(original, **{key: wrong}))
        for invalid in (None, [], tuple(original.items())):
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ValueError):
                validate_source_forward_context(invalid)
        for native in (1, 'true', None):
            with self.subTest(native=native), self.assertRaises(ValueError):
                validate_source_forward_context(original, native=native)

    def test_bad_device_precision_versions_and_inference_values_are_rejected(self):
        original = native_fixture()
        for changes in (dict(schema='other'), dict(device_type='mps'), dict(device_index=-1),
                dict(device_index=None), dict(device_type='cpu', device_index=0),
                dict(input_dtype='torch.float64'), dict(grad_enabled=True),
                dict(policy_training=True), dict(float32_matmul_precision='unknown'),
                dict(torch_version=''), dict(torch_cuda_version=''), dict(cudnn_version=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_source_forward_context(dict(original, **changes))
        unchanged = copy.deepcopy(original)
        validate_source_forward_context(original, native=True)
        self.assertEqual(original, unchanged)

    def test_getter_requires_actual_no_grad_eval_float32_matching_device(self):
        policy, observation, _ = cpu_readback()
        with self.assertRaises(ValueError):
            source_forward_context(policy, observation)
        with torch.no_grad(), self.assertRaises(ValueError):
            source_forward_context(policy, observation.double())
        with torch.no_grad(), self.assertRaises(ValueError):
            source_forward_context(policy, torch.empty(4, 3, device='meta'))
        policy.train()
        with torch.no_grad(), self.assertRaises(ValueError):
            source_forward_context(policy, observation)


if __name__ == '__main__':
    unittest.main()
