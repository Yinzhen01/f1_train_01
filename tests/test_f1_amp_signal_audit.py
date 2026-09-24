import unittest
import torch

from tools.amp.verify_signal_formal import assert_equal, inspect_tracebacks


class SignalAuditTests(unittest.TestCase):
    def test_exact_sdk_traceback_is_recorded_not_hidden(self):
        trace = ('Traceback (most recent call last):\n'
                 '  File "/env/site-packages/pika/adapters/utils/io_services_utils.py", line 110, in _on_readable\n'
                 '    callback()\n'
                 'ConnectionResetError: [Errno 104] Connection reset by peer\n')
        self.assertEqual(inspect_tracebacks('completed'), 0)
        self.assertEqual(inspect_tracebacks(trace+trace), 2)
        for wrong in (trace.replace('io_services_utils.py', 'other.py'),
                      trace.replace('ConnectionResetError:', 'RuntimeError:'),
                      'Traceback (most recent call last):\ntruncated'):
            with self.assertRaises(AssertionError): inspect_tracebacks(wrong)

    def test_learning_state_comparison_includes_nested_optimizer_tensors(self):
        a = {'weights': torch.tensor([1., 2.]), 'optimizer': [dict(step=1250, lr=5e-5)]}
        b = {'weights': a['weights'].clone(), 'optimizer': [dict(step=1250, lr=5e-5)]}
        assert_equal(a, b)
        b['optimizer'][0]['step'] = 1251
        with self.assertRaises(AssertionError): assert_equal(a, b)
        b['optimizer'][0]['step'] = 1250
        b['weights'][0] += .001
        with self.assertRaises(AssertionError): assert_equal(a, b)


if __name__ == '__main__':
    unittest.main()
