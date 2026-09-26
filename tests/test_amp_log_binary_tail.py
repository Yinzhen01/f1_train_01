import unittest
from unittest.mock import Mock

from tools.amp.verify_horizon_formal import decode_post_completion_log, read_cloud_log


class BinaryTailTests(unittest.TestCase):
    prefix = (b'[amp-eval-complete] saved\n[f1-amp-complete] {"complete":true,"all_finite":true}\n'
              b'[SDK][INFO] PT file uploaded successfully\n')

    def test_only_opt_in_tail_escaped_without_hiding_ascii_errors(self):
        raw = self.prefix+b'\0\0\xb0\x051\nTraceback: real failure\nRL task is failed.\n'
        diag = {}; result = decode_post_completion_log(raw, diag)
        self.assertIn(r'\x00\x00\xb0\x05', result)
        self.assertIn('Traceback: real failure', result)
        self.assertIn('RL task is failed.', result)
        self.assertEqual(diag['nul_count'], 2)
        self.assertEqual(diag['invalid_utf8_byte_count'], 1)
        self.assertGreater(diag['first_binary_byte_offset'], diag['completion_record_end_byte'])
        path = Mock(suffix='.log', read_text=Mock(side_effect=UnicodeDecodeError('utf-8', b'\xb0', 0, 1, 'invalid')), read_bytes=Mock(return_value=raw))
        with self.assertRaises(UnicodeDecodeError): read_cloud_log(path)
        self.assertEqual(read_cloud_log(path, allow_post_completion_binary=True), result)

    def test_corruption_before_or_inside_complete_record_is_rejected(self):
        for raw in (b'\0'+self.prefix, self.prefix.replace(b'true', b'\xb0', 1),
                    self.prefix.replace(b'complete":true', b'complete":false')+b'\0',
                    self.prefix.replace(b'[amp-eval-complete]', b'no-evaluation')+b'\0',
                    self.prefix.replace(b'uploaded successfully', b'upload failed')+b'\0'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeDecodeError)):
                decode_post_completion_log(raw, {})

    def test_valid_unicode_preserved_and_nul_only_still_requires_opt_in(self):
        raw = self.prefix+'正常结束\n'.encode('utf-8')
        self.assertEqual(decode_post_completion_log(raw, {}), raw.decode('utf-8'))
        path = Mock(suffix='.log', read_text=Mock(return_value=raw.decode('utf-8')+'\0'), read_bytes=Mock(return_value=raw+b'\0'))
        with self.assertRaises(ValueError): read_cloud_log(path)
        self.assertIn('正常结束', read_cloud_log(path, allow_post_completion_binary=True))
        path.suffix = '.json'
        with self.assertRaises(ValueError): read_cloud_log(path, allow_post_completion_binary=True)


if __name__ == '__main__': unittest.main()
