"""Explicit offline comparison labels; synthetic identities are not training evidence."""
import contextlib
import copy
import io
import unittest

from tools.amp.audit_mu_candidates import candidate_case_names, parse_arguments
from tools.amp.audit_mu_fixed_inputs import (
    candidate_groups, validate_candidate_binding, build_parser)
from test_mu_frozen_candidates_audit import required_arguments
from test_amp_fixed_input_freeze import binding_fixture


FIRST = 'freeze_split_temporal01'
SECOND = 'freeze_split_temporal05'
COMMIT = '1'*40


class ComponentQualityRoutingTests(unittest.TestCase):
    def test_legacy_quality_case_labels_are_unchanged(self):
        args = parse_arguments(required_arguments())
        self.assertEqual(candidate_case_names(args), {'anchor': 'anchor', 'temporal': 'temporal'})

    def test_two_temporal_candidates_have_explicit_not_anchor_case_names(self):
        args = parse_arguments(required_arguments()+['--first-group', FIRST, '--second-group', SECOND])
        self.assertEqual(candidate_case_names(args), {'anchor': 'temporal01', 'temporal': 'temporal05'})

    def test_half_selected_or_swapped_quality_comparison_is_rejected(self):
        for flags in (['--first-group', FIRST], ['--second-group', SECOND],
                      ['--first-group', SECOND, '--second-group', FIRST]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(required_arguments()+flags)

    def test_fixed_input_slots_are_bound_to_the_two_new_registered_groups(self):
        args = build_parser().parse_args(['--output', 'synthetic-output', '--first-group', FIRST,
            '--second-group', SECOND, '--expected-commit', COMMIT])
        self.assertEqual(candidate_groups(args.anchor_group, args.temporal_group, args.expected_commit),
                         {'081': FIRST, '082': SECOND})
        for first, second in ((FIRST, 'temporal'), ('anchor', SECOND), (SECOND, FIRST)):
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                candidate_groups(first, second, COMMIT)

    def test_each_new_identity_is_bound_to_exact_contract_source_and_commit(self):
        for run, group in (('081', FIRST), ('082', SECOND)):
            identity, training, full_hash = binding_fixture(group, COMMIT)
            validate_candidate_binding(run, identity, training, identity.copy(), full_hash, group, COMMIT)
            bad = copy.deepcopy(training)
            bad['continuation']['mu_contract']['temporal_coef'] *= 2
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_candidate_binding(run, identity, bad, identity.copy(), full_hash, group, COMMIT)
            with self.assertRaises(ValueError):
                validate_candidate_binding(run, identity, training, identity.copy(), full_hash, group, '2'*40)


if __name__ == '__main__':
    unittest.main()
