#!/usr/bin/env python3
"""Regression checks for alias resolution and statistical meaning."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
from library_metadata import build_summary, event_rows, interface_from_text, ncu_rows

INPUT = 'def _make_inputs(n, device="cuda"):\n    pass\n'


class MetadataTests(unittest.TestCase):
    def test_forward_alias_keeps_keyword_only_signature(self):
        source = 'class Model:\n def __call__(self, x, *, out=None): pass\n forward = __call__\n'
        api = interface_from_text(source, INPUT)
        self.assertEqual(api['forward'], 'self, x, *, out=None')
        self.assertEqual(api['forward_target'], '__call__')
        self.assertIsNone(api['__init__'])

    def test_alias_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            interface_from_text('class Model:\n forward = run\n run = forward\n', INPUT)

    def test_prepare_hook_is_recorded(self):
        api = interface_from_text('class Model:\n def forward(self, x): pass\n',
                                  INPUT + 'def prepare_model(model_class, inputs, init_kwargs, input_kwargs): pass\n')
        self.assertEqual(api['prepare_model'], 'model_class, inputs, init_kwargs, input_kwargs')

    def test_paired_ratio_is_not_ratio_of_medians(self):
        record = {'shapes': {'0': {'R_median_us': 10, 'N0_median_us': 5,
                                   'median_paired_R_over_N0': 1.7,
                                   'bootstrap_95pct_interval': [1.5, 1.9]}}}
        row = event_rows(record, ['0'])[0]
        self.assertEqual(row['candidate_over_baseline'], 1.7)
        self.assertEqual((row['candidate_us'], row['baseline_us']), (10, 5))
        self.assertEqual(row['ci95'], [1.5, 1.9])

    def test_balanced_ratios_do_not_mix_absolute_latencies(self):
        record = {'shapes': {'0': {'R_median_us': 999, 'N0_median_us': 222,
                                   'balanced_R_over_N0': 0.9, 'N0_first_median_ratio': 0.8,
                                   'R_first_median_ratio': 1.0125}}}
        row = event_rows(record, ['0'], balanced=True)[0]
        self.assertIsNone(row['candidate_us'])
        self.assertIsNone(row['baseline_us'])
        self.assertEqual(row['candidate_over_baseline'], 0.9)
        self.assertEqual(row['by_order']['R_first']['candidate_over_baseline'], 1.0125)

    def test_missing_baseline_is_not_zero_or_passed(self):
        row = ncu_rows({'shapes': {}, 'unavailable_comparisons': {'0': {}}}, ['0'])[0]
        self.assertEqual(row['status'], 'baseline_unsupported')
        self.assertIsNone(row['baseline_us'])
        self.assertIsNone(row['candidate_over_baseline'])

    def test_ncu_order_intervals_are_preserved(self):
        data = {'shapes': {'0': {'candidate_us': 9, 'production_us': 10,
                                 'orders': {'PCCP': {'C_over_P': 0.88, 'bootstrap_95pct': [0.85, 0.95]}}}}}
        row = ncu_rows(data, ['0'])[0]
        self.assertEqual(row['candidate_over_baseline'], 0.9)
        self.assertEqual(row['by_order']['PCCP'], {'candidate_over_baseline': 0.88, 'ci95': [0.85, 0.95]})

    def test_manual_waiver_remains_a_waiver(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / 'problem').mkdir()
            (p / 'performance').mkdir()
            (p / 'problem/shape_cases.json').write_text('{"0": {}}')
            raw = {'metric': 'NCU gpu__time_duration.sum',
                   'status': 'accepted_by_user_with_performance_waiver',
                   'user_acceptance': {'explicit': True},
                   'performance': {'shapes': {'0': {'candidate_us': 12, 'production_us': 10,
                                                    'status': 'over_1pct'}}}}
            (p / 'performance/ACCEPTANCE.json').write_text(json.dumps(raw))
            entry = {'operator': 'example', 'solution_origin': 'REFERENCE',
                     'candidate_files': {}, 'problem_verification': 'compatible_latest_receipt_missing'}
            summary = build_summary(p, entry)
            self.assertTrue(summary['performance_waiver'])
            self.assertEqual(summary['acceptance_status'], raw['status'])
            self.assertEqual(summary['measurements'][0]['rows'][0]['status'], 'over_1pct')


if __name__ == '__main__':
    unittest.main(verbosity=2)
