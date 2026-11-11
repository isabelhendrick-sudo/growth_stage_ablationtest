"""Checks for interpretation precedence, chronology and published report numbers."""
import ast
import csv
import json
from pathlib import Path
import unittest
from interpretation import interpret_results, interval_verdict

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class InterpretationTests(unittest.TestCase):
    def summary(self, lower, run="leaf_crop", deltas=None):
        return interpret_results(
            [{"arm": "no_stage", "macro_f1": 0.7, "ci_lo": lower, "ci_hi": 0.8}],
            deltas or [{"comparison": a + " vs no_stage", "ci_lo": 0.02, "ci_hi": 0.04}
                       for a in ("stage_additive", "stage_interaction")], run)

    def test_failed_gate_overrides_apparently_useful_stage_effect(self):
        result = self.summary(0.59)
        self.assertEqual(result["stage_comparison_status"], "not_interpreted")
        self.assertTrue(all(v.startswith("NOT INTERPRETED") for v in result["verdicts"]))
        self.assertNotIn("supports a gain", result["decision"])

    def test_gate_uses_lower_bound_and_is_strict(self):
        self.assertEqual(self.summary(0.60)["stage_comparison_status"], "not_interpreted")
        self.assertEqual(self.summary(0.600001)["stage_comparison_status"], "interpreted")

    def test_gate_not_retrospectively_applied_to_run_one(self):
        result = self.summary(0.477, "full_image")
        self.assertFalse(result["baseline_gate_applies"])
        self.assertEqual(result["stage_comparison_status"], "interpreted")

    def test_positive_effect_is_not_automatically_practically_meaningful(self):
        self.assertTrue(interval_verdict(0.001, 0.03).startswith("POSITIVE"))
        self.assertTrue(interval_verdict(0.011, 0.03).startswith("HELPS"))
        self.assertTrue(interval_verdict(-0.005, 0.005).startswith("EQUIVALENT"))
        self.assertTrue(interval_verdict(-0.04, 0.04).startswith("INCONCLUSIVE"))

    def test_published_runs_have_report_decisions(self):
        overall, deltas = rows("results/overall_comparison.csv"), rows("results/paired_comparisons.csv")
        for run in ("full_image", "leaf_crop"):
            result = interpret_results([r for r in overall if r["experiment"] == run],
                                       [r for r in deltas if r["experiment"] == run], run)
            expected = "not_interpreted" if run == "leaf_crop" else "interpreted"
            self.assertEqual(result["stage_comparison_status"], expected)
        self.assertTrue(interpret_results(overall[:6], deltas[:4], "full_image")["decision"].startswith("INCONCLUSIVE"))

    def test_confusion_matrix_reconstructs_published_score(self):
        cm = rows("results/leaf_crop/confusion_matrix.csv")
        a, b = int(cm[0]["predicted_tipburn"]), int(cm[0]["predicted_yellowing"])
        c, d = int(cm[1]["predicted_tipburn"]), int(cm[1]["predicted_yellowing"])
        self.assertEqual(a+b+c+d, 1902)
        score = (2*a/(2*a+b+c) + 2*d/(2*d+b+c))/2
        baseline = next(r for r in rows("results/overall_comparison.csv")
                        if r["experiment"] == "leaf_crop" and r["arm"] == "no_stage")
        self.assertAlmostEqual(score, float(baseline["macro_f1"]), places=14)
        self.assertAlmostEqual(a/(a+b), float(cm[0]["recall"]), places=14)
        self.assertAlmostEqual(d/(c+d), float(cm[1]["recall"]), places=14)

    def test_coverage_matches_report_table_two(self):
        coverage = rows("results/stage_coverage.csv")
        expected = {"full_image": [(18,13),(28,16),(14,10),(27,22),(28,19)],
                    "leaf_crop": [(18,13),(28,16),(12,8),(24,19),(28,17)]}
        for run, pairs in expected.items():
            subset = [r for r in coverage if r["experiment"] == run]
            self.assertEqual([int(r["plants"]) for r in subset], [n for pair in pairs for n in pair])
            self.assertEqual(sum(int(r["images"]) for r in subset), 1902 if run == "leaf_crop" else 3503)

    def test_notebook_code_is_valid_and_has_no_stale_execution(self):
        nb = json.loads((ROOT / "ablation_test.ipynb").read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
