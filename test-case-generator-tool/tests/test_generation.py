import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class GenerationTests(unittest.TestCase):
    def test_refresh_preserves_manual_rules_and_matches_shapes(self):
        manifest = {"oracles": {
            "age": {"type": "regex", "regex": r"^\d{1,2}$",
                    "shapes": [r"\d{2}"]},
            "kind": {"type": "enum", "values": ["custom"]},
        }}
        refreshed = benchmark.refresh_oracles(manifest)
        regex, enum = refreshed["oracles"].values()
        self.assertEqual(regex["regex"], r"^\d{1,2}$")
        self.assertEqual(enum["values"], ["custom"])
        shape_accepts = benchmark.predicate({"type": "regex", "regex": "|".join(regex["shapes"])})
        for value in ("", "?", "0", "9", "10", "99", "100", "abc"):
            self.assertEqual(shape_accepts(value), benchmark.predicate(regex)(value))

    def test_rare_categories_survive_exact_corruption_budget(self):
        rows = [{"category": "rare" if i == 0 else "common", "number": str(i)}
                for i in range(20)]
        specs = {"category": {"type": "enum", "values": ["rare", "common"]},
                 "number": {"type": "regex"}}
        eligible = [(i, col) for i in range(20) for col in rows[i]]
        for seed in range(20):
            selected, _ = benchmark.select_corruptions(rows, specs, eligible, 36, random.Random(seed))
            self.assertEqual(len(set(selected)), 36)
            survivors = {row["category"] for i, row in enumerate(rows)
                         if (i, "category") not in selected}
            self.assertEqual(survivors, {"rare", "common"})
            self.assertEqual(selected, benchmark.select_corruptions(
                rows, specs, eligible, 36, random.Random(seed))[0])

    def test_impossible_budget_fails(self):
        with self.assertRaisesRegex(ValueError, "too few cells"):
            benchmark.select_corruptions([{"a": "unique"}],
                {"a": {"type": "enum"}}, [(0, "a")], 1, random.Random(0))

    def test_peaks_are_maxima_not_sums(self):
        self.assertEqual(benchmark.example_peaks({
            "a": {"max_positive_examples_stored": 10, "max_negative_examples_stored": 3},
            "b": {"max_positive_examples_stored": 2, "max_negative_examples_stored": 20},
        }), {"max_positive_examples_stored": 10, "max_negative_examples_stored": 20})
        self.assertIsNone(benchmark.example_peaks({"legacy": {}})["max_negative_examples_stored"])

    def test_summary_combines_datasets_and_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path, positive, negative in (
                ("generated-50/adult/run/benchmark.json", 30, 4),
                ("generated-90/bank/benchmark.json", 10, 90),
            ):
                benchmark.write_json(root / path, {"table": {
                    "max_positive_examples_stored": positive,
                    "max_negative_examples_stored": negative}})
            benchmark.write_example_summary(root)
            summary = benchmark.read_json(root / "example-storage-summary.json")
            self.assertEqual(summary["max_positive_examples_stored"], 30)
            self.assertEqual(summary["max_negative_examples_stored"], 90)
            self.assertEqual(len(summary["datasets"]), 2)


if __name__ == "__main__":
    unittest.main()
