import csv
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ALGORITHM = Path(__file__).resolve().parent / "algorithm"
sys.path.insert(0, str(ALGORITHM))

import core
import sherma
import test_suite
from oracle.build import build_oracle, validate_manifest


class SetOracle:
    def __init__(self, accepted):
        self.accepted = set(accepted)
        self.batches = []

    def accepts_batch(self, values, deadline=None):
        self.batches.append(list(values))
        return [value in self.accepted for value in values]


class TimerTests(unittest.TestCase):
    def test_run_progress_reports_elapsed_time_and_cases(self):
        output = io.StringIO()
        with patch.object(test_suite.time, "monotonic", side_effect=(10.0, 15.0)):
            timer = test_suite.RunProgress(2)
            timer.case_name = "example"
            with patch("sys.stdout", output):
                timer._write()
        self.assertEqual(output.getvalue(), "[5s] example: test cases 0/2\n")

    def test_worker_count_uses_available_cpus_up_to_job_count(self):
        with patch.object(test_suite.os, "process_cpu_count", return_value=12,
                          create=True):
            self.assertEqual(test_suite.available_workers(8), 8)
            self.assertEqual(test_suite.available_workers(20), 12)

    def test_smoke_and_full_modes_build_one_oracle_per_dataset(self):
        smoke = test_suite.cases("smoke")
        full = test_suite.cases("test")
        self.assertEqual([case.name for case in smoke], ["adult_20"])
        self.assertEqual(len(full), 8)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                test_suite, "build_oracle", side_effect=lambda _manifest, output: output
        ) as build:
            root = Path(directory)
            smoke_oracles = test_suite.build_dataset_oracles(smoke, root / "smoke")
            self.assertEqual(set(smoke_oracles), {"adult_20"})
            self.assertEqual(build.call_count, 1)
            build.reset_mock()
            full_oracles = test_suite.build_dataset_oracles(full, root / "test")
            self.assertEqual(set(full_oracles), {
                "adult_20", "bank_marketing_222",
                "census_income_kdd_117", "support2_880",
            })
            self.assertEqual(build.call_count, 4)
            self.assertTrue(all(path.name in {"oracle", "oracle.exe"}
                                for path in full_oracles.values()))

def core_config(**changes):
    values = dict(n=2, ngrams_batch_size=core.UNLIMITED,
                  max_candidate_length=core.UNLIMITED, max_iterations=8,
                  max_states=100, max_queue_size=core.UNLIMITED,
                  max_rsr_candidates=core.UNLIMITED,
                  positive_capacity=100, negative_capacity=100, seed=0)
    values.update(changes)
    return core.Config(**values)


class CoreTests(unittest.TestCase):
    def test_pta_and_rsr(self):
        pta = core.build_pta(["", "ab", "ac"], 100)
        self.assertTrue(pta.accepts(""))
        self.assertTrue(pta.accepts("ab"))
        self.assertFalse(pta.accepts("a"))
        repairs = core.rsr_minimum_repairs(pta, "ad")
        self.assertEqual([repair.value for repair in repairs], ["ab", "ac"])

    def test_rpni2_refines_an_accepted_negative_without_history(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab", "ac"], core_config(n=0), measurements)
        overgeneralized = next(
            value for value in ("", "a", "b", "aa", "abc", "cab")
            if value not in session.positives and session.partition.accepts(session.pta, value))
        session._learn_negative(overgeneralized, time.monotonic() + 5, measurements)
        self.assertFalse(session.partition.accepts(session.pta, overgeneralized))
        self.assertTrue(all(session.partition.accepts(session.pta, value)
                            for value in session.positives))
        self.assertFalse(hasattr(session, "history"))
        self.assertGreater(measurements.split_execution_time_ns, 0)

    def test_rpni2_uses_the_paper_acceptance_prefix(self):
        pta = core.build_pta(["", "a", "bb", "bba", "baab", "baaaba"], 100)
        by_prefix = {state.prefix: index for index, state in enumerate(pta.states)}
        partition = core.PartitionedDfa.from_blocks(pta, [
            {by_prefix[value] for value in ("", "a", "ba", "baa", "baaa", "baaaba")},
            {by_prefix[value] for value in ("b", "baab", "baaab")},
            {by_prefix["bb"]},
            {by_prefix["bba"]},
        ])
        order = core._state_order(pta)
        rank = {state: position for position, state in enumerate(order)}
        self.assertTrue(partition.accepts(pta, "b"))
        self.assertEqual(
            core._acceptance_prefix_rank(pta, partition, {"b", "abbb"}, rank),
            rank[by_prefix["baab"]],
        )
        with patch.object(core, "rejects_all", wraps=core.rejects_all) as rejects:
            refined = core.refine_for_negatives(pta, partition, {"b", "abbb"})
        self.assertEqual(rejects.call_count, 2)
        self.assertFalse(refined.accepts(pta, "b"))
        self.assertFalse(refined.accepts(pta, "abbb"))

    def test_cell_timeout_interrupts_rsr_inner_work(self):
        pta = core.build_pta(["ab", "ac"], 100)
        checks = 0

        def interrupt(_deadline):
            nonlocal checks
            checks += 1
            if checks == 5:
                raise RuntimeError("cell timeout")

        with patch.object(core, "_check_deadline", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "cell timeout"):
                core.rsr_minimum_repairs(pta, "ad", deadline=time.monotonic() + 5)
        self.assertEqual(checks, 5)

    def test_rsr_queue_limit_is_enforced(self):
        pta = core.build_pta(["a", "b"], 10)
        with self.assertRaisesRegex(RuntimeError, "queue size"):
            core.rsr_minimum_repairs(pta, "", max_queue_size=1)
        self.assertEqual(len(core.rsr_minimum_repairs(
            pta, "", max_queue_size=2)), 2)

    def test_rsr_reconstructs_candidates_without_python_recursion(self):
        value = "a" * 1100
        pta = core.build_pta([value], 1200)
        self.assertEqual(core.rsr_minimum_repairs(pta, "")[0].value, value)

    def test_timed_out_negative_does_not_evict_retained_example(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(
            ["ab", "ac"], core_config(n=0, negative_capacity=1), measurements)
        session._learn_negative("zz", time.monotonic() + 5, measurements)
        accepted = next(value for value in ("", "a", "b", "aa", "abc", "cab")
                        if session.partition.accepts(session.pta, value))
        with patch.object(core, "refine_for_negatives",
                          side_effect=RuntimeError("cell timeout")):
            with self.assertRaisesRegex(RuntimeError, "cell timeout"):
                session._learn_negative(
                    accepted, time.monotonic() + 5, measurements)
        self.assertEqual(session.negatives.values(), {"zz"})

    def test_rpni2_config_defaults_and_validation(self):
        base = {
            "repair": {"n": 2, "ngrams_batch_size": 2},
            "batching": {"initial_positive_capacity": 5,
                         "initial_negative_capacity": 5,
                         "positive_capacity": 5, "negative_capacity": 5,
                         "cell_timeout_seconds": 5},
            "limits": {"max_iterations": 5, "max_states": 20},
        }
        parsed = sherma.parse_config(base)
        self.assertEqual(parsed.core.seed, 0)
        defaults = dict(base)
        defaults["batching"] = {"cell_timeout_seconds": 5}
        default_config = sherma.parse_config(defaults)
        self.assertEqual(default_config.num_candidates, 5)
        self.assertEqual(default_config.edit_distance_cap, 3)
        self.assertEqual(default_config.jev_confidence_threshold, 0.5)
        self.assertEqual(default_config.jev_timeout_seconds, 15)
        self.assertEqual(default_config.initial_positive_capacity, core.UNLIMITED)
        self.assertEqual(default_config.initial_negative_capacity, core.UNLIMITED)
        self.assertEqual(default_config.core.positive_capacity, core.UNLIMITED)
        self.assertEqual(default_config.core.negative_capacity, core.UNLIMITED)
        base["batching"]["positive_capacity"] = 4
        with self.assertRaisesRegex(RuntimeError, "at least"):
            sherma.parse_config(base)
        base["batching"]["positive_capacity"] = 5
        base["legacy"] = {"k": 0}
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            sherma.parse_config(base)

    def test_candidate_config_validation(self):
        base = {
            "repair": {"n": 2, "ngrams_batch_size": 2,
                       "num_candidates": 0, "edit_distance_cap": 3},
            "batching": {"cell_timeout_seconds": 5},
            "limits": {"max_iterations": 5, "max_states": 20},
        }
        with self.assertRaisesRegex(RuntimeError, "positive"):
            sherma.parse_config(base)
        base["repair"]["num_candidates"] = 5
        base["repair"]["edit_distance_cap"] = -2
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            sherma.parse_config(base)

    def test_initial_dfa_uses_negative_examples(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(
            ["ab"], core_config(n=0), measurements, [""])
        self.assertIn("", session.negatives)
        self.assertFalse(session.partition.accepts(session.pta, ""))

    def test_initial_rpni2_starts_general_and_splits_for_negatives(self):
        general = core.BatchSession(["ab", "ac"], core_config(n=0))
        self.assertEqual(len(general.partition.roots()), 1)
        self.assertTrue(general.partition.accepts(general.pta, ""))
        constrained = core.BatchSession(
            ["ab", "ac"], core_config(n=0), negatives=[""])
        self.assertFalse(constrained.partition.accepts(constrained.pta, ""))
        self.assertTrue(constrained.partition.accepts(constrained.pta, "ab"))
        self.assertTrue(constrained.partition.accepts(constrained.pta, "ac"))

    def test_capacity_eviction_keeps_dfa_and_ngram_effects(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(
            ["xa", "yb"], core_config(n=2, positive_capacity=2), measurements)
        session._learn_positive("zc", time.monotonic() + 5, measurements)
        self.assertNotIn("xa", session.positives)
        self.assertTrue(session.partition.accepts(session.pta, "xa"))
        self.assertIn("xa", session.all_positives)
        self.assertIn(ord("x"), session.model.vocabulary)

    def test_batch_session_learns_rejected_prefix(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=0), measurements)
        oracle = SetOracle({"ab"})
        self.assertEqual(session.repair("az", oracle, time.monotonic() + 5, measurements), "ab")
        queried = [value for batch in oracle.batches for value in batch]
        rejected = queried[:queried.index("ab")]
        self.assertIn("az", session.negatives.members)
        self.assertTrue(set(rejected).issubset(session.negatives.members))
        self.assertIn("ab", session.positives)
        self.assertTrue(session.partition.accepts(session.pta, "ab"))
        self.assertEqual(measurements.oracle_total_calls, 1)
        self.assertEqual(measurements.oracle_candidates_submitted, len(oracle.batches[0]))

    def test_oracle_accepted_repair_extends_dfa_and_updates_ngrams(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=2), measurements)
        original_pta = session.pta
        original_states = {
            state.prefix: index for index, state in enumerate(session.pta.states)
        }
        old_state_count = len(session.pta.states)
        old_score = session.model.score("ac")
        oracle = SetOracle({"ac"})
        with patch.object(core, "rsr_repairs",
                          return_value=[core.RepairCandidate("ac", 1)]):
            repaired = session.repair(
                "zz", oracle, time.monotonic() + 5, measurements)
        self.assertEqual(repaired, "ac")
        self.assertIs(session.pta, original_pta)
        self.assertEqual(
            {state.prefix: index for index, state in enumerate(session.pta.states)
             if state.prefix in original_states},
            original_states,
        )
        self.assertIn("ac", session.positives)
        self.assertTrue(session.partition.accepts(session.pta, "ac"))
        self.assertGreater(len(session.pta.states), old_state_count)
        self.assertGreater(session.model.score("ac"), old_score)

    def test_collects_multiple_candidates_and_checks_distance_before_oracle(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=0), measurements)
        oracle = SetOracle({"ac", "ad"})
        with patch.object(core, "rsr_repairs", return_value=[
                core.RepairCandidate("ac", 1), core.RepairCandidate("ad", 1),
                core.RepairCandidate("faraway", 7)]):
            candidates, failure = session.repair_candidates(
                "az", oracle, time.monotonic() + 5, measurements, 2, 2)
        self.assertEqual(candidates, ["ac", "ad"])
        self.assertEqual(failure, "")
        self.assertNotIn("faraway", [value for batch in oracle.batches for value in batch])
        self.assertTrue({"ac", "ad"}.issubset(session.positives.values()))

    def test_full_oracle_batch_may_exceed_candidate_minimum(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=0), measurements)
        with patch.object(core, "rsr_repairs", return_value=[
                core.RepairCandidate("ac", 1), core.RepairCandidate("ad", 1)]):
            candidates, failure = session.repair_candidates(
                "az", SetOracle({"ac", "ad"}), time.monotonic() + 5,
                measurements, 1, 3)
        self.assertEqual(candidates, ["ac", "ad"])
        self.assertEqual(failure, "")

    def test_distance_cap_can_return_no_candidates_without_oracle_call(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=0), measurements)
        oracle = SetOracle({"faraway"})
        with patch.object(core, "rsr_repairs",
                          return_value=[core.RepairCandidate("faraway", 7)]):
            candidates, failure = session.repair_candidates(
                "az", oracle, time.monotonic() + 5, measurements, 5, 3)
        self.assertEqual(candidates, [])
        self.assertEqual(failure, "edit-distance cap reached")
        self.assertEqual(oracle.batches, [])

    def test_timeout_returns_candidates_collected_so_far(self):
        class TimeoutAfterFirst(SetOracle):
            def accepts_batch(self, values, deadline=None):
                if self.batches:
                    raise RuntimeError("cell timeout")
                return super().accepts_batch(values, deadline)

        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=0), measurements)
        oracle = TimeoutAfterFirst({"ac", "ad"})
        with patch.object(core, "rsr_repairs", side_effect=[
                [core.RepairCandidate("ac", 1)],
                [core.RepairCandidate("ad", 1)]]):
            candidates, failure = session.repair_candidates(
                "az", oracle, time.monotonic() + 5, measurements, 2, 3)
        self.assertEqual(candidates, ["ac"])
        self.assertEqual(failure, "cell timeout")
        self.assertNotIn("ac", session.positives)

    def test_positive_extension_refines_if_it_revives_a_negative(self):
        measurements = core.AlgorithmMeasurements()
        session = core.BatchSession(["ab"], core_config(n=0), measurements)
        session._learn_negative("", time.monotonic() + 5, measurements)
        session._learn_positive("a", time.monotonic() + 5, measurements)
        self.assertTrue(session.partition.accepts(session.pta, "a"))
        self.assertFalse(session.partition.accepts(session.pta, ""))

    def test_example_fifo_evicts_oldest(self):
        fifo = core.ExampleFifo(2)
        fifo.add("a")
        fifo.add("b")
        fifo.add("c")
        self.assertEqual(fifo.values(), {"b", "c"})


class FakeExternalOracle:
    def __init__(self, executable, column):
        self.column = column
        self.calls = 0
        self.candidates_submitted = 0
        self.cache = {}

    def accepts_batch(self, values, deadline=None):
        result = []
        queried = False
        for value in values:
            if value not in self.cache:
                queried = True
                self.candidates_submitted += 1
                self.cache[value] = (
                    value in {"A", "B"} if self.column == "kind" else
                    value in {"1", "2", "3", "10", "11", "12"})
            result.append(self.cache[value])
        self.calls += queried
        return result


class PipelineTests(unittest.TestCase):
    def test_oracle_manifest_rejects_legacy_and_unknown_fields(self):
        manifest = {
            "dataset": "sample", "source": "clean.csv",
            "oracles": {"value": {
                "type": "enum", "values": ["a"],
            }},
        }
        self.assertEqual(validate_manifest(manifest), manifest)
        for field, value in (("schema_version", 2), ("unknown", True)):
            legacy = {**manifest, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, "unsupported fields"):
                validate_manifest(legacy)
        legacy_rule = {**manifest, "oracles": {"value": {
            **manifest["oracles"]["value"], "accepts_empty": False}}}
        with self.assertRaisesRegex(RuntimeError, "accepts_empty"):
            validate_manifest(legacy_rule)
        legacy_array = {**manifest, "oracles": [manifest["oracles"]["value"]]}
        with self.assertRaisesRegex(RuntimeError, "object keyed by column"):
            validate_manifest(legacy_array)
        old_empty_policy = {**manifest, "empty_policy": "rejected"}
        with self.assertRaisesRegex(RuntimeError, "empty_policy"):
            validate_manifest(old_empty_policy)

    def test_oracle_compiler_requires_dataset_filename(self):
        manifest = {
            "dataset": "sample", "source": "clean.csv",
            "oracles": {"value": {
                "type": "enum", "values": ["a"],
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wrong-name.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "<dataset>.json"):
                build_oracle(source, root / "oracle.exe")

    def test_candidate_output_name_replaces_the_requested_suffix(self):
        self.assertEqual(sherma.candidate_output_path(Path("output.csv")),
                         Path("output.tmp.csv"))
        self.assertEqual(sherma.candidate_output_path(Path("output")),
                         Path("output.tmp.csv"))

    def test_fatal_failure_writes_error_log_and_removes_stale_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.csv"
            telemetry = root / "telemetry.json"
            output.write_text("stale", encoding="utf-8")
            telemetry.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot open config"):
                sherma.run_pipeline(
                    root / "input.csv", root / "truth.csv", root / "schema.json",
                    root / "missing-config.json", root / "oracle", output,
                    telemetry)
            log = (root / "error.log").read_text(encoding="utf-8")
            self.assertIn("RuntimeError: cannot open config", log)
            self.assertIn("Traceback", log)
            self.assertFalse(output.exists())
            self.assertFalse(telemetry.exists())
            self.assertTrue(sherma.candidate_output_path(output).exists())

    def test_external_oracle_batch_protocol(self):
        oracle = sherma.ExternalOracle(Path("oracle"), "age")
        with patch.object(sherma.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="[false,true]")) as run:
            self.assertEqual(oracle.accepts_batch(["a", "b"]), [False, True])
            self.assertEqual(json.loads(run.call_args.kwargs["input"]),
                             {"column": "age", "candidates": ["a", "b"]})
            self.assertEqual(oracle.accepts_batch(["a", "b"]), [False, True])
            self.assertEqual(run.call_count, 1)
        for output in ("[]", "[false]", "[1]", "not-json"):
            with self.subTest(output=output), patch.object(
                    sherma.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=output)):
                with self.assertRaises(RuntimeError):
                    sherma.ExternalOracle(Path("oracle"), "age").accepts_batch(["a", "b"])

    def test_external_oracle_obeys_deadline(self):
        with patch.object(sherma.subprocess, "run",
                          side_effect=sherma.subprocess.TimeoutExpired("oracle", 1)):
            with self.assertRaisesRegex(RuntimeError, "cell timeout"):
                sherma.ExternalOracle(Path("oracle"), "age").accepts_batch(
                    ["a"], time.monotonic() + 1)

    def test_row_jev_request_contains_all_candidate_sets(self):
        response = {
            "model": sherma.JEV_MODEL,
            "answers": {
                "column_1": {"choice": "candidate_1", "confidence": 0.8},
                "column_2": {"choice": "candidate_0", "confidence": 0.9},
            },
        }
        with patch.dict(sherma.os.environ, {"TYPESAFE_API_KEY": "secret"}), \
                patch.object(sherma, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = (
                json.dumps(response).encode())
            selected = sherma.select_row_with_jev(
                ["age", "job", "income"],
                {"age": "Age", "job": "Job", "income": "Income"},
                ["30", "?", "bad"], {1: ["A", "B"], 2: ["1"]}, 15)
        self.assertEqual(selected[1].candidate, 1)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["state"]["fixed_values"], {"age": "30"})
        self.assertEqual(payload["state"]["candidate_sets"],
                         {"job": ["A", "B"], "income": ["1"]})
        self.assertEqual(set(payload["questions"]), {"column_1", "column_2"})

    def test_row_jev_rejects_an_unexpected_model(self):
        response = {
            "model": "unexpected",
            "answers": {"column_0": {
                "choice": "candidate_0", "confidence": 1.0}},
        }
        with patch.dict(sherma.os.environ, {"TYPESAFE_API_KEY": "secret"}), \
                patch.object(sherma, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = (
                json.dumps(response).encode())
            with self.assertRaisesRegex(RuntimeError, "invalid Jev response"):
                sherma.select_row_with_jev(
                    ["value"], {"value": "Value"}, ["?"], {0: ["A"]}, 15)

    def test_csv_pipeline_generates_candidates_and_fills_missing_cells(self):
        config = {
            "seed": 0,
            "repair": {"n": 2, "ngrams_batch_size": -1,
                       "max_candidate_length": -1},
            "batching": {"initial_positive_capacity": -1,
                         "initial_negative_capacity": -1,
                         "positive_capacity": -1, "negative_capacity": -1,
                         "cell_timeout_seconds": 5},
            "limits": {"max_iterations": 8, "max_states": 100,
                       "max_queue_size": -1,
                       "max_rsr_candidates": -1},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "corrupted.csv"
            ground_truth_path = root / "original.csv"
            schema_path = root / "schema.json"
            config_path = root / "config.json"
            oracle = root / "oracle"
            output = root / "repaired.csv"
            telemetry = root / "telemetry.json"
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows([
                    ["kind", "value"], ["A", "1"], ["A", "3"],
                    ["B", "10"], ["B", "12"], ["A", "?"], ["B", "bad"],
                ])
            with ground_truth_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows([
                    ["kind", "value"], ["A", "1"], ["A", "3"],
                    ["B", "10"], ["B", "12"], ["A", "2"], ["B", "11"],
                ])
            config_path.write_text(json.dumps(config), encoding="utf-8")
            schema_path.write_text(json.dumps({
                "schema_version": 1,
                "columns": {"kind": "Category", "value": "Value"},
            }), encoding="utf-8")
            sherma.candidate_output_path(output).write_text("stale\n", encoding="utf-8")
            telemetry.write_text("stale\n", encoding="utf-8")

            def select(_columns, _descriptions, _row, candidate_sets, _timeout):
                return {column: sherma.JevSelection(0, 0.4, sherma.JEV_MODEL)
                        for column in candidate_sets}

            with patch.object(sherma, "oracle_path", return_value=oracle), \
                    patch.object(sherma, "ExternalOracle", FakeExternalOracle), \
                    patch.object(sherma, "select_row_with_jev", side_effect=select):
                with patch.dict(sherma.os.environ,
                                {"TYPESAFE_API_KEY": "test-key"}):
                    sherma.run_pipeline(input_path, ground_truth_path, schema_path,
                                        config_path, oracle, output, telemetry)
            candidate_output = sherma.candidate_output_path(output)
            with candidate_output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertTrue(output.exists())
            self.assertEqual(rows[0], ["row", "column", "malformed_value",
                                       "candidate_rank", "candidate"])
            self.assertTrue(any(row[0] == "5" and row[1] == "value"
                                for row in rows[1:]))
            self.assertFalse(any(row[0] == "4" and row[1] == "value"
                                 for row in rows[1:]))
            report = json.loads(telemetry.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 11)
            self.assertEqual(report["input_correct_cells"], 10)
            self.assertGreaterEqual(report["final_correct_cells"], 10)
            self.assertEqual(report["unresolved_cells"], 0)
            self.assertGreater(report["jev_low_confidence"], 0)
            missing = next(cell for cell in report["cells"]
                           if cell["row"] == 4 and cell["column"] == "value")
            self.assertEqual(missing["unresolved_reason"], "")
            self.assertEqual(missing["jev_model"], sherma.JEV_MODEL)
            self.assertEqual(missing["jev_fallback_reason"], "low_confidence")

    def test_config_rejects_old_batch_limit(self):
        value = {
            "repair": {"n": 2, "ngrams_batch_size": 2, "rsr_batch_size": 2},
            "batching": {"initial_positive_capacity": 5,
                         "initial_negative_capacity": 5,
                         "positive_capacity": 5, "negative_capacity": 5,
                         "cell_timeout_seconds": 5},
            "limits": {"max_iterations": 5, "max_states": 20},
        }
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            sherma.parse_config(value)

    def test_config_rejects_removed_total_oracle_limit(self):
        value = {
            "repair": {"n": 2, "ngrams_batch_size": 2},
            "batching": {"cell_timeout_seconds": 5},
            "limits": {"max_iterations": 5, "max_states": 20,
                       "max_total_oracle_calls": -1},
        }
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            sherma.parse_config(value)

    def test_csv_rejects_non_ascii_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("column\n\u00e9\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-ASCII"):
                sherma.read_csv(path)


if __name__ == "__main__":
    unittest.main()
