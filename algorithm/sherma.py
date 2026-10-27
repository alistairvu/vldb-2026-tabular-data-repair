from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from core import UNLIMITED, AlgorithmMeasurements, BatchSession, Config, Oracle


@dataclass(frozen=True)
class PipelineConfig:
    core: Config
    initial_positive_capacity: int
    initial_negative_capacity: int
    cell_timeout_seconds: int
    num_candidates: int
    edit_distance_cap: int
    jev_confidence_threshold: float
    jev_timeout_seconds: int


@dataclass
class CellState:
    timed_out: bool = False
    execution_time_seconds: float = 0.0
    sherma_outcome: str = ""
    unresolved_reason: str = ""
    candidate_count: int = 0
    jev_confidence: float | None = None
    jev_fallback_reason: str = ""
    jev_model: str = ""


@dataclass(frozen=True)
class JevSelection:
    candidate: int
    confidence: float
    model: str


JEV_MODEL = "jev-1.13.0"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def is_missing(value: str) -> bool:
    return value in ("", "?")


class ExternalOracle(Oracle):
    def __init__(self, executable: Path, column: str) -> None:
        self.executable = executable
        self.column = column
        self.calls = 0
        self.candidates_submitted = 0
        self.cache: dict[str, bool] = {}

    def accepts_batch(
        self, values: Sequence[str], deadline: float | None = None
    ) -> list[bool]:
        if not values:
            return []
        unknown = list(
            dict.fromkeys(value for value in values if value not in self.cache)
        )
        if unknown:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError("cell timeout")
            self.calls += 1
            self.candidates_submitted += len(unknown)
            timeout = None if deadline is None else max(0, deadline - time.monotonic())
            try:
                response = subprocess.run(
                    [str(self.executable)],
                    input=json.dumps(
                        {"column": self.column, "candidates": unknown},
                        separators=(",", ":"),
                    ),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("cell timeout") from error
            if response.returncode:
                raise RuntimeError(f"oracle exited with code {response.returncode}")
            try:
                answers = json.loads(response.stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid oracle JSON: {error}") from error
            if not isinstance(answers, list) or len(answers) != len(unknown):
                raise RuntimeError("oracle result count mismatch")
            if any(type(item) is not bool for item in answers):
                raise RuntimeError("oracle result is not boolean")
            for candidate, accepted in zip(unknown, answers):
                self.cache[candidate] = accepted
        return [self.cache[value] for value in values]


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a JSON object")
    return value


def _integer(
    obj: dict[str, object], key: str, context: str, boundary: bool = False
) -> int:
    if key not in obj or type(obj[key]) is not int:
        raise RuntimeError(f"{context}.{key} must be a non-negative integer")
    value = obj[key]
    assert isinstance(value, int)
    if boundary and value == -1:
        return UNLIMITED
    if not 0 <= value <= UNLIMITED:
        raise RuntimeError(f"{context}.{key} must be a non-negative integer")
    return value


def parse_config(value: object) -> PipelineConfig:
    root = _object(value, "config")
    if set(root) - {"seed", "repair", "batching", "limits", "jev"}:
        raise RuntimeError("config contains unsupported fields")
    sections = {}
    for name in ("repair", "batching", "limits"):
        if name not in root:
            raise RuntimeError(f"config is missing field '{name}'")
        sections[name] = _object(root[name], f"config.{name}")
    repair, batching = sections["repair"], sections["batching"]
    limits = sections["limits"]
    jev = _object(root.get("jev", {}), "config.jev")
    if set(jev) - {"confidence_threshold", "timeout_seconds"}:
        raise RuntimeError("config.jev contains unsupported fields")
    threshold = jev.get("confidence_threshold", 0.5)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise RuntimeError("config.jev.confidence_threshold must be between 0 and 1")
    if "rsr_batch_size" in repair:
        raise RuntimeError("config.repair.rsr_batch_size is not supported")
    repair_fields = {
        "n",
        "ngrams_batch_size",
        "max_candidate_length",
        "num_candidates",
        "edit_distance_cap",
    }
    if set(repair) - repair_fields:
        raise RuntimeError("config.repair contains unsupported fields")
    capacity_fields = {
        "initial_positive_capacity",
        "initial_negative_capacity",
        "positive_capacity",
        "negative_capacity",
        "cell_timeout_seconds",
    }
    if set(batching) - capacity_fields:
        raise RuntimeError("config.batching contains unsupported fields")
    limit_fields = {
        "max_iterations",
        "max_states",
        "max_queue_size",
        "max_rsr_candidates",
    }
    if set(limits) - limit_fields:
        raise RuntimeError("config.limits contains unsupported fields")
    core = Config(
        seed=_integer(root, "seed", "config") if "seed" in root else 0,
        n=_integer(repair, "n", "config.repair"),
        ngrams_batch_size=_integer(repair, "ngrams_batch_size", "config.repair", True),
        max_candidate_length=_integer(
            repair, "max_candidate_length", "config.repair", True
        )
        if "max_candidate_length" in repair
        else UNLIMITED,
        max_iterations=_integer(limits, "max_iterations", "config.limits", True),
        max_states=_integer(limits, "max_states", "config.limits", True),
        max_queue_size=_integer(limits, "max_queue_size", "config.limits", True)
        if "max_queue_size" in limits
        else UNLIMITED,
        max_rsr_candidates=_integer(limits, "max_rsr_candidates", "config.limits", True)
        if "max_rsr_candidates" in limits
        else UNLIMITED,
        positive_capacity=_integer(
            batching, "positive_capacity", "config.batching", True
        )
        if "positive_capacity" in batching
        else UNLIMITED,
        negative_capacity=_integer(
            batching, "negative_capacity", "config.batching", True
        )
        if "negative_capacity" in batching
        else UNLIMITED,
    )
    result = PipelineConfig(
        core,
        _integer(batching, "initial_positive_capacity", "config.batching", True)
        if "initial_positive_capacity" in batching
        else UNLIMITED,
        _integer(batching, "initial_negative_capacity", "config.batching", True)
        if "initial_negative_capacity" in batching
        else UNLIMITED,
        _integer(batching, "cell_timeout_seconds", "config.batching"),
        _integer(repair, "num_candidates", "config.repair")
        if "num_candidates" in repair
        else 5,
        _integer(repair, "edit_distance_cap", "config.repair")
        if "edit_distance_cap" in repair
        else 3,
        float(threshold),
        _integer(jev, "timeout_seconds", "config.jev")
        if "timeout_seconds" in jev
        else 15,
    )
    if not all(
        (
            core.ngrams_batch_size,
            core.max_candidate_length,
            core.max_iterations,
            core.max_states,
            core.max_queue_size,
            core.max_rsr_candidates,
            core.negative_capacity,
            result.initial_positive_capacity,
            result.initial_negative_capacity,
            result.cell_timeout_seconds,
            result.num_candidates,
            result.jev_timeout_seconds,
        )
    ):
        raise RuntimeError(
            "configured counts and resource limits must be positive or -1 where supported"
        )
    if (
        core.positive_capacity < result.initial_positive_capacity
        or core.negative_capacity < result.initial_negative_capacity
    ):
        raise RuntimeError(
            "ongoing capacities must be at least their initial capacities"
        )
    return result


def read_config(path: Path) -> PipelineConfig:
    try:
        with path.open(encoding="utf-8") as stream:
            return parse_config(json.load(stream))
    except OSError as error:
        raise RuntimeError(f"cannot open config: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error


def load_dotenv(path: Path = ENV_PATH) -> None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    except OSError as error:
        raise RuntimeError(f"cannot open environment file: {path}") from error
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == "TYPESAFE_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                os.environ["TYPESAFE_API_KEY"] = value
            return


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.reader(stream))
    except OSError as error:
        raise RuntimeError(f"cannot open input CSV: {path}") from error
    if not records or not records[0]:
        raise RuntimeError("CSV has no header")
    columns = records[0]
    if len(set(columns)) != len(columns):
        raise RuntimeError("CSV headers must be unique")
    for index, row in enumerate(records[1:], 2):
        if len(row) != len(columns):
            raise RuntimeError(f"CSV row {index} has wrong field count")
        if any(not value.isascii() for value in row):
            raise RuntimeError(f"CSV row {index} contains a non-ASCII value")
    return columns, records[1:]


def read_schema(path: Path, columns: list[str]) -> dict[str, str]:
    try:
        with path.open(encoding="utf-8") as stream:
            root = json.load(stream)
    except OSError as error:
        raise RuntimeError(f"cannot open schema: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error
    root = _object(root, "schema")
    if set(root) != {"schema_version", "columns"} or root["schema_version"] != 1:
        raise RuntimeError("schema must contain schema_version 1 and columns")
    descriptions = _object(root["columns"], "schema.columns")
    if set(descriptions) != set(columns) or any(
        not isinstance(value, str) or not value.strip()
        for value in descriptions.values()
    ):
        raise RuntimeError(
            "schema columns must exactly match CSV columns with descriptions"
        )
    return {column: str(descriptions[column]) for column in columns}


def select_row_with_jev(
    columns: list[str],
    descriptions: dict[str, str],
    row: list[str],
    candidate_sets: dict[int, list[str]],
    timeout: float,
) -> dict[int, JevSelection]:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    questions = {}
    for column, candidates in candidate_sets.items():
        questions[f"column_{column}"] = {
            "type": "choice",
            "instructions": (
                f"Choose the most likely exact original value for column "
                f"'{columns[column]}' so all selected values form the most coherent row."
            ),
            "criteria": {
                f"candidate_{index}": {"value": candidate}
                for index, candidate in enumerate(candidates)
            },
        }
    payload = {
        "model": JEV_MODEL,
        "state": {
            "columns": [
                {"name": name, "description": descriptions[name]} for name in columns
            ],
            "fixed_values": {
                name: value
                for index, (name, value) in enumerate(zip(columns, row))
                if index not in candidate_sets and not is_missing(value)
            },
            "candidate_sets": {
                columns[column]: candidates
                for column, candidates in candidate_sets.items()
            },
        },
        "questions": questions,
    }
    request = Request(
        JEV_URL,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except Exception as error:
        raise RuntimeError("Jev request failed") from error
    if (
        not isinstance(result, dict)
        or result.get("model") != JEV_MODEL
        or not isinstance(result.get("answers"), dict)
    ):
        raise RuntimeError("invalid Jev response")
    answers = result["answers"]
    if set(answers) != set(questions):
        raise RuntimeError("invalid Jev response")
    selections = {}
    for column, candidates in candidate_sets.items():
        answer = answers[f"column_{column}"]
        if not isinstance(answer, dict):
            raise RuntimeError("invalid Jev response")
        choice, confidence = answer.get("choice"), answer.get("confidence")
        if (
            not isinstance(choice, str)
            or not choice.startswith("candidate_")
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise RuntimeError("invalid Jev response")
        try:
            candidate = int(choice.removeprefix("candidate_"))
        except ValueError as error:
            raise RuntimeError("invalid Jev response") from error
        if not 0 <= candidate < len(candidates):
            raise RuntimeError("invalid Jev response")
        selections[column] = JevSelection(candidate, float(confidence), JEV_MODEL)
    return selections


def oracle_path(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"missing oracle: {path}")
    return path.resolve()


def _query_oracle(
    oracle: Oracle, values: Sequence[str], deadline: float | None = None
) -> list[bool]:
    result = oracle.accepts_batch(values, deadline)
    if len(result) != len(values) or any(type(item) is not bool for item in result):
        raise RuntimeError("oracle query did not return one boolean per candidate")
    return result


def _singleton(oracle: Oracle, value: str, deadline: float | None = None) -> bool:
    return _query_oracle(oracle, [value], deadline)[0]


def _measurement_fields(measurements: AlgorithmMeasurements) -> dict[str, object]:
    return {
        "rsr_execution_time_ns": measurements.rsr_execution_time_ns,
        "oracle_total_calls": measurements.oracle_total_calls,
        "oracle_candidates_submitted": measurements.oracle_candidates_submitted,
        "oracle_execution_time_ns": measurements.oracle_execution_time_ns,
        "rpni2_execution_time_ns": measurements.rpni2_execution_time_ns,
        "acceptance_prefix_ns": measurements.acceptance_prefix_ns,
        "split_execution_time_ns": measurements.split_execution_time_ns,
        "ngrams_execution_time_ns": measurements.ngrams_execution_time_ns,
        "total_iterations": measurements.total_iterations,
        "rsr_iterations": len(measurements.rsr_iterations),
        "rsr_candidates_generated": sum(
            item.unique_candidates for item in measurements.rsr_iterations
        ),
        "rsr_enumeration_truncated": any(
            not item.enumeration_complete for item in measurements.rsr_iterations
        ),
    }


def candidate_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".tmp.csv")


def _read_candidate_sets(
    path: Path, columns: list[str], row_count: int
) -> dict[tuple[int, int], list[str]]:
    header, records = read_csv(path)
    expected = ["row", "column", "malformed_value", "candidate_rank", "candidate"]
    if header != expected:
        raise RuntimeError("candidate CSV has the wrong header")
    indexes = {name: index for index, name in enumerate(columns)}
    result: dict[tuple[int, int], list[str]] = {}
    for record in records:
        try:
            row = int(record[0])
            column = indexes[record[1]]
        except (ValueError, KeyError) as error:
            raise RuntimeError("candidate CSV has an invalid cell address") from error
        if not 0 <= row < row_count:
            raise RuntimeError("candidate CSV has an invalid row index")
        values = result.setdefault((row, column), [])
        if not record[3] and not record[4]:
            if values:
                raise RuntimeError(
                    "candidate CSV mixes candidates with an empty sentinel"
                )
            continue
        if record[3] != str(len(values) + 1) or record[4] in values:
            raise RuntimeError(
                "candidate CSV has invalid candidate ranks or duplicates"
            )
        values.append(record[4])
    return result


def _run_pipeline(
    input_path: Path,
    ground_truth_path: Path,
    schema_path: Path,
    config_path: Path,
    oracle_executable: Path,
    output_path: Path,
    telemetry_path: Path,
) -> None:
    started = time.monotonic()
    candidate_path = candidate_output_path(output_path)
    if candidate_path.resolve() == telemetry_path.resolve():
        raise RuntimeError("candidate output and telemetry paths must differ")
    try:
        with candidate_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerow(
                ["row", "column", "malformed_value", "candidate_rank", "candidate"]
            )
            stream.flush()
    except OSError as error:
        raise RuntimeError(f"cannot write candidate CSV: {candidate_path}") from error

    config = read_config(config_path)
    columns, source_rows = read_csv(input_path)
    truth_columns, truth_rows = read_csv(ground_truth_path)
    if truth_columns != columns or len(truth_rows) != len(source_rows):
        raise RuntimeError("ground-truth CSV shape differs from input CSV")
    descriptions = read_schema(schema_path, columns)
    input_correct_cells = sum(
        not is_missing(truth_rows[row][column]) and value == truth_rows[row][column]
        for row, values in enumerate(source_rows)
        for column, value in enumerate(values)
    )
    candidate_correct_cells = 0
    states = [[CellState() for _ in columns] for _ in source_rows]
    accepted_counts = [Counter() for _ in columns]
    column_metrics: dict[str, dict[str, object]] = {}
    total_cells = len(source_rows) * len(columns)
    completed = 0

    def progress() -> None:
        print(f"SHERMA_PROGRESS {completed} {total_cells}", file=sys.stderr, flush=True)

    progress()
    try:
        candidate_stream = candidate_path.open("a", newline="", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot write candidate CSV: {candidate_path}") from error
    executable = oracle_path(oracle_executable)
    with candidate_stream:
        writer = csv.writer(candidate_stream, lineterminator="\n")
        for column, name in enumerate(columns):
            column_started = time.monotonic()
            oracle = ExternalOracle(executable, name)
            accepted_values: set[str] = set()
            rejected_order: list[str] = []
            occurrences: dict[str, list[int]] = {}
            distinct = list(
                dict.fromkeys(
                    values[column]
                    for values in source_rows
                    if not is_missing(values[column])
                )
            )
            initial_answers = _query_oracle(
                oracle, distinct, time.monotonic() + config.cell_timeout_seconds
            )
            initially_accepted = dict(zip(distinct, initial_answers))
            for row, values in enumerate(source_rows):
                value = values[column]
                if is_missing(value):
                    states[row][column].sherma_outcome = "sherma_skipped_missing"
                    states[row][column].unresolved_reason = "missing_input"
                    completed += 1
                elif initially_accepted[value]:
                    states[row][column].sherma_outcome = "sherma_valid"
                    accepted_values.add(value)
                    accepted_counts[column][value] += 1
                    candidate_correct_cells += value == truth_rows[row][column]
                    completed += 1
                else:
                    if value not in occurrences:
                        rejected_order.append(value)
                        occurrences[value] = []
                    occurrences[value].append(row)
            progress()

            measurements = AlgorithmMeasurements()
            session = None
            if accepted_values:
                positives = sorted(accepted_values)
                initial_negatives = list(rejected_order)
                generator = random.Random(config.core.seed + column)
                generator.shuffle(positives)
                generator.shuffle(initial_negatives)
                if config.initial_positive_capacity != UNLIMITED:
                    positives = positives[: config.initial_positive_capacity]
                if config.initial_negative_capacity != UNLIMITED:
                    initial_negatives = initial_negatives[
                        : config.initial_negative_capacity
                    ]
                session = BatchSession(
                    positives, config.core, measurements, initial_negatives
                )

            repaired_count = 0
            timed_out_count = 0
            for dirty in rejected_order:
                repair_started = time.monotonic()
                if session is None:
                    candidates, failure = [], "no_accepted_values"
                else:
                    candidates, failure = session.repair_candidates(
                        dirty,
                        oracle,
                        repair_started + config.cell_timeout_seconds,
                        measurements,
                        config.num_candidates,
                        config.edit_distance_cap,
                    )
                    candidates = [
                        value for value in candidates if not is_missing(value)
                    ]
                elapsed = time.monotonic() - repair_started
                timed_out = failure == "cell timeout"
                for occurrence, row in enumerate(occurrences[dirty]):
                    state = states[row][column]
                    state.timed_out = timed_out
                    state.execution_time_seconds = elapsed if occurrence == 0 else 0.0
                    state.candidate_count = len(candidates)
                    if timed_out:
                        state.sherma_outcome = "sherma_timeout"
                    elif candidates:
                        state.sherma_outcome = "sherma_candidates"
                    else:
                        state.sherma_outcome = "sherma_no_candidates"
                    if not candidates:
                        state.unresolved_reason = failure or "repair_failed"
                        writer.writerow([row, name, dirty, "", ""])
                    else:
                        for rank, candidate in enumerate(candidates, 1):
                            writer.writerow([row, name, dirty, rank, candidate])
                        truth = truth_rows[row][column]
                        candidate_correct_cells += (
                            not is_missing(truth) and truth in candidates
                        )
                candidate_stream.flush()
                completed += len(occurrences[dirty])
                progress()
                repaired_count += bool(candidates)
                timed_out_count += timed_out
            column_metrics[name] = {
                "execution_time_seconds": time.monotonic() - column_started,
                "distinct_accepted": len(accepted_values),
                "distinct_rejected": len(rejected_order),
                "distinct_with_candidates": repaired_count,
                "distinct_timed_out": timed_out_count,
                "oracle_calls": oracle.calls,
                "oracle_candidates_submitted": oracle.candidates_submitted,
                "training_examples": 0
                if session is None
                else len(session.positives) + len(session.negatives),
                "max_positive_examples_stored": 0
                if session is None
                else len(session.positives),
                "max_negative_examples_stored": 0
                if session is None
                else len(session.negatives),
                "n": config.core.n,
                **_measurement_fields(measurements),
            }
    for column, name in enumerate(columns):
        column_metrics[name]["sherma_outcomes"] = dict(
            Counter(
                states[row][column].sherma_outcome for row in range(len(source_rows))
            )
        )

    repair_finished = time.monotonic()
    candidate_sets = _read_candidate_sets(candidate_path, columns, len(source_rows))
    column_pools = [
        [
            value
            for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ][: config.num_candidates]
        for counts in accepted_counts
    ]
    load_dotenv()
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    jev_started = time.monotonic()
    jev_calls = 0
    jev_low_confidence = 0
    final_correct_cells = 0
    unresolved_cells = 0
    output_tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    telemetry_tmp = telemetry_path.with_name(
        f".{telemetry_path.name}.{os.getpid()}.tmp"
    )
    try:
        with output_tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            for row, source in enumerate(source_rows):
                final = list(source)
                row_candidates: dict[int, list[str]] = {}
                for column in range(len(columns)):
                    state = states[row][column]
                    if state.sherma_outcome == "sherma_valid":
                        continue
                    values = candidate_sets.get((row, column), [])
                    if not values:
                        values = column_pools[column]
                    if values:
                        state.candidate_count = len(values)
                        row_candidates[column] = values
                    else:
                        final[column] = ""
                        state.unresolved_reason = "no_accepted_values"
                if row_candidates:
                    jev_calls += 1
                    selections = select_row_with_jev(
                        columns,
                        descriptions,
                        final,
                        row_candidates,
                        config.jev_timeout_seconds,
                    )
                    for column, candidates in row_candidates.items():
                        selection = selections[column]
                        state = states[row][column]
                        state.jev_confidence = selection.confidence
                        state.jev_model = selection.model
                        if selection.confidence < config.jev_confidence_threshold:
                            final[column] = candidates[0]
                            state.jev_fallback_reason = "low_confidence"
                            jev_low_confidence += 1
                        else:
                            final[column] = candidates[selection.candidate]
                        state.unresolved_reason = ""
                for column, value in enumerate(final):
                    truth = truth_rows[row][column]
                    final_correct_cells += not is_missing(truth) and value == truth
                    unresolved_cells += is_missing(value)
                writer.writerow(final)
        jev_finished = time.monotonic()
        telemetry = {
            "schema_version": 11,
            "cells": [
                {"row": row, "column": name, **asdict(states[row][column])}
                for row in range(len(source_rows))
                for column, name in enumerate(columns)
            ],
            "columns": column_metrics,
            "sherma_execution_time_seconds": repair_finished - started,
            "jev_execution_time_seconds": jev_finished - jev_started,
            "jev_calls": jev_calls,
            "jev_failures": 0,
            "jev_low_confidence": jev_low_confidence,
            "unresolved_cells": unresolved_cells,
            "input_correct_cells": input_correct_cells,
            "candidate_correct_cells": candidate_correct_cells,
            "final_correct_cells": final_correct_cells,
            "total_execution_time_seconds": jev_finished - started,
        }
        with telemetry_tmp.open("w", encoding="utf-8") as stream:
            json.dump(telemetry, stream, indent=2)
            stream.write("\n")
        os.replace(output_tmp, output_path)
        os.replace(telemetry_tmp, telemetry_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        telemetry_tmp.unlink(missing_ok=True)

    completed = total_cells
    progress()


def run_pipeline(
    input_path: Path,
    ground_truth_path: Path,
    schema_path: Path,
    config_path: Path,
    oracle_executable: Path,
    output_path: Path,
    telemetry_path: Path,
) -> None:
    log_path = output_path.parent / "error.log"
    try:
        output_path.unlink(missing_ok=True)
        telemetry_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        _run_pipeline(
            input_path,
            ground_truth_path,
            schema_path,
            config_path,
            oracle_executable,
            output_path,
            telemetry_path,
        )
    except Exception as error:
        for path in (output_path, telemetry_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            with log_path.open("w", encoding="utf-8") as stream:
                stream.write(datetime.now(timezone.utc).isoformat())
                stream.write(f"\n{type(error).__name__}: {error}\n\n")
                stream.write(traceback.format_exc())
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair a CSV with Sherma")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output.csv"))
    parser.add_argument("--telemetry", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_pipeline(
            args.input,
            args.ground_truth,
            args.schema,
            args.config,
            args.oracle,
            args.output,
            args.telemetry,
        )
        return 0
    except Exception as error:
        print(f"sherma: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
