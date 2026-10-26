from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ALGORITHM = HERE / "algorithm"
sys.path.insert(0, str(ALGORITHM))

import sherma as algorithm
from oracle.build import build_oracle


SHARED = HERE / "test-cases"
SCHEMAS = SHARED / "schemas"
CONFIG = HERE / "test-config.json"
RESULTS = HERE / "test-results"
MISSING = {"", "?"}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows[0], rows[1:]


def predicate(spec: dict[str, object]) -> Callable[[str], bool]:
    if spec.get("type") == "enum":
        allowed = set(spec["values"])
        return lambda value: value not in MISSING and value in allowed
    if spec.get("type") == "regex":
        expression = re.compile(str(spec["regex"]))
        return lambda value: value not in MISSING and expression.fullmatch(value) is not None
    raise RuntimeError(f"unsupported oracle type: {spec.get('type')!r}")


def manifest_path(case: Path) -> Path:
    return HERE / "test-case-generator-tool" / "oracles" / f"{case.name}.json"


def oracle_specs(case: Path) -> dict[str, dict[str, object]]:
    manifest = read_json(manifest_path(case))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("oracles"), dict):
        raise RuntimeError(f"invalid oracle manifest: {manifest_path(case)}")
    return manifest["oracles"]


def evaluate(case: Path, output_path: Path, telemetry_path: Path,
             specs: dict[str, dict[str, object]]) -> dict[str, object]:
    columns, original = read_csv(case / "original.csv")
    output_columns, output = read_csv(output_path)
    if output_columns != columns or len(output) != len(original):
        raise RuntimeError("output CSV shape differs from original.csv")
    if set(specs) != set(columns):
        raise RuntimeError("oracle columns differ from CSV columns")

    telemetry = read_json(telemetry_path)
    if not isinstance(telemetry, dict) or telemetry.get("schema_version") != 11:
        raise RuntimeError("expected schema-version 11 telemetry")
    records = telemetry.get("cells")
    if not isinstance(records, list):
        raise RuntimeError("telemetry.cells must be an array")
    indexed = {(item.get("row"), item.get("column")): item for item in records
               if isinstance(item, dict)}
    expected = {(row, column) for row in range(len(original)) for column in columns}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise RuntimeError("telemetry cells are duplicate, unexpected, or incomplete")
    recorded_unresolved = sum(bool(item.get("unresolved_reason"))
                              for item in indexed.values())
    if telemetry.get("unresolved_cells") != recorded_unresolved:
        raise RuntimeError("telemetry summary differs from its cell records")

    valid = exact = eligible = missing = invalid = 0
    accepts = [predicate(specs[column]) for column in columns]
    for clean_row, output_row in zip(original, output):
        for column, (clean, value) in enumerate(zip(clean_row, output_row)):
            if clean not in MISSING:
                eligible += 1
                exact += value == clean
            valid += value not in MISSING and accepts[column](value)
            missing += value in MISSING
            invalid += value not in MISSING and not accepts[column](value)
    if invalid:
        raise RuntimeError(f"output CSV contains {invalid} invalid values")

    total = len(original) * len(columns)
    report = {
        "schema_version": 1,
        "dataset": case.name,
        "cells": total,
        "oracle_correct": valid,
        "oracle_accuracy": valid / total if total else None,
        "exact_correct": exact,
        "exact_accuracy": exact / eligible if eligible else None,
        "input_correct_cells": telemetry.get("input_correct_cells"),
        "final_correct_cells": telemetry.get("final_correct_cells"),
        "final_missing": missing,
        "unresolved": telemetry.get("unresolved_cells"),
        "total_execution_time_seconds": telemetry.get("total_execution_time_seconds"),
    }
    if report["unresolved"] != missing:
        raise RuntimeError("telemetry unresolved count differs from candidate output")
    if report["final_correct_cells"] != exact:
        raise RuntimeError("telemetry exact-correct count differs from candidate output")
    return report


def run_case(case: Path, output: Path, oracle: Path) -> dict[str, object]:
    output.mkdir()
    specs = oracle_specs(case)
    repaired = output / "repaired.csv"
    telemetry = output / "telemetry.json"
    with (output / "progress.log").open("w", encoding="utf-8") as progress, \
            redirect_stderr(progress):
        algorithm.run_pipeline(
            case / "corrupted.csv", case / "original.csv",
            SCHEMAS / f"{case.name}.json", CONFIG, oracle, repaired, telemetry)
    report = evaluate(case, repaired, telemetry, specs)
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{case.parent.name}/{case.name}: "
          f"oracle={report['oracle_accuracy']:.3f}, "
          f"exact={report['exact_accuracy']:.3f}, "
          f"missing={report['final_missing']}", flush=True)
    return report


def run_unit_tests() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(HERE),
         "-p", "test_*.py"], cwd=HERE, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"unit tests failed:\n{completed.stdout}")
    print(completed.stdout.strip(), flush=True)


class RunProgress:
    """Print case progress and a heartbeat during long runs."""

    def __init__(self, total_cases: int) -> None:
        self.total_cases = total_cases
        self.completed_cases = 0
        self.case_name = "preparing"
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _write(self) -> None:
        elapsed = int(time.monotonic() - self.started)
        print(f"[{elapsed}s] {self.case_name}: test cases "
              f"{self.completed_cases}/{self.total_cases}", flush=True)

    def _run(self) -> None:
        while not self.stop.wait(30):
            with self.lock:
                self._write()

    def start(self) -> None:
        self.thread.start()

    def begin_case(self, name: str) -> None:
        with self.lock:
            self.case_name = name
            self._write()

    def complete_case(self) -> None:
        with self.lock:
            self.completed_cases += 1
            self._write()

    def close(self) -> None:
        self.stop.set()
        self.thread.join()


def cases(mode: str) -> list[Path]:
    if mode == "smoke":
        return [SHARED / "generated-smoke" / "adult_20"]
    return [root / name for root in (SHARED / "generated-50", SHARED / "generated-90")
            for name in ("adult_20", "bank_marketing_222",
                         "census_income_kdd_117", "support2_880")]


def available_workers(job_count: int) -> int:
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        count = process_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return min(job_count, count or 1)


def build_dataset_oracles(selected: list[Path], root: Path) -> dict[str, Path]:
    compiled: dict[str, Path] = {}
    suffix = ".exe" if os.name == "nt" else ""
    for case in selected:
        if case.name in compiled:
            continue
        output = root / "oracles" / case.name / f"oracle{suffix}"
        print(f"Compiling oracle for {case.name}...", flush=True)
        compiled[case.name] = build_oracle(manifest_path(case), output)
    return compiled


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sherma against bundled VLDB cases")
    parser.add_argument("mode", choices=("smoke", "test"))
    args = parser.parse_args()
    if not SHARED.is_dir():
        raise RuntimeError(f"bundled test data not found: {SHARED}")
    run_unit_tests()
    RESULTS.mkdir(exist_ok=True)
    root = RESULTS / args.mode
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    print(f"Artifacts: {root}", flush=True)
    selected = cases(args.mode)
    compiled = build_dataset_oracles(selected, root)
    jobs = [(case, root / f"{case.parent.name}-{case.name}", compiled[case.name])
            for case in selected]
    workers = available_workers(len(jobs))
    timer = RunProgress(len(jobs))
    executor = ProcessPoolExecutor(max_workers=workers)
    timer_started = False
    try:
        futures = {executor.submit(run_case, case, output, oracle): case
                   for case, output, oracle in jobs}
        timer.start()
        timer_started = True
        timer.begin_case(f"running with {workers} workers")
        for future in as_completed(futures):
            future.result()
            timer.complete_case()
    finally:
        executor.shutdown(cancel_futures=True)
        if timer_started:
            timer.close()
    print(f"{args.mode.capitalize()} suite passed ({len(selected)} bundled cases).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
