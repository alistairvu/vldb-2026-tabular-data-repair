#!/usr/bin/env python3
"""Generate shared VLDB cases, validate manual copies, and calculate benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2] / "test-suites"
SUITES = tuple(
    path
    for path in (
        ROOT / name
        for name in (
            "vldb-gammamax",
            "vldb-gammamax-dt",
            "vldb-gammamax-knn",
            "vldb-gidcl",
            "vldb-lopster",
        )
    )
    if path.is_dir()
)
DATASETS = ("openpowerlifting",)
IGNORED_VALUES = {"", "?"}
INSERT_ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError(f"invalid CSV header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_ignored(value: str) -> bool:
    return value in IGNORED_VALUES


def predicate(spec: dict[str, Any]) -> Callable[[str], bool]:
    if spec.get("type") == "enum":
        allowed = set(spec["values"])
        return lambda value: not is_ignored(value) and value in allowed
    if spec.get("type") == "regex":
        expression = re.compile(spec["regex"])
        return lambda value: (
            not is_ignored(value) and expression.fullmatch(value) is not None
        )
    raise ValueError(f"unsupported oracle type: {spec.get('type')!r}")


def refresh_oracles(manifest: dict[str, Any]) -> dict[str, Any]:
    """Refresh derived shapes while preserving manually curated oracle rules."""
    for spec in manifest["oracles"].values():
        predicate(spec)  # Validate types and regular expressions before writing.
        if spec["type"] == "regex":
            expression = spec["regex"]
            body = expression[1:] if expression.startswith("^") else expression
            if body.endswith("$") and not body.endswith(r"\$"):
                body = body[:-1]
            spec["shapes"] = [body]
        else:
            spec.pop("shapes", None)
    return manifest


def apply_edit(value: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    operations = ["insert"] + (["delete", "substitute"] if value else [])
    operation = rng.choice(operations)
    if operation == "insert":
        position = rng.randrange(len(value) + 1)
        character = rng.choice(INSERT_ALPHABET)
        return value[:position] + character + value[position:], {
            "operation": operation,
            "position": position,
            "character": character,
        }
    position = rng.randrange(len(value))
    if operation == "delete":
        return value[:position] + value[position + 1 :], {
            "operation": operation,
            "position": position,
            "character": value[position],
        }
    choices = INSERT_ALPHABET.replace(value[position], "")
    character = rng.choice(choices)
    return value[:position] + character + value[position + 1 :], {
        "operation": operation,
        "position": position,
        "from": value[position],
        "to": character,
    }


def corrupt_value(
    source: str,
    accepts: Callable[[str], bool],
    rng: random.Random,
    minimum: int,
    maximum: int,
) -> tuple[str, list[dict[str, Any]]]:
    for _ in range(1000):
        value, edits = source, []
        for _ in range(rng.randint(minimum, maximum)):
            value, edit = apply_edit(value, rng)
            edits.append(edit)
        if value != source and not accepts(value):
            return value, edits
    raise RuntimeError(f"could not corrupt value {source!r}")


def select_corruptions(original, specs, eligible, count, rng):
    """Redraw unsafe cell selections while retaining every observed enum value."""
    remaining = {
        column: Counter(row[column] for row in original if not is_ignored(row[column]))
        for column, spec in specs.items()
        if spec["type"] == "enum"
    }
    required = sum(len(values) for values in remaining.values())
    if count > len(eligible) - required:
        raise ValueError(
            "corruption count leaves too few cells to preserve every enum value"
        )
    candidates = list(eligible)
    rng.shuffle(candidates)
    selected = []
    redraws = 0
    for cell in candidates:
        if len(selected) == count:
            break
        row, column = cell
        if column in remaining:
            value = original[row][column]
            if remaining[column][value] == 1:
                redraws += 1
                continue
            remaining[column][value] -= 1
        selected.append(cell)
    assert len(selected) == count
    return selected, redraws


def generate_cases(rate: int) -> None:
    config = read_json(HERE / "config.json")
    generation = dict(config["generation"])
    configured_rates = generation.pop("corruption_rates")
    selected_rate = rate / 100
    if selected_rate not in configured_rates:
        raise ValueError(f"corruption rate {selected_rate} is not configured")
    generate_case_set(
        config,
        generation,
        selected_rate,
        config["output_roots"][str(rate)],
        DATASETS,
    )


def generate_smoke_case() -> None:
    config = read_json(HERE / "config.json")
    smoke = config["smoke"]
    generation = dict(config["generation"])
    generation.pop("corruption_rates")
    generation["max_rows"] = smoke["rows"]
    generate_case_set(
        config,
        generation,
        smoke["corruption_rate"],
        smoke["output_root"],
        (smoke["dataset"],),
    )


def generate_case_set(
    config: dict[str, Any],
    generation: dict[str, Any],
    corruption_rate: float,
    output_name: str,
    dataset_names: tuple[str, ...],
) -> None:
    if type(corruption_rate) not in (int, float) or not 0 <= corruption_rate <= 1:
        raise ValueError("corruption rate must be between 0 and 1")
    missing_rate = generation.get("missing_rate", 0.1)
    if type(missing_rate) not in (int, float) or not 0 <= missing_rate <= 1:
        raise ValueError("generation.missing_rate must be between 0 and 1")
    generation["corruption_rate"] = corruption_rate
    output_root = HERE / output_name
    dataset_root = (HERE / config["datasets_root"]).resolve()
    for dataset_name in dataset_names:
        columns, all_rows = read_csv(dataset_root / dataset_name / "clean.csv")
        rng = random.Random(generation["seed"])
        count = min(generation["max_rows"], len(all_rows))
        source_indices = rng.sample(range(len(all_rows)), count)
        original = [dict(all_rows[index]) for index in source_indices]
        oracle_path = HERE / "oracles" / f"{dataset_name}.json"
        manifest = refresh_oracles(read_json(oracle_path))
        specs = manifest["oracles"]
        if list(specs) != columns:
            raise ValueError(f"oracle columns do not match dataset: {dataset_name}")
        eligible = [
            (row, column)
            for row in range(count)
            for column in columns
            if not is_ignored(original[row][column])
        ]
        total_corruption_count = round(len(eligible) * generation["corruption_rate"])
        missing_count = round(total_corruption_count * missing_rate)
        affected, redraws = select_corruptions(
            original, specs, eligible, total_corruption_count, rng
        )
        rng.shuffle(affected)
        missing_cells = affected[:missing_count]
        selected = affected[missing_count:]
        corrupted = [dict(row) for row in original]
        corruptions = []
        for row, column in missing_cells:
            source = original[row][column]
            corrupted[row][column] = ""
            corruptions.append(
                {
                    "row": row,
                    "column": column,
                    "clean_value": source,
                    "corrupted_value": "",
                    "edits": [{"operation": "set_missing"}],
                    "corruption_type": "missing",
                }
            )
        for row, column in selected:
            source = original[row][column]
            dirty, edits = corrupt_value(
                source,
                predicate(specs[column]),
                rng,
                generation["min_edits_per_cell"],
                generation["max_edits_per_cell"],
            )
            corrupted[row][column] = dirty
            corruptions.append(
                {
                    "row": row,
                    "column": column,
                    "clean_value": source,
                    "corrupted_value": dirty,
                    "edits": edits,
                    "corruption_type": "character_edit",
                }
            )
        corruptions.sort(key=lambda item: (item["row"], columns.index(item["column"])))
        destination = output_root / dataset_name
        write_csv(destination / "original.csv", columns, original)
        write_csv(destination / "corrupted.csv", columns, corrupted)
        write_json(oracle_path, manifest)
        write_json(
            destination / "case.json",
            {
                "schema_version": 1,
                "dataset": dataset_name,
                "seed": generation["seed"],
                "source_row_indices": source_indices,
                "row_count": count,
                "column_count": len(columns),
                "eligible_cell_count": len(eligible),
                "corrupted_cell_count": len(corruptions),
                "character_corrupted_cell_count": len(selected),
                "missing_cell_count": len(missing_cells),
                "category_preservation_redraws": redraws,
                "generation": generation,
                "run_metadata": config.get("run_metadata", {}),
                "corruptions": corruptions,
            },
        )
        print(
            f"{output_name}/{dataset_name}: {len(selected)} character corruptions, "
            f"{len(missing_cells)} missing cells"
        )


def oracle_manifest(case_dir: Path) -> Path:
    case = read_json(case_dir / "case.json")
    return HERE / "oracles" / f"{case['dataset']}.json"


def reference_dir(case_dir: Path) -> Path:
    if case_dir.parent.name not in {"generated-50", "generated-90"}:
        return case_dir
    shared = case_dir.parents[1] / "shared-generation" / case_dir.name
    if (shared / "original.csv").is_file():
        return shared
    return case_dir


EXAMPLE_PEAKS = ("max_positive_examples_stored", "max_negative_examples_stored")


def example_peaks(columns: dict[str, Any]) -> dict[str, Any]:
    """Maxima across columns; absent legacy measurements remain unknown."""
    return {
        key: max(values) if values and all(v is not None for v in values) else None
        for key in EXAMPLE_PEAKS
        for values in [[column.get(key) for column in columns.values()]]
    }


def write_example_summary(suite: Path) -> None:
    datasets = {}
    for path in sorted(suite.glob("generated-*/**/benchmark.json")):
        result = read_json(path)
        datasets[str(path.relative_to(suite))] = {
            key: result["table"].get(key) for key in EXAMPLE_PEAKS
        }
    summary = {"schema_version": 1, **example_peaks(datasets), "datasets": datasets}
    (suite / "example-storage-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(
    case_dir: Path,
    repaired_path: Path,
    telemetry_path: Path | None,
    algorithm_config: dict[str, Any],
) -> dict[str, Any]:
    references = reference_dir(case_dir)
    columns, original = read_csv(references / "original.csv")
    repaired_columns, repaired = read_csv(repaired_path)
    if repaired_columns != columns:
        raise ValueError(
            "repaired CSV headers or column order do not match original.csv"
        )
    if len(repaired) != len(original):
        raise ValueError("repaired CSV row count does not match original.csv")
    specs = read_json(oracle_manifest(case_dir))["oracles"]
    telemetry = (
        read_json(telemetry_path)
        if telemetry_path and telemetry_path.is_file()
        else None
    )
    timed = {}
    if telemetry:
        records = telemetry.get("cells")
        if not isinstance(records, list):
            raise ValueError("telemetry.cells must be an array")
        expected = {
            (row, column)
            for row in range(len(original))
            for column in columns
            if telemetry.get("schema_version") == 2
            or not is_ignored(original[row][column])
        }
        for item in records:
            key = (item.get("row"), item.get("column"))
            if key not in expected or key in timed:
                raise ValueError(
                    f"invalid, duplicate, or unexpected telemetry cell: {key}"
                )
            timed[key] = item
        if set(timed) != expected:
            raise ValueError(
                f"telemetry is incomplete: expected {len(expected)}, received {len(timed)} cells"
            )
    totals = {
        key: 0
        for key in (
            "cells",
            "eligible_cells",
            "oracle_eligible_cells",
            "exact_eligible_cells",
            "originally_missing_cells",
            "oracle_correct",
            "exact_correct",
            "timeouts",
            "knn_imputed",
            "final_missing",
            "final_invalid",
        )
    }
    totals["execution_time_seconds"] = 0.0
    per_column = {}
    for column in columns:
        metric = {
            key: (0.0 if key == "execution_time_seconds" else 0) for key in totals
        }
        accepts = predicate(specs[column])
        for row, clean in enumerate(original):
            output = repaired[row][column]
            missing = is_ignored(clean[column])
            accepted = accepts(output)
            record = timed.get((row, column), {})
            metric["cells"] += 1
            metric["oracle_eligible_cells"] += 1
            metric["originally_missing_cells"] += int(missing)
            metric["oracle_correct"] += int(accepted)
            metric["final_missing"] += int(is_ignored(output))
            metric["final_invalid"] += int(not is_ignored(output) and not accepted)
            metric["timeouts"] += int(record.get("timed_out", False))
            metric["knn_imputed"] += int(record.get("knn_imputed", False))
            if not missing:
                metric["eligible_cells"] += 1
                metric["exact_eligible_cells"] += 1
                metric["exact_correct"] += int(output == clean[column])
        metric["oracle_accuracy"] = (
            metric["oracle_correct"] / metric["oracle_eligible_cells"]
            if metric["oracle_eligible_cells"]
            else None
        )
        metric["exact_accuracy"] = (
            metric["exact_correct"] / metric["exact_eligible_cells"]
            if metric["exact_eligible_cells"]
            else None
        )
        metric["timeout_rate"] = (
            metric["timeouts"] / metric["exact_eligible_cells"]
            if telemetry and metric["exact_eligible_cells"]
            else None
        )
        if telemetry:
            column_data = telemetry.get("columns", {}).get(column, {})
            metric["execution_time_seconds"] = float(
                column_data.get("execution_time_seconds", 0.0)
            )
            metric["algorithm_counts"] = {
                key: value
                for key, value in column_data.items()
                if key != "execution_time_seconds"
            }
        per_column[column] = metric
        for key in totals:
            totals[key] += metric[key]
    totals["oracle_accuracy"] = (
        totals["oracle_correct"] / totals["oracle_eligible_cells"]
        if totals["oracle_eligible_cells"]
        else None
    )
    totals["exact_accuracy"] = (
        totals["exact_correct"] / totals["exact_eligible_cells"]
        if totals["exact_eligible_cells"]
        else None
    )
    totals["timeout_rate"] = (
        totals["timeouts"] / totals["exact_eligible_cells"]
        if telemetry and totals["exact_eligible_cells"]
        else None
    )
    if telemetry:
        totals["total_execution_time_seconds"] = float(
            telemetry["total_execution_time_seconds"]
        )
    for column, metric in per_column.items():
        column_data = telemetry.get("columns", {}).get(column, {}) if telemetry else {}
        for key in EXAMPLE_PEAKS:
            metric[key] = column_data.get(key)
    totals.update(example_peaks(per_column))
    case = read_json(case_dir / "case.json")
    return {
        "schema_version": 2,
        "dataset": case["dataset"],
        "generation": case["generation"],
        "algorithm": algorithm_config,
        "run_metadata": case.get("run_metadata", {}),
        "columns": per_column,
        "table": totals,
        "telemetry_supplied": telemetry is not None,
        "validation_failures": [],
    }


def locate_run(suite: Path, case: Path) -> tuple[Path, Path, Path] | None:
    candidates = (case / "run", case)
    for run in candidates:
        repaired = run / "repaired.csv"
        if repaired.is_file():
            return run, repaired, run / "telemetry.json"
    return None


def regenerate_benchmarks() -> None:
    config = read_json(HERE / "config.json")
    count = 0
    for suite in SUITES:
        evaluator = evaluate
        if suite.name.startswith("vldb-gammamax"):
            spec = importlib.util.spec_from_file_location(
                "suite_evaluator", suite / "evaluator.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            evaluator = module.evaluate
        for configured_rate in config["generation"]["corruption_rates"]:
            rate = round(configured_rate * 100)
            output_name = config["output_roots"][str(rate)]
            for dataset in DATASETS:
                case = suite / output_name / dataset
                located = locate_run(suite, case)
                if located is None:
                    print(
                        f"skip {suite.name}/generated-{rate}/{dataset}: no repaired.csv"
                    )
                    continue
                run, repaired, telemetry = located
                config_path = run / "algorithm-config.json"
                if not config_path.is_file():
                    config_path = case.parent / "algorithm-config.json"
                algorithm = read_json(config_path) if config_path.is_file() else {}
                write_json(
                    run / "benchmark.json",
                    evaluator(case, repaired, telemetry, algorithm),
                )
                count += 1
    for suite in SUITES:
        if suite.name.startswith("vldb-gammamax"):
            write_example_summary(suite)
    print(f"wrote {count} benchmark.json files")


def case_artifact(root: Path, output_name: str, dataset: str, filename: str) -> Path:
    if output_name in {"generated-50", "generated-90"} and filename in {
        "original.csv",
    }:
        shared = root / "shared-generation" / dataset / filename
        if shared.is_file():
            return shared
    return root / output_name / dataset / filename


def artifact_digest(path: Path, filename: str) -> bytes:
    if filename == "case.json":
        case = read_json(path)
        payload = json.dumps(case, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).digest()
    return hashlib.sha256(path.read_bytes()).digest()


def validate_case_set(
    output_name: str, dataset_names: tuple[str, ...], corruption_rate: float
) -> int:
    checked = 0
    for dataset in dataset_names:
        canonical = HERE / output_name / dataset
        case = read_json(canonical / "case.json")
        missing = [
            item
            for item in case["corruptions"]
            if item.get("corruption_type") == "missing"
        ]
        character = [
            item
            for item in case["corruptions"]
            if item.get("corruption_type") == "character_edit"
        ]
        expected_total = round(case["eligible_cell_count"] * corruption_rate)
        expected_missing = round(expected_total * case["generation"]["missing_rate"])
        if (
            len(missing) != expected_missing
            or len(missing) + len(character) != expected_total
        ):
            raise ValueError(
                f"corruption allocation differs from configured rates: {output_name}/{dataset}"
            )
        if len(missing) != case["missing_cell_count"] or any(
            item["corrupted_value"] != "" for item in missing
        ):
            raise ValueError(f"invalid missing-cell metadata: {output_name}/{dataset}")
        if len(character) != case["character_corrupted_cell_count"]:
            raise ValueError(
                f"invalid character-corruption metadata: {output_name}/{dataset}"
            )
        coordinates = [(item["row"], item["column"]) for item in case["corruptions"]]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError(f"overlapping corruptions: {output_name}/{dataset}")
        for filename in ("original.csv", "corrupted.csv", "case.json"):
            expected = artifact_digest(
                case_artifact(HERE, output_name, dataset, filename),
                filename,
            )
            for suite in SUITES:
                actual = artifact_digest(
                    case_artifact(suite, output_name, dataset, filename),
                    filename,
                )
                if actual != expected:
                    raise ValueError(
                        f"suite case differs: {suite.name}/{output_name}/{dataset}/{filename}"
                    )
            checked += 1
    return checked


def validate_cases() -> None:
    config = read_json(HERE / "config.json")
    checked = 0
    for configured_rate in config["generation"]["corruption_rates"]:
        rate = round(configured_rate * 100)
        output_name = config["output_roots"][str(rate)]
        checked += validate_case_set(output_name, DATASETS, configured_rate)
    smoke = config["smoke"]
    checked += validate_case_set(
        smoke["output_root"],
        (smoke["dataset"],),
        smoke["corruption_rate"],
    )
    print(f"validated {checked} canonical artifacts across {len(SUITES)} suites")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--rates", nargs="+", type=int)
    sub.add_parser("benchmarks")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "generate":
        configured = read_json(HERE / "config.json")["generation"]["corruption_rates"]
        rates = args.rates or tuple(round(rate * 100) for rate in configured)
        for rate in rates:
            generate_cases(rate)
        generate_smoke_case()
    elif args.command == "benchmarks":
        regenerate_benchmarks()
    if args.command in ("validate", "all"):
        validate_cases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
