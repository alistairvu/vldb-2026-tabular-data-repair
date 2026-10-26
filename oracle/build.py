from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TOP_LEVEL_FIELDS = {"dataset", "source", "oracles"}
COMMON_RULE_FIELDS = {"type"}


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a JSON object")
    return value


def _exact_fields(value: dict[str, object], expected: set[str], context: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise RuntimeError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise RuntimeError(f"{context} has unsupported fields: {', '.join(sorted(unknown))}")


def validate_manifest(value: object) -> dict[str, object]:
    manifest = _object(value, "manifest")
    _exact_fields(manifest, TOP_LEVEL_FIELDS, "manifest")
    if not isinstance(manifest["dataset"], str) or not manifest["dataset"]:
        raise RuntimeError("manifest.dataset must be a non-empty string")
    if not isinstance(manifest["source"], str) or not manifest["source"]:
        raise RuntimeError("manifest.source must be a non-empty string")
    rules = manifest["oracles"]
    if not isinstance(rules, dict) or not rules:
        raise RuntimeError("manifest.oracles must be a non-empty object keyed by column")

    for column, raw_rule in rules.items():
        context = f"manifest.oracles[{column!r}]"
        if not isinstance(column, str) or not column:
            raise RuntimeError("oracle column names must be non-empty strings")
        rule = _object(raw_rule, context)
        rule_type = rule.get("type")
        extra = {"values"} if rule_type == "enum" else {"regex", "shapes"}
        if rule_type not in {"enum", "regex"}:
            raise RuntimeError(f"{context}.type must be 'enum' or 'regex'")
        _exact_fields(rule, COMMON_RULE_FIELDS | extra, context)
        if rule_type == "enum":
            values = rule["values"]
            if (not isinstance(values, list) or not values or
                    any(not isinstance(item, str) for item in values)):
                raise RuntimeError(f"{context}.values must be a non-empty string array")
            if len(values) != len(set(values)):
                raise RuntimeError(f"{context}.values contains duplicates")
        else:
            expression, shapes = rule["regex"], rule["shapes"]
            if not isinstance(expression, str) or not expression:
                raise RuntimeError(f"{context}.regex must be a non-empty string")
            if (not isinstance(shapes, list) or not shapes or
                    any(not isinstance(item, str) for item in shapes)):
                raise RuntimeError(f"{context}.shapes must be a non-empty string array")
    return manifest


def _run(command: list[str]) -> None:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
    except OSError as error:
        raise RuntimeError(f"cannot run {command[0]}: {error}") from error
    if completed.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stdout}")


def build_oracle(manifest_path: Path, output: Path) -> Path:
    if os.name == "nt" and output.suffix.lower() != ".exe":
        raise RuntimeError("oracle output must end in .exe on Windows")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read oracle manifest {manifest_path}: {error}") from error
    manifest = validate_manifest(raw)
    if manifest_path.suffix.lower() != ".json" or manifest_path.stem != manifest["dataset"]:
        raise RuntimeError("oracle filename must be <dataset>.json and match manifest.dataset")
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    values = ",".join(str(byte) for byte in encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending_name = (f".{output.stem}.{os.getpid()}.tmp.exe" if os.name == "nt" else
                    f".{output.name}.{os.getpid()}.tmp")
    pending = output.with_name(pending_name)
    pending.unlink(missing_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="sherma-oracle-") as directory:
            build = Path(directory)
            (build / "manifest.hpp").write_text(
                "#pragma once\n"
                f"inline constexpr unsigned char embedded_manifest[] = {{{values}}};\n"
                f"inline constexpr std::size_t embedded_manifest_size = {len(encoded)};\n",
                encoding="utf-8",
            )
            _run(["cmake", "-S", str(HERE), "-B", str(build),
                  "-DCMAKE_BUILD_TYPE=Release"])
            _run(["cmake", "--build", str(build), "--config", "Release"])
            suffix = ".exe" if os.name == "nt" else ""
            binaries = list(build.rglob(f"dataset_oracle{suffix}"))
            if len(binaries) != 1:
                raise RuntimeError("CMake did not produce exactly one dataset oracle")
            shutil.copy2(binaries[0], pending)
        if os.name != "nt":
            pending.chmod(pending.stat().st_mode | 0o111)

        for column in manifest["oracles"]:
            request = json.dumps({"column": column, "candidates": []})
            completed = subprocess.run([str(pending.resolve())], input=request, text=True,
                                       capture_output=True, check=False)
            if completed.returncode or completed.stdout != "[]":
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"compiled oracle rejected column {column!r}: {detail}")
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile a Sherma oracle manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        build_oracle(args.manifest, args.output)
        return 0
    except RuntimeError as error:
        print(f"oracle build: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
