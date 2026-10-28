import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from multiprocessing import Manager, Process


def process_input_flags(dataset, testcase) -> dict[str, str]:
    res = {}

    res["input"] = f"./test-case-generator-tool/{testcase}/{dataset}/corrupted.csv"
    res["ground-truth"] = (
        f"./test-case-generator-tool/{testcase}/{dataset}/original.csv"
    )
    res["schema"] = f"./test-case-generator-tool/schemas/{dataset}.json"
    res["config"] = "./test-config.json"
    res["oracle"] = f"./build/oracles/{dataset}"
    res["output"] = "output.csv"
    res["telemetry"] = "telemetry.json"

    return res


def report_accuracy(input_flags):
    corrupted_data = []
    cleaned_data = []
    original_data = []

    with open(input_flags["input"], "r") as f:
        reader = csv.reader(f)
        for row in reader:
            corrupted_data.append(list(row))

    with open(input_flags["output"], "r") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned_data.append(list(row))

    with open(input_flags["ground-truth"], "r") as f:
        reader = csv.reader(f)
        for row in reader:
            original_data.append(list(row))

    zipped = zip(corrupted_data, cleaned_data, original_data)

    total = 0
    correct = 0

    for corrupted, cleaned, original in zipped:
        for corr, clean, orig in zip(corrupted, cleaned, original):
            if corr != orig:
                total += 1
                if clean == orig:
                    correct += 1

    print(f"Corrupted: {total}, Correct: {correct}")
    print(f"Accuracy: {round(correct / total * 100, 2)}%")
    return correct, total


def _exec(cmd, result):
    p = subprocess.Popen(cmd)
    p.wait()
    rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["returncode"] = p.returncode
    result["max_rss"] = rusage.ru_maxrss
    result["utime"] = rusage.ru_utime
    result["stime"] = rusage.ru_stime


def exec_algorithm(input_flags, algorithm) -> tuple[dict, float | None]:

    start = time.time()

    print(f"Executing algorithm: {algorithm} on dataset: {input_flags['input']}")

    with Manager() as manager:
        result = manager.dict()
        cmd = [
            "python",
            f"algorithm/{algorithm}.py",
            *[f"--{k}={v}" for k, v in input_flags.items()],
        ]
        process = Process(
            target=_exec,
            args=(
                cmd,
                result,
            ),
        )
        process.start()
        process.join()
        result = dict(result)

    end = time.time()
    elapsed = end - start
    print(f"Elapsed time: {elapsed} seconds")

    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harness for tabular data repair algorithms"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--testcase", required=True)
    parser.add_argument("--algorithm", required=True)
    args = parser.parse_args()

    input_flags = process_input_flags(args.dataset, args.testcase)

    result, time = exec_algorithm(input_flags, args.algorithm)
    correct, total = report_accuracy(input_flags)

    test_output = {
        "dataset": args.dataset,
        "testcase": args.testcase,
        "algorithm": args.algorithm,
        "result": result,
        "time": time,
        "correct": correct,
        "total": total,
    }

    with open("test_output.json", "w") as f:
        json.dump(test_output, f, indent=2)

    return result.get("returncode", 1)


if __name__ == "__main__":
    sys.exit(main())
