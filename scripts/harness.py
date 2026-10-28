import argparse
import subprocess
import sys
import time


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


def exec_algorithm(input_flags, algorithm) -> int:
    start = time.time()

    print(f"Executing algorithm: {algorithm} on dataset: {input_flags['input']}")

    cmd = [
        "python",
        f"algorithm/{algorithm}.py",
        *[f"--{k}={v}" for k, v in input_flags.items()],
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return e.returncode

    end = time.time()
    elapsed = end - start
    print(f"Elapsed time: {elapsed} seconds")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harness for tabular data repair algorithms"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--testcase", required=True)
    parser.add_argument("--algorithm", required=True)
    args = parser.parse_args()

    input_flags = process_input_flags(args.dataset, args.testcase)

    exit_code = exec_algorithm(input_flags, args.algorithm)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
