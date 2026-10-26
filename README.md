# Sherma-JEV

## Requirements

- Python 3.10 or newer. Sherma itself is Python and does not need compilation.
- CMake 3.20 or newer and a C++20 compiler for compiling dataset oracles.
- A Typesafe API key for JEV candidate selection.

## Setup

```console
python -m pip install -r requirements.txt
```

Set `TYPESAFE_API_KEY` in the environment, or create `.env` beside this README:

```dotenv
TYPESAFE_API_KEY=your-api-key
```

## Build Sherma

There is no native compilation step for Sherma. After setup, run the Python
entry point directly as shown below. Only the external dataset oracles are
compiled.

## Compile an oracle

Each file in `test-case-generator-tool/oracles/` is the complete oracle
manifest for one dataset. Compile it with `oracle/build.py`; the build script
generates the embedded manifest and invokes CMake for you.

On Linux or macOS:

```console
python oracle/build.py test-case-generator-tool/oracles/adult_20.json build/oracles/adult_20
```

On Windows, the output must end in `.exe`:

```console
python oracle/build.py test-case-generator-tool/oracles/adult_20.json build/oracles/adult_20.exe
```

Repeat the command with another `<dataset>.json` file to build that dataset's
oracle. One executable serves every column in its dataset; do not build
per-column executables.

The manifest is a strict build input containing exactly `dataset`, `source`,
and an `oracles` object keyed by column name. Each rule is either an `enum`
with `values`, or a `regex` with `regex` and `shapes`. Empty strings and `?`
are always rejected. The resulting executable embeds every rule and does not
read the manifest at runtime.

Sherma sends the executable `{"column":"age","candidates":["42","?"]}` on
standard input and expects an ordered JSON boolean array such as `[true,false]`
on standard output. Each request runs in a separate process.

## Run Sherma

Sherma does not inspect the oracle manifest. Pass only the compiled executable:

```console
python algorithm/sherma.py \
  --input corrupted.csv \
  --ground-truth original.csv \
  --schema schema.json \
  --config algorithm-config.json \
  --oracle build/oracles/adult_20 \
  --output output.csv \
  --telemetry telemetry.json
```

Use `build/oracles/adult_20.exe` for `--oracle` on Windows. The input CSV,
ground-truth CSV, and schema must have the same columns and row count.

## Configuration

Use `test-config.json` as the configuration template. `-1` means unlimited for
fields that support it.

- `seed`: deterministic random seed; default `0`.
- `repair.n`: n-gram size.
- `repair.ngrams_batch_size`: n-grams processed per batch; positive or `-1`.
- `repair.max_candidate_length`: maximum candidate length; positive or `-1`.
- `repair.num_candidates`: maximum accepted candidates per malformed value;
  default `5`.
- `repair.edit_distance_cap`: maximum Levenshtein distance; default `3`.
- `batching.initial_positive_capacity`: initial positive examples retained;
  positive or `-1`.
- `batching.initial_negative_capacity`: initial negative examples retained;
  positive or `-1`.
- `batching.positive_capacity`: ongoing positive-example capacity; positive or
  `-1`, and not below its initial capacity.
- `batching.negative_capacity`: ongoing negative-example capacity; positive or
  `-1`, and not below its initial capacity.
- `batching.cell_timeout_seconds`: timeout for each malformed value, in seconds.
- `jev.confidence_threshold`: minimum accepted JEV confidence from `0` to `1`;
  default `0.5`.
- `jev.timeout_seconds`: timeout for each JEV request, in seconds; default `15`.
- `limits.max_iterations`: maximum repair-search iterations; positive or `-1`.
- `limits.max_states`: maximum automaton states; positive or `-1`.
- `limits.max_queue_size`: maximum repair-search queue size; positive or `-1`.
- `limits.max_rsr_candidates`: maximum RSR candidates; positive or `-1`.

## Run the test cases

The test runner first runs the unit tests, compiles one oracle per selected
dataset, and then passes those executables to Sherma. No separately compiled
oracle is required.

Run the one-case smoke suite:

```console
python test_suite.py smoke
```

Run all eight bundled cases:

```console
python test_suite.py test
```

The equivalent Make targets are:

```console
make smoke
make test
```

Remove generated test artifacts with:

```console
make clean
```

## Generate corrupted datasets

The dataset generator is bundled in `test-case-generator-tool`. From this
directory, generate all configured cases with:

```console
python test-case-generator-tool/benchmark.py generate
```

To add a new dataset:

1. Create `test-case-generator-tool/datasets/<dataset-name>/clean.csv`. The CSV
   must have a header with unique, non-empty column names.
2. Create `test-case-generator-tool/oracles/<dataset-name>.json`. Its `oracles` object
   must contain one entry per CSV column, keyed by column name. Use an existing
   manifest as a template; each entry defines either an accepted `regex` or an
   `enum` with its accepted `values`. Empty strings and `?` are always rejected.
3. Add `<dataset-name>` to the `DATASETS` tuple near the top of
   `test-case-generator-tool/benchmark.py`.
4. Adjust `test-case-generator-tool/config.json` if needed. `max_rows` controls the sampled
   row count, `corruption_rates` controls the fraction of eligible cells changed,
   and `missing_rate` controls how many changed cells are made empty. The
   remaining changed cells receive random character insertions, deletions, or
   substitutions.
5. Run the generation command above. To generate only one configured corruption
   rate, pass its percentage, for example:

   ```console
   python test-case-generator-tool/benchmark.py generate --rates 50
   ```

Generated cases are written to
`test-case-generator-tool/generated-<rate>/<dataset-name>/`. Each case contains the sampled
`original.csv`, its `corrupted.csv`, and `case.json` describing every changed
cell. The generator uses the configured seed, so identical inputs and settings
produce identical cases.
