# Shared VLDB cases and metrics

This directory contains the shared test-case generator, generation
configuration, and CSV-derived schema-v2 metrics.

```sh
cd sherma-jev
python test-case-generator-tool/benchmark.py generate
```

`generate` creates canonical 1,000-row 50% and 90% cases plus a 100-row smoke case under
`generated-smoke`. Copying
cases into a `vldb-*` suite is intentionally manual; this tool performs no
cloning or synchronization. `validate` checks manually copied cases against
the canonical files. `benchmarks` writes `benchmark.json` beside every
available `repaired.csv`.

`generation.corruption_rate` is the total share of eligible cells affected.
`generation.missing_rate` is the share of those corrupted cells allocated to
empty values; the remainder receives character edits. Missing cells and
character-corrupted cells are disjoint. For example, a 1,000-cell table with
50% corruption and 10% missing produces 500 affected cells: 50 missing and 450
malformed.

Corruption selection redraws any cell that would remove the last unchanged
occurrence of an enum value in the sampled original table. This includes numeric
enum columns. Counts and ratios remain exact; selection is no longer uniform
over all cells because rare categories must survive. Impossible budgets raise
an error. `category_preservation_redraws` records rejected selections. Character
edits can still produce additional empty values, as before.

GammaMax and GammaMax-DT report `max_positive_examples_stored` and
`max_negative_examples_stored` per column and as maxima in `benchmark.json`'s
`table`. Positives count distinct training examples after the configured cap;
negatives count distinct examples in the session FIFO, including rejected repair
candidates. FIFO occupancy never decreases, so its final size is its peak.
These count logical examples, not container copies or bytes. Each suite writes
`example-storage-summary.json` with maxima across available datasets and rates
(including smoke), plus the contributing benchmark paths. Legacy measurements
remain null until rerun. `benchmarks` also refreshes these summaries.

Every generated case uses the single dataset manifest at `oracles/<dataset>.json`.
Corruption and sampling settings live in `config.json`. Manifests are manually
curated: `generate` preserves their regexes and enum values, refreshes `shapes`
from each regex, and regenerates cases using those rules. It never infers or
replaces oracle rules. Smoke and full cases use the same manifest.
