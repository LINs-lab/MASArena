# BenchAgent / MASArena

Code accompanying the current AAMAS manuscript, **Do More Agents Help?**

BenchAgent compares one task-level controller (`bench_agent`) and six normalized
multi-agent workflows (`evoagent`, `llm_debate`, `camel`, `autogen`, `jarvis`,
`chateval`) through common task loaders, tool interfaces and final evaluators.
The search helper is a shared service, not a separate task-level baseline.
`single_agent` is a different legacy implementation and is not the paper's Core.

## Current reporting protocol

- Ten fixed benchmarks, 3,302 tasks per workflow per run, three runs.
- GPT-4.1 snapshot `gpt-4.1-2025-04-14`; temperature 0.2, top_p 1.0, completion cap
  8,192, except AutoGen critic temperature 0.7.
- Accuracy uses fixed split sizes; failures receive zero credit. Pooled accuracy
  weights benchmarks by their task counts. Variation is the sample standard
  deviation across three runs, not a confidence interval or task-sampling error.
- Token values are retained usage summaries. Round each benchmark mean within
  each run, then average the three runs; pool with benchmark-size weights.
  Missing usage is not reconstructed. Runtime summaries expose observed coverage.
- GAIA is a separate single run over 165 tasks (53/86/26 by level); its final
  evaluator uses `gpt-4o-mini`. The external CC runtime has separate execution
  conditions and is not a controlled replacement for Core.

| Workflow | Pooled accuracy, mean ± sample SD | Pooled tokens per task |
|---|---:|---:|
| Core | 76.65 ± 0.32% | 23,712.97 |
| EvoAgent | 79.70 ± 0.26% | 30,036.80 |
| LLM-Debate | 74.69 ± 0.41% | 32,987.40 |
| CAMEL | 68.59 ± 0.16% | 7,724.76 |
| AutoGen | 67.89 ± 0.54% | 11,634.54 |
| Jarvis | 69.32 ± 0.29% | 8,654.41 |
| ChatEval | 74.94 ± 0.96% | 103,022.69 |

These are the manuscript's retained results, not new model runs performed while
preparing this release. The release fixes implementation mismatches; see
`docs/PAPER_ALIGNMENT.md`. No historical experimental outcomes are overwritten.

## Installation

Python 3.11 or 3.12 is required for model execution. The lock includes the
research workspace's optional dependencies; installing it needs network access.

```bash
uv sync --locked
cp .env.example .env
```

Set provider credentials locally via environment variables or `.env`. Never
include that file in a release. `.env.example` contains empty credential fields.
The offline tests and result verifier use only the Python standard library.

## Reproduce the paper task sets

This branch distributes source code and reproducibility metadata. Dataset records,
logs, trajectories and generated memory state remain local and are not tracked.
Dataset download helpers remain under `data/download/`.

Obtain the processed files from `Brian-Wang117/MASArena-Dataset`, revision
`468540bdc34bc0f8c7ed52185bcdc8281287b6c3`, subject to the source datasets' terms.
`configs/paper/datasets.json` records the supplied local processed-file hashes;
these hashes describe those exact bytes, not a freshly verified upstream download.
DROP's local processed file is already the first 400 of the 800-record file.
Do not resample it. GAIA requires its official 2023 validation data and attachments.

```bash
python3 scripts/prepare_paper_data.py --source-dir data --output-dir data/paper
python3 scripts/run_paper.py --data-dir data/paper
```

The second command prints all 210 broad-suite commands without calling an API.
To run them with your configured provider, use the same command with `--execute`
in the installed environment (`uv run python scripts/run_paper.py ... --execute`).
The prepared files contain the final tasks, so no second sampling limit is used.

For a single evaluation:

```bash
MODEL_NAME=gpt-4.1-2025-04-14 BENCHMARK=math AGENT_SYSTEM=bench_agent \
  DATA_PATH=data/paper/math_test.jsonl ./run_benchmark.sh
```

For broad benchmarks except HotpotQA, the manager has `python_interpreter` and
search tools are disabled; HotpotQA and GAIA use the expanded registry. Every
role retains the final-answer contract. Workflow-specific prompting is described
in the appendix. External tools may need additional provider credentials and
browser dependencies; obtaining those dependencies is separate from offline validation.
Memory and optimizers are outside the reported protocol.

## Offline validation and release

```bash
python3 -m unittest discover -s tests -p 'test_paper_*.py' -v
python3 scripts/verify_paper_results.py --paper-root /path/to/aamas --output /tmp/results-check.json
python3 scripts/build_paper_release.py --paper-root /path/to/aamas \
  --materials-manifest /path/to/materials.json --validation /path/to/validation.json --output-dir dist
```

Release archives include current PDFs, manuscript source, experiment tables,
analysis scripts, supplementary evidence and a checksum manifest. They exclude
credentials, version-control history, caches, generated runtime state and review
notes. The anonymous archive additionally removes author-identifying metadata.
See each archive's `README.md`, `MATERIALS.md` and `VALIDATION.json` for coverage,
commands and the limits of retained evidence. Paid model experiments are not part
of the offline release checks.
