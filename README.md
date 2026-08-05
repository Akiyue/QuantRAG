# QuantRAG

Does quantization change what small language models trust?
Measuring context–memory arbitration in local RAG.

- `docs/PROPOSAL.md` — research proposal (abstract, hypotheses, contributions)
- `docs/PLAN.md` — execution plan (protocol, metrics, 4-week schedule, risks)

## Setup

Python 3.11 or 3.12 — **not 3.13+**, which has no wheels for `torch`,
`llama-cpp-python` or `autoawq`. `run.sh` refuses to run on the wrong version.

Conda (what the GPU server uses):

```bash
conda create -n quantrag python=3.12 -y && conda activate quantrag
pip install -e ".[dev]"
```

Or a local venv:

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

`run.sh` detects either. **Inference backends are a separate step** — see
[docs/INSTALL.md](docs/INSTALL.md) for the CUDA build of `llama-cpp-python`, the
`llama.cpp` binaries needed to quantize, and the optional tier B stack.

## Layout

```
configs/
  models.yaml        model x precision, GGUF paths
  experiment.yaml    grid, filter pass, dose ladder, XAI settings
data/
  facts.sample.jsonl three worked examples of the schema
src/quantrag/
  schema.py          Fact record + validation
  prompts.py         conditions, instruction modes, ChatML template
  backends/          base interface, llama.cpp, MockBackend
  normalize.py       answer folding + classification (Vietnamese-aware)
  metrics.py         CCR/PRR/QFR, reliance, paired stats, attribution
  runner.py          resumable grid runner
scripts/
  inspect_prompts.py render every condition for eyeball inspection
```

## Running the pipeline

```bash
./run.sh status           # what has run, what is gated
./run.sh setup            # venv + deps + test suite
./run.sh survey           # Wikidata coverage per relation (no GPU)
./run.sh models           # download + quantize the GGUF ladder
./run.sh dataset          # build data/facts.jsonl
./run.sh pilot            # 50 facts twice + noise floor
./run.sh filter           # C0 pass -> parametric-known subsets
./run.sh main             # full behavioural grid
./run.sh dose             # evidence-pressure ladder
./run.sh xai              # span ablation + attribution tables
./run.sh review sample    # draw 200 generations for hand labelling
./run.sh review score     # evaluator/human agreement
./run.sh analyze          # metrics, statistics, result tables
```

Results land in `results/known_all/`. Read them in this order:

| File | Why first |
|---|---|
| `00_margin_control.md` | The falsification test. If flips are fully explained by how close F16 already was to the boundary, the central claim is not about quantization |
| `05_diagnostics.md` | Evaluator sanity: generation vs teacher-forced agreement, label distribution, ambiguous answers |
| `02_flips.md` | Instance-level instability and its **asymmetry** — the directional claim that survives a null net effect |
| `03_reliance.md` | ΔR and ΔP_ctx: the arbitration shift itself |
| `04_interactions.md` | precision × mode and precision × language, where the signal is if the net effect is null |

There is **no target that runs everything end to end**, on purpose. Four points
in this pipeline need a human, and a script that drives past them at 2am
produces numbers nobody can defend. They are enforced as gate files:

| Gate | What it certifies |
|---|---|
| `aliases` | Wikidata aliases hand-filtered; English label added to every Vietnamese alias set |
| `paraphrase` | Counterfactual paraphrases hand-written, not LLM-generated |
| `pilot` | Throughput measured; run-to-run noise smaller than the flip rate you intend to report |
| `known` | Known-subset sizes checked per language before the main grid burns GPU time |
| `margin` | Flips are not fully explained by baseline margin |
| `evaluator` | 200 items hand-labelled, agreement recorded for the paper |

Sign one off with `./run.sh gate <name>`; revoke with `rm .gates/<name>`.

## Check the setup

```bash
pytest -q
python scripts/inspect_prompts.py
python scripts/run_grid.py --pass main --mock --facts data/facts.sample.jsonl
```

`MockBackend` exercises the runner, metrics and resume logic without model
weights, so the pipeline can be developed on a laptop with no GPU.

## Things that will silently corrupt results

Each has a test or an assertion behind it; do not route around them.

- **Same question for filtering and evaluation.** Selecting the known-subset on
  the same question used to evaluate is selection on the baseline arm and
  manufactures an apparent quantization effect. `q_filter` and `q_eval` must be
  different paraphrases; `Fact.validate()` enforces it.
- **Mixed inference stacks.** F16 in one runtime and 4-bit in another compares
  runtimes, not precisions. Tier B exists only as a separate robustness check.
- **Unnormalised log-probabilities across languages.** `London` may be one token
  and `Luân Đôn` three. Length-normalised scores are the reported ones.
- **Continuations that do not start on a token boundary.** Raises
  `BoundaryError` rather than returning a number nobody can trust.
- **CCR without its instruction mode.** The same number means "followed
  instructions" under strict grounding and "was steered by misinformation"
  under truth-seeking.
- **Compliance on C0/C4.** No document endorses either answer there; both are
  still scored, but only as the baseline margin.
- **Attribution without a control span.** Deleting any tokens lowers the
  log-probability; only the excess over a length-matched control is evidence.
- **Discarding per-token log-probabilities.** Re-running the grid is the
  expensive path, and the journal extension needs them.

## Status

Pipeline complete end to end, 49 tests. Every stage from raw CounterFact to the
paper's result tables runs.

Not yet done, and none of it blocks the conference paper:

- figure generation (tables are written as Markdown and CSV; plots are manual)
- the journal-extension work: activation patching, mitigation, real retrieval

## Licence

MIT for the code. The datasets fetched at build time keep their own terms — see
`LICENSE`.
