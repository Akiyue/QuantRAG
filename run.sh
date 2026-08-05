#!/usr/bin/env bash
#
# QuantRAG pipeline driver.
#
#   ./run.sh setup            environment + dependency check
#   ./run.sh models           download and quantize the GGUF ladder
#   ./run.sh survey           Wikidata coverage per relation  (no GPU needed)
#   ./run.sh dataset          build data/facts.jsonl
#   ./run.sh pilot            50 facts, timing + non-determinism check
#   ./run.sh filter           C0 pass -> parametric-known subsets
#   ./run.sh main             full behavioural grid
#   ./run.sh dose             evidence-pressure ladder
#   ./run.sh xai              span ablation + attribution tables
#   ./run.sh review sample    draw 200 generations for hand labelling
#   ./run.sh review score     agreement between evaluator and human
#   ./run.sh analyze          metrics, statistics, result tables, figures
#   ./run.sh status           what has run, what is gated
#
# There is deliberately no target that runs everything end to end. Four points
# in this pipeline require a human, and a script that drives past them at 2am
# produces numbers nobody can defend. Those points are enforced as gate files:
# a stage refuses to start until the gate it depends on has been signed off with
#
#   ./run.sh gate <name>
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/.venv/Scripts/python.exe"   # Windows layout
LLAMA_CPP="${LLAMA_CPP:-$ROOT/../llama.cpp}"
HF_MODELS="${HF_MODELS:-$ROOT/models}"
LOGS="$ROOT/logs"
GATES="$ROOT/.gates"

mkdir -p "$LOGS" "$GATES" "$ROOT/runs" "$ROOT/data"

# ---------------------------------------------------------------- utilities

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_off=$'\033[0m'
log()  { printf '%s[%s]%s %s\n' "$c_grn" "$(date +%H:%M:%S)" "$c_off" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_ylw" "$c_off" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

# Run a stage with its output teed to a timestamped log. Long GPU passes get
# interrupted; the log is what tells you where you were.
staged() {
  local name="$1"; shift
  local logfile="$LOGS/${name}__$(date +%Y%m%d-%H%M%S).log"
  log "$name -> $logfile"
  "$@" 2>&1 | tee "$logfile"
}

# --------------------------------------------------------------------- gates
#
# Each gate marks a judgement only a person can make. They are ordinary files so
# that a stale gate can be revoked with `rm`, and so their existence is visible
# in git status rather than buried in someone's memory.

gate_desc() {
  case "$1" in
    aliases)   echo "Wikidata aliases hand-filtered: English label added to every Vietnamese alias set; epithets and semantic stretches removed (PLAN 3.3)" ;;
    templates) echo "Vietnamese relation templates in configs/relations.yaml reviewed by a native speaker - this is the wording the model is judged on (PLAN 3.1)" ;;
    paraphrase) echo "Counterfactual paraphrases hand-written for the XAI subset - NOT LLM-generated, or SemanticGap measures paraphrase quality instead of the model (PLAN 6.4)" ;;
    pilot)     echo "Pilot reviewed: throughput measured, non-determinism smaller than the QFR you expect to report, prompts eyeballed (PLAN day 6)" ;;
    known)     echo "Known-subset sizes checked per language. If Vietnamese KNOWN_ALL is small, the bilingual arm is demoted BEFORE the main grid runs (PLAN day 5)" ;;
    margin)    echo "Margin control run: flips are not fully explained by baseline margin, so the central claim survives (PLAN 6.3, day 12)" ;;
    evaluator) echo "200 items hand-labelled, agreement with the automatic evaluator computed and recorded for the paper (PLAN 5.3)" ;;
    *) echo "unknown gate" ;;
  esac
}

require_gate() {
  local g="$1"
  [[ -f "$GATES/$g" ]] || die "blocked by gate '$g'
  $(gate_desc "$g")
  Sign off with: ./run.sh gate $g"
}

cmd_gate() {
  local g="${1:-}"
  [[ -n "$g" ]] || die "usage: ./run.sh gate <aliases|paraphrase|pilot|known|margin|evaluator>"
  [[ "$(gate_desc "$g")" != "unknown gate" ]] || die "unknown gate: $g"
  printf 'signed %s by %s\n%s\n' "$(date -Iseconds)" "${USER:-${USERNAME:-unknown}}" \
    "$(gate_desc "$g")" > "$GATES/$g"
  log "gate '$g' signed"
}

# --------------------------------------------------------------------- stages

cmd_setup() {
  need uv
  log "creating venv (3.12; torch and llama-cpp-python have no 3.14 wheels)"
  uv venv --python 3.12 || true
  uv pip install -e ".[dev]"
  "$PY" -m pytest -q || die "test suite failed - fix before running anything expensive"
  log "environment ok"
}

cmd_models() {
  need huggingface-cli
  [[ -d "$LLAMA_CPP" ]] || die "llama.cpp not found at $LLAMA_CPP
  git clone https://github.com/ggerganov/llama.cpp \"$LLAMA_CPP\"
  cmake -B build -DGGML_CUDA=ON \"$LLAMA_CPP\" && cmake --build \"$LLAMA_CPP/build\" -j"

  local quantizer
  quantizer="$(find "$LLAMA_CPP" -name 'llama-quantize*' -type f 2>/dev/null | head -1)"
  [[ -n "$quantizer" ]] || die "llama-quantize not built in $LLAMA_CPP"

  mkdir -p "$HF_MODELS"
  for repo in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do
    local short="${repo##*/}"; short="$(echo "$short" | tr '[:upper:]' '[:lower:]')"
    local src="$HF_MODELS/hf/$short"
    local f16="$HF_MODELS/${short}-f16.gguf"

    [[ -d "$src" ]] || huggingface-cli download "$repo" --local-dir "$src"
    [[ -f "$f16" ]] || "$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$src" \
        --outfile "$f16" --outtype f16

    # One quantizer, one runtime, four bit widths: the only thing that differs
    # between arms is the number of bits.
    for q in Q8_0 Q4_K_M Q3_K_M; do
      local lower; lower="$(echo "$q" | tr '[:upper:]' '[:lower:]')"
      local out="$HF_MODELS/${short}-${lower}.gguf"
      [[ -f "$out" ]] || "$quantizer" "$f16" "$out" "$q"
    done
  done

  log "checksums -> $HF_MODELS/SHA256SUMS"
  (cd "$HF_MODELS" && sha256sum ./*.gguf > SHA256SUMS)
  log "model ladder ready"
}

cmd_survey() {
  # Answers the riskiest open question - whether the Vietnamese arm has enough
  # paired items - with no GPU and no model weights.
  staged survey "$PY" scripts/survey_counterfact_vi.py "${1:-2000}"
  warn "pick relations by Vietnamese coverage before building the dataset (PLAN 3.2)"
}

cmd_dataset() {
  staged dataset "$PY" scripts/build_dataset.py "$@"
  staged inspect "$PY" scripts/inspect_prompts.py data/facts.jsonl
  log "eyeball the prompts above; prompt bugs are invisible in aggregate numbers"
  warn "now review the alias lists and configs/relations.yaml, then:"
  warn "  ./run.sh gate aliases && ./run.sh gate templates"
}

cmd_pilot() {
  require_gate aliases
  require_gate templates
  # Two questions: how fast is it really, and how much does the same config move
  # between identical runs. If run-to-run noise is not clearly smaller than the
  # flip rate you intend to report, QFR is measuring the GPU, not quantization.
  staged pilot_a "$PY" scripts/run_grid.py --pass main --limit 50 --out-dir runs/pilot_a
  staged pilot_b "$PY" scripts/run_grid.py --pass main --limit 50 --out-dir runs/pilot_b
  staged noise "$PY" scripts/check_noise.py runs/pilot_a runs/pilot_b
}

cmd_filter() {
  require_gate pilot
  # Runs on q_filter, never q_eval. Selecting items on the same question used to
  # evaluate is selection on the baseline arm and manufactures an effect.
  staged filter "$PY" scripts/run_grid.py --pass filter
  staged splits "$PY" scripts/build_splits.py
}

cmd_main() {
  require_gate pilot
  require_gate known
  staged main "$PY" scripts/run_grid.py --pass main
}

cmd_dose() {
  require_gate pilot
  # Mandatory to COLLECT even if week 3 has no time to analyse it: this is the
  # raw material for the journal metric contribution, and re-running months
  # later loses comparability with the conference numbers.
  staged dose "$PY" scripts/run_grid.py --pass dose
}

cmd_xai() {
  require_gate paraphrase
  staged xai "$PY" scripts/run_ablation.py "$@"
  staged xai_analysis "$PY" scripts/analyze_xai.py
}

cmd_review() {
  # Every headline rate is built on the automatic evaluator. This is what puts
  # a number on how much it can be trusted, and that number goes in the paper.
  case "${1:-sample}" in
    sample) staged review_sample "$PY" scripts/evaluator_review.py sample --n 200
            warn "fill in the human_label column of results/evaluator_review.csv" ;;
    score)  staged review_score  "$PY" scripts/evaluator_review.py score ;;
    *) die "usage: ./run.sh review [sample|score]" ;;
  esac
}

cmd_analyze() {
  require_gate evaluator
  staged analyze "$PY" scripts/analyze.py --split known_all
  # The secondary view, selected on the baseline arm alone. Reported with the
  # bias stated so a reader can see the conclusions do not depend on it.
  staged analyze_fp16 "$PY" scripts/analyze.py --split known_fp16
  staged figures "$PY" scripts/make_figures.py --split known_all
  warn "read results/known_all/00_margin_control.md and 05_diagnostics.md first"
  warn "then open the figure PNGs - the palette is validated, the layout is not"
}

cmd_status() {
  printf '\ngates\n'
  for g in templates aliases paraphrase pilot known margin evaluator; do
    if [[ -f "$GATES/$g" ]]; then
      printf '  %s[x]%s %-11s %s\n' "$c_grn" "$c_off" "$g" "$(head -1 "$GATES/$g")"
    else
      printf '  [ ] %-11s %s\n' "$g" "$(gate_desc "$g" | cut -c1-70)..."
    fi
  done

  printf '\nartifacts\n'
  for f in data/facts.jsonl models/SHA256SUMS; do
    [[ -e "$f" ]] && printf '  %s[x]%s %s\n' "$c_grn" "$c_off" "$f" \
                  || printf '  [ ] %s\n' "$f"
  done

  printf '\nruns\n'
  if compgen -G "runs/*.jsonl" > /dev/null; then
    for f in runs/*.jsonl; do
      printf '  %-52s %8s lines\n' "$(basename "$f")" "$(wc -l < "$f")"
    done
  else
    printf '  (none)\n'
  fi
  printf '\n'
}

# ----------------------------------------------------------------- dispatch

case "${1:-}" in
  setup)   shift; cmd_setup   "$@" ;;
  models)  shift; cmd_models  "$@" ;;
  survey)  shift; cmd_survey  "$@" ;;
  dataset) shift; cmd_dataset "$@" ;;
  pilot)   shift; cmd_pilot   "$@" ;;
  filter)  shift; cmd_filter  "$@" ;;
  main)    shift; cmd_main    "$@" ;;
  dose)    shift; cmd_dose    "$@" ;;
  xai)     shift; cmd_xai     "$@" ;;
  review)  shift; cmd_review  "$@" ;;
  analyze) shift; cmd_analyze "$@" ;;
  gate)    shift; cmd_gate    "$@" ;;
  status)  shift; cmd_status  "$@" ;;
  *) sed -n '3,28p' "${BASH_SOURCE[0]}" | sed 's/^# \?//' ; exit 1 ;;
esac
