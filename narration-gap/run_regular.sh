#!/usr/bin/env bash

# run_regular.sh — E1 (explained narration) and E2 (wording replication) for
#
# Usage:
#   bash run_regular.sh smoke        # ~5-10 min real-model check (llama, 2 instances)
#   bash run_regular.sh e1           # E1, both seeds
#   bash run_regular.sh e2           # E2, both seeds
#   bash run_regular.sh all          # E1 then E2
#   bash run_regular.sh report       # pooled tables from whatever is on disk
#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/oracle_infidelity_rv.py"
SEEDS=(7 759904)
ENGINE="${RV_ENGINE:-ollama}"
OUTDIR="${RV_OUTDIR:-results_regular}"
E1_BASE="${E1_BASE:-16}"
E2_ORIGINAL="${E2_ORIGINAL:-1}"
MODELS_ARG=()
[ -n "${MODELS:-}" ] && MODELS_ARG=(--models "$MODELS")

export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30}"
mkdir -p "$OUTDIR"

log_time() {  # phase seed seconds
  echo "$(date +%Y-%m-%dT%H:%M:%S),$1,$2,$ENGINE,$3" >> "$OUTDIR/runtimes_regular.csv"
}

run_phase() {  # phase seed extra-args...
  local phase="$1" seed="$2"; shift 2
  local t0=$SECONDS
  echo; echo ">>> $phase seed=$seed engine=$ENGINE $(date)"
  python3 "$SCRIPT" run --phase "$phase" --seed "$seed" --engine "$ENGINE" \
      --outdir "$OUTDIR" ${MODELS_ARG[@]+"${MODELS_ARG[@]}"} "$@" | tee "$OUTDIR/log_${phase}_seed${seed}.txt"
  log_time "$phase" "$seed" $((SECONDS - t0))
}

report() {
  local e1=() e2=()
  for s in "${SEEDS[@]}"; do
    [ -f "$OUTDIR/raw_explain_seed$s.jsonl" ]  && e1+=("$OUTDIR/raw_explain_seed$s.jsonl")
    [ -f "$OUTDIR/raw_wordings_seed$s.jsonl" ] && e2+=("$OUTDIR/raw_wordings_seed$s.jsonl")
  done
  if [ ${#e1[@]} -gt 0 ]; then
    python3 "$SCRIPT" score --phase explain --raw "${e1[@]}" --out "$OUTDIR/explain_pooled.csv" \
      | tee "$OUTDIR/report_explain.txt"
  fi
  if [ ${#e2[@]} -gt 0 ]; then
    python3 "$SCRIPT" score --phase wordings --raw "${e2[@]}" --out "$OUTDIR/wordings_pooled.csv" \
      | tee "$OUTDIR/report_wordings.txt"
  fi
}

e1() { for s in "${SEEDS[@]}"; do run_phase explain "$s" --base "$E1_BASE"; done; }
e2() {
  local extra=(); [ "$E2_ORIGINAL" = "1" ] && extra=(--include-original)
  for s in "${SEEDS[@]}"; do run_phase wordings "$s" ${extra[@]+"${extra[@]}"}; done
}

case "${1:-}" in
  smoke)
    default_model=meta_llama8b; [ "$ENGINE" = "fake" ] && default_model=fake
    MODELS_ARG=(--models "${MODELS:-$default_model}")
    run_phase explain 7 --base 2
    echo; echo ">>> smoke: inspect a raw response:"
    python3 - "$OUTDIR/raw_explain_seed7.jsonl" <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
r = recs[-1]
print("attack:", r["attack"], "| defense:", r["defense"], "| error:", r["error"])
print(r["response"][-1200:])
EOF
    ;;
  e1)     e1; report ;;
  e2)     e2; report ;;
  all)    e1; e2; report ;;
  report) report ;;
  *) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 1 ;;
esac
