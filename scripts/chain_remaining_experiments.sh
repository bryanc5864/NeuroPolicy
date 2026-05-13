#!/usr/bin/env bash
# Chained experiment launcher.  Waits for GPU memory < 5GB then launches each
# remaining experiment sequentially.  Logs to logs/chain_<name>.log.
set -u
cd "$(dirname "$0")/.." || exit 1

wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 5000 ]; then
      echo "[$(date '+%F %T')] GPU free (used=${used}MB)"
      return 0
    fi
    sleep 60
  done
}

run() {
  name=$1; shift
  echo "[$(date '+%F %T')] ====================== ${name} ======================"
  wait_gpu
  echo "[$(date '+%F %T')] launching: $*"
  PYTHONIOENCODING=utf-8 "$@" 2>&1 | tee "logs/chain_${name}.log"
  echo "[$(date '+%F %T')] ${name} complete"
}

# 1) Extend NeuroPolicy canonical bci2a to all 9 subjects (resolves issue #3).
run neuropolicy_9subj python scripts/neuropolicy_canonical_bci2a_conformer_9subj.py

# 2) CVaR alpha ablation (resolves issue #5).
run cvar_alpha    python scripts/cvar_alpha_ablation_bci2a.py

# 3) Conformer at full 2000-epoch budget (resolves issue #1).
run conformer_2k  python scripts/train_mandatory_bci2a.py \
                     --encoder conformer \
                     --epochs 2000 \
                     --subjects 1 2 3 4 5 6 7 8 9 \
                     --seeds 0 1 2 \
                     --out experiments/mandatory_bci2a_conformer_2k

echo "[$(date '+%F %T')] all chained experiments complete"
