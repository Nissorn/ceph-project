#!/bin/bash
# ============================================================================
# scripts/run_overnight_fleet.sh
# ============================================================================
# Autonomous Night Watchdog: parallel 4-GPU 5-fold landmark training fleet.
# Bypasses GPU 0,3,5,7 (external workloads: shrimp-tf + others).
# Uses ONLY GPU 1, 2, 4, 6 — all verified vacant 2026-05-26.
#
# Fold distribution:
#   GPU 1  → Fold 1  (background)
#   GPU 1  → Fold 2  (background)   ← same GPU: sequential, not parallel
#   GPU 2  → Fold 3  (background)
#   GPU 4  → Fold 4  (background)
#   GPU 6  → Fold 5  (background)
#
# After ALL folds complete → auto-trigger generate_validation_report.py
# on GPU 1 (the master worker that launched folds 1+2 sequentially).
# ============================================================================

set -euo pipefail

LOGDIR="/home/iddi/ceph-v2-auto/logs"
MARKER="/tmp/overnight_fleet.done"
ERR_MARKER="/tmp/overnight_fleet.error"

mkdir -p "$LOGDIR"

echo "=========================================================="
echo " Overnight Fleet Launch  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
echo "GPU isolation: external workloads (GPU 0,3,5,7) — BYPASSED"
echo "Worker GPUs:   1, 2, 4, 6"
echo "Fold mapping:  GPU1→F1+F2  GPU2→F3  GPU4→F4  GPU6→F5"
echo "Post-completion trigger: generate_validation_report.py"
echo "=========================================================="

# ── Cleanup any stale markers ───────────────────────────────────────────────
rm -f "$MARKER" "$ERR_MARKER"

# ── Training fold functions ──────────────────────────────────────────────────
launch_fold() {
    local fold=$1
    local gpu=$2
    local log="$LOGDIR/fold${fold}_gpu${gpu}.log"
    echo "[Fold $fold] launching on GPU $gpu → $log"
    CUDA_VISIBLE_DEVICES=$gpu \
        .venv/bin/python scripts/run_phase2_train.py \
            --config config.yaml \
            --fold "$fold" \
            > "$log" 2>&1 &
    echo "  PID $! launched for Fold $fold on GPU $gpu"
}

# ── Launch all 5 folds across 4 GPUs ────────────────────────────────────────
echo ""
echo ">>> Launching Fold 1 on GPU 1..."
launch_fold 1 1

echo ">>> Launching Fold 2 on GPU 1 (sequential on same GPU)..."
launch_fold 2 1

echo ">>> Launching Fold 3 on GPU 2..."
launch_fold 3 2

echo ">>> Launching Fold 4 on GPU 4..."
launch_fold 4 4

echo ">>> Launching Fold 5 on GPU 6..."
launch_fold 5 6

echo ""
echo ">>> All 5 folds dispatched. Waiting for completion..."
echo ""

# ── Wait for all background jobs ─────────────────────────────────────────────
FAIL=0
for job in $(jobs -p); do
    wait "$job" || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo ""
    echo "!!! ONE OR MORE FOLDS FAILED — see logs above"
    touch "$ERR_MARKER"
    exit 1
fi

echo ""
echo ">>> ALL 5 FOLDS COMPLETED SUCCESSFULLY"
echo ""

# ── Post-completion: generate validation report on GPU 1 ───────────────────
echo ">>> Post-completion trigger: generate_validation_report.py (GPU 1)"
CUDA_VISIBLE_DEVICES=1 \
    .venv/bin/python scripts/generate_validation_report.py \
    > "$LOGDIR/validation_report.log" 2>&1

if [ $? -eq 0 ]; then
    echo ">>> Validation report generated successfully"
    touch "$MARKER"
else
    echo "!!! Validation report generation failed"
    touch "$ERR_MARKER"
    exit 1
fi

echo ""
echo "=========================================================="
echo " Overnight Fleet COMPLETE  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
echo "Log files: $LOGDIR/fold{1-5}_*.log"
echo "Validation report: $LOGDIR/validation_report.log"
echo "Done marker: $MARKER"
echo "=========================================================="