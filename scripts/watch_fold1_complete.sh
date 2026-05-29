#!/bin/bash
# watch_fold1_complete.sh
# Watches for fold1_best.pth creation, then runs generate_validation_report.py
# Called by: bash scripts/watch_fold1_complete.sh

VENV_PYTHON="/home/iddi/ceph-v2-auto/.venv/bin/python3.10"
VAL_SCRIPT="/home/iddi/ceph-v2-auto/scripts/generate_validation_report.py"
CHECKPOINT="/home/iddi/ceph-v2-auto/outputs/checkpoints/fold1_best.pth"
POLL_INTERVAL=30  # seconds

echo "[watchdog] $(date) — Starting Fold 1 watchdog (PID $$)"
echo "[watchdog] Watching: $CHECKPOINT"
echo "[watchdog] Polling every ${POLL_INTERVAL}s"

# If already exists, exit cleanly (already done)
if [ -f "$CHECKPOINT" ]; then
    echo "[watchdog] $CHECKPOINT already exists — skipping"
    exit 0
fi

# Long-poll loop
while true; do
    if [ -f "$CHECKPOINT" ]; then
        echo ""
        echo "[watchdog] $(date) — DETECTED: fold1_best.pth created!"
        echo "[watchdog] Waiting 5s for file write to flush..."
        sleep 5

        CKPT_SIZE=$(stat -c%s "$CHECKPOINT" 2>/dev/null || echo "0")
        echo "[watchdog] fold1_best.pth size: $CKPT_SIZE bytes"

        if [ "$CKPT_SIZE" -gt 10000000 ]; then
            echo "[watchdog] File size looks valid (>$((10000000/1000000))MB)"
            echo "[watchdog] Running validation report generator..."
            cd /home/iddi/ceph-v2-auto
            $VENV_PYTHON $VAL_SCRIPT 2>&1
            EXIT=$?
            if [ $EXIT -eq 0 ]; then
                echo "[watchdog] $(date) — VALIDATION REPORT GENERATED SUCCESSFULLY"
                echo "[watchdog] Outputs:"
                ls -la /home/iddi/ceph-v2-auto/outputs/val_case_*.png 2>/dev/null
                ls -la /home/iddi/ceph-v2-auto/outputs/VALIDATION_REPORT.md 2>/dev/null
            else
                echo "[watchdog] $(date) — WARNING: validation script exited with code $EXIT"
            fi
        else
            echo "[watchdog] WARNING: file too small (${CKPT_SIZE}B), skipping validation"
        fi
        exit 0
    fi
    sleep $POLL_INTERVAL
done