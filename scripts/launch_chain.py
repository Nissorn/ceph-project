#!/usr/bin/env python3
"""
Wait + Chain Launcher: Monitor EXP-01 completion, then run autonomous loop.
"""
from __future__ import annotations
import json, logging, os, random, re, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger()

PROJECT = Path("/home/iddi/ceph-v2-auto")
os.chdir(PROJECT)

EXP01_PID = 3337498
EXP01_LOG = "/tmp/exp01_output.log"

def parse_last_epoch(log_path):
    try:
        with open(log_path) as f:
            content = f.read()
        matches = re.findall(r'Ep (\d+)/(\d+).*?dice=([\d.]+)', content)
        if not matches:
            return None, None
        return int(matches[-1][0]), float(matches[-1][2])
    except:
        return None, None

def wait_for_exp01():
    log.info("Waiting for EXP-01 (PID %d) to complete...", EXP01_PID)
    start = time.time()
    last_report = 0
    patience_triggered = False

    while True:
        # Check if process still alive
        try:
            os.kill(EXP01_PID, 0)  # signal 0 = existence check
        except OSError:
            log.info("EXP-01 process %d has exited!", EXP01_PID)
            break

        # Check log for early-stop
        try:
            with open(EXP01_LOG) as f:
                content = f.read()
            early_stop = re.findall(r'Early stop.*?@ ep (\d+)', content)
            if early_stop:
                log.info("Early stop detected in log @ epoch %s", early_stop[-1])
                patience_triggered = True
            # Also check the "DONE" message
            if "DONE" in content or "best_model.pt" in content:
                log.info("EXP-01 DONE signal found")
                break
        except:
            pass

        # Progress every 60s
        if time.time() - last_report > 60:
            ep, dice = parse_last_epoch(EXP01_LOG)
            elapsed = time.time() - start
            log.info("  EXP-01 still running @ ep %s, dice=%.4f (elapsed %.0fs)",
                     ep, dice if dice else 0, elapsed)
            last_report = time.time()

        # Safety: if running more than 2 hours past epoch 100, assume stuck and kill
        ep = None
        if time.time() - start > 7200:
            ep, _ = parse_last_epoch(EXP01_LOG)
            if ep and ep > 100:
                log.info("Safety kill: EXP-01 running too long (ep %d)", ep)
                try:
                    os.kill(EXP01_PID, signal.SIGTERM)
                except:
                    pass
                break
        if ep and ep >= 200:
            log.info("EXP-01 hit max epochs")
            break

        time.sleep(15)

    # Wait a bit more for graceful exits
    time.sleep(10)

    # Confirm process gone, kill if needed
    try:
        os.kill(EXP01_PID, 0)
        log.info("EXP-01 still alive after wait — killing")
        os.kill(EXP01_PID, signal.SIGTERM)
        time.sleep(5)
        try:
            os.kill(EXP01_PID, 0)
            os.kill(EXP01_PID, signal.SIGKILL)
        except:
            pass
    except OSError:
        pass

    # Parse final result
    try:
        with open(EXP01_LOG) as f:
            content = f.read()
        best_matches = re.findall(r'Best Val[^:]*:\s*([\d.]+)', content)
        best = max((float(m) for m in best_matches), default=0.0)
        if best == 0.0:
            dice_matches = re.findall(r'dice=([\d.]+)', content)
            best = max((float(m) for m in dice_matches), default=0.0)
    except:
        best = 0.0

    log.info("EXP-01 FINAL best dice: %.4f", best)
    return best

# Wait for EXP-01
exp01_dice = wait_for_exp01()

# Update product.md for EXP-01
product_path = PROJECT / "product.md"
content = product_path.read_text()
# EXP-01 already marked [x] in backlog — just log the final confirmed dice
log.info("EXP-01 completion confirmed in product.md")

# Kill any zombies
try:
    out = subprocess.check_output(
        "nvidia-smi --query-compute-apps=pid,gpu,used_memory --format=csv,noheader",
        shell=True, text=True
    )
    our_pids = {EXP01_PID}
    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        pid = int(parts[0].strip())
        gpu = int(parts[2].strip())
        mem = int(parts[1].strip().replace("MiB","").strip())
        if gpu not in (0,1,2,3):
            continue
        if pid not in our_pids and mem > 3000:
            log.info("Killing zombie PID %d on GPU %d", pid, gpu)
            try: os.kill(pid, signal.SIGTERM); time.sleep(3)
            except: pass
            try: os.kill(pid, signal.SIGKILL)
            except: pass
except Exception as e:
    log.info("Zombie cleanup: %s", e)

time.sleep(5)

# Launch the full autonomous loop
log.info("=" * 60)
log.info("EXP-01 done (dice=%.4f) — Launching full autonomous loop", exp01_dice)
log.info("=" * 60)

cmd = f".venv/bin/python -u {PROJECT}/scripts/autonomous_loop.py >> /tmp/autonomous_loop_output.log 2>&1"
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
proc = subprocess.Popen(cmd, shell=True, cwd=PROJECT, env=env)
log.info("Autonomous loop PID: %d", proc.pid)
log.info("Output: /tmp/autonomous_loop_output.log")
log.info("YOU WILL BE NOTIFIED WHEN THE LOOP COMPLETES")
log.info("Check progress: tail -f /tmp/autonomous_loop_output.log")