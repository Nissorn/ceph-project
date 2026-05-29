#!/usr/bin/env python3
"""
Watchdog Monitor for auto_research_iter1.py
- Checks GPU status every 5 minutes
- Tracks best Dice score from logs
- Auto-updates product.md on new best
- Auto-commits to git
- Restarts process if crashed
"""
import subprocess
import time
import re
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/iddi/ceph-v2-auto")
PROCESS_CMD = "auto_research_iter1"
LOG_FILE = PROJECT_ROOT / "watchdog.log"
PRODUCT_MD = PROJECT_ROOT / "product.md"
PROCESS_SESSION = "proc_ccf6500a703a"

GPU_IDS = [0, 1, 2, 3]
CHECK_INTERVAL = 300  # 5 minutes

best_dice = 0.3235  # last known best from log
start_time = datetime.now()

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def get_gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        status = {}
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            idx = int(parts[0])
            util = int(parts[1].replace("%", ""))
            mem_used = int(parts[2].replace(" MiB", ""))
            mem_total = int(parts[3].replace(" MiB", ""))
            status[idx] = {"util": util, "mem_used": mem_used, "mem_total": mem_total}
        return status
    except Exception as e:
        log(f"ERROR getting GPU status: {e}")
        return {}

def _find_pid():
    """Dynamically find auto_research_iter1.py PID."""
    try:
        result = subprocess.run(["pgrep", "-f", "auto_research_iter1.py"],
                               capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        return pids[0] if pids else None
    except:
        return None

def is_process_alive(pid):
    if not pid:
        return False
    try:
        subprocess.run(["ps", "-p", pid], capture_output=True, check=True)
        return True
    except:
        return False

def get_latestExperiment_and_bestdice():
    pid = _find_pid()
    if pid:
        try:
            result = subprocess.run(
                ["tail", "-n", "200", f"/proc/{pid}/fd/1"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout
        except:
            pass
    # Fallback: check models directories
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return ""
    experiments = sorted(models_dir.glob("exp????_*"), key=lambda p: p.stat().st_mtime)
    return f"Last dirs: {[e.name for e in experiments[-5:]]}"

def restart_process():
    log("PROCESS CRASHED — restarting...")
    try:
        subprocess.run(["tmux", "kill-session", "-t", "autoresearch"], capture_output=True)
    except:
        pass
    subprocess.run([
        "tmux", "new-session", "-d", "-s", "autoresearch",
        f"cd {PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=0,1,2,3 {PROJECT_ROOT}/.venv/bin/python3 {PROJECT_ROOT}/scripts/auto_research_iter1.py --epochs-per-run 12"
    ], capture_output=True)
    log("Restarted in tmux session 'autoresearch'")

def update_product_md(exp_num, dice, iou, arch_name, notes=""):
    try:
        content = PRODUCT_MD.read_text()
    except:
        return

    # Find the results table in section 7 and append/update
    marker = "<!-- WIDGET UPDATE -->"
    entry = f"\n| #{exp_num}  | {dice:.4f}  | {iou:.4f} | {arch_name}          | {notes} |"

    if marker in content:
        content = content.replace(marker, entry + f"\n{marker}")
    else:
        # Append after best results table
        section_end = content.find("### Key Observations")
        if section_end == -1:
            content += entry
        else:
            content = content[:section_end] + entry + "\n" + content[section_end:]

    PRODUCT_MD.write_text(content)
    log(f"Updated product.md: exp#{exp_num} dice={dice:.4f}")

def auto_commit():
    try:
        subprocess.run(["git", "add", "product.md"], capture_output=True)
        result = subprocess.run([
            "git", "commit", "-m",
            f"chore(ai): [Watchdog] new best score detected"
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            log("Committed to git")
            subprocess.run(["git", "push", "origin", "optimize"], capture_output=True)
            log("Pushed to origin/optimize")
        else:
            log(f"Git commit failed: {result.stderr}")
    except Exception as e:
        log(f"Git error: {e}")

def run_watchdog():
    log("=" * 60)
    log("WATCHDOG MONITOR STARTED")
    log(f"Check interval: {CHECK_INTERVAL}s")
    log(f"Using pgrep to track auto_research_iter1.py dynamically")
    log("=" * 60)

    log("Loading last best dice from product.md...")
    global best_dice
    try:
        content = PRODUCT_MD.read_text()
        # Parse latest best from table
        for line in content.split("\n"):
            m = re.search(r'\|\s*#(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', line)
            if m:
                d = float(m.group(2))
                if d > best_dice:
                    best_dice = d
    except:
        pass
    log(f"Starting best_dice threshold: {best_dice:.4f}")

    consecutive_zero_util = {i: 0 for i in GPU_IDS}

    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"\n--- CHECK {ts} ---")

        # 1. GPU status
        gpu_status = get_gpu_status()
        log("GPU Status:")
        all_active = True
        for i in GPU_IDS:
            if i in gpu_status:
                s = gpu_status[i]
                log(f"  GPU{i}: {s['util']}% util | {s['mem_used']}/{s['mem_total']} MiB")
                if s['util'] < 1 and s['mem_used'] < 500:
                    consecutive_zero_util[i] += 1
                else:
                    consecutive_zero_util[i] = 0
            else:
                log(f"  GPU{i}: NOT FOUND in nvidia-smi")
                all_active = False

        pid = _find_pid()
        alive = is_process_alive(pid)
        log(f"Process alive: {alive} (PID: {pid})")
        if not alive:
            log("WARNING: auto_research_iter1.py not found!")
            # Check if new process exists
            pids = _find_pid()
            if pids:
                log(f"Found replacement PID: {pids}")
            else:
                log("No replacement process found — restarting...")
                restart_process()

        # 3. Check for new best in model folders
        models_dir = PROJECT_ROOT / "models"
        new_best_found = False
        if models_dir.exists():
            configs = sorted(models_dir.glob("exp????_*/config.json"), key=lambda p: p.stat().st_mtime)
            if configs:
                latest_config = configs[-1]
                try:
                    cfg = json.loads(latest_config.read_text())
                    dice = cfg.get("val_dice", 0)
                    iou = cfg.get("val_iou", 0)
                    arch = cfg.get("arch_name", "?")
                    run_id = latest_config.parent.name
                    exp_num = cfg.get("experiment_index", "?")

                    if dice > best_dice:
                        log(f"NEW BEST: exp#{exp_num} dice={dice:.4f} iou={iou:.4f} arch={arch}")
                        update_product_md(exp_num, dice, iou, arch, "watchdog update")
                        best_dice = dice
                        auto_commit()
                        new_best_found = True
                    else:
                        log(f"Latest experiment: exp#{exp_num} dice={dice:.4f} (best: {best_dice:.4f})")
                except Exception as e:
                    log(f"Error reading config: {e}")

        # 4. Check for stuck GPUs (0% util for 3+ consecutive checks = 15 min)
        for i in GPU_IDS:
            if consecutive_zero_util[i] >= 3:
                log(f"WARNING: GPU {i} at 0% util for {consecutive_zero_util[i]*CHECK_INTERVAL}s")

        elapsed_h = (datetime.now() - start_time).total_seconds() / 3600
        log(f"Elapsed: {elapsed_h:.1f}h | Best Dice: {best_dice:.4f} | Next check in {CHECK_INTERVAL}s")

        log(f"Sleeping {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_watchdog()
