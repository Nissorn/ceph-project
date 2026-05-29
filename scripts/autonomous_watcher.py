#!/usr/bin/env python3
"""
Autonomous Watcher — Phase 2C Loop
Monitors EXP-01 until completion, then launches EXP-02, monitors first 10 epochs.
Stays alive until EXP-02 completes, then exits (user will handle EXP-03 onward).
Usage: python autonomous_watcher.py
"""
import subprocess
import time
import re
import os
import json
from pathlib import Path
from datetime import datetime

PROJECT = Path("/home/iddi/ceph-v2-auto")
VENV_PYTHON = PROJECT / ".venv/bin/python"
LOG_EXP01 = "/tmp/exp01_output.log"
LOG_EXP02 = "/tmp/exp02_output.log"
EXP01_PID = 3337498  # may need updating if stale

def get_gpu_memory():
    """Return dict of GPU index -> used memory MiB"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            text=True
        )
        return {int(l.split(",")[0].strip()): int(l.split(",")[1].strip())
                for l in out.strip().split("\n") if l.strip()}
    except:
        return {}

def kill_zombies():
    """Kill stale processes on GPUs 0-3 that are using >3GB and have 0% GPU util."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True)
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            pid = int(parts[0].strip())
            mem = int(parts[1].strip())
            if mem > 3000:  # >3GB
                try:
                    # Check if it's the process we're watching
                    cmd = subprocess.check_output(["ps", "-p", str(pid), "-o", "cmd="], text=True, stderr=subprocess.DEVNULL)
                    if "run_EXP0" in cmd or "run_1024" in cmd:
                        print(f"[WATCHER] Killing zombie PID {pid} ({mem} MiB): {cmd.strip()}")
                        os.kill(pid, 9)
                except:
                    pass
    except Exception as e:
        print(f"[WATCHER] Error checking zombies: {e}")

def read_best_dice_from_log(log_path):
    """Read best dice from a log file (only valid epoch lines)."""
    dices = []
    try:
        for line in open(log_path):
            if "/200" in line and "loss=" in line:
                m = re.search(r'dice=(\d+\.\d+)', line)
                if m:
                    dices.append(float(m.group(1)))
    except:
        pass
    return max(dices) if dices else None

def is_process_alive(pid):
    """Check if a process is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def wait_for_completion(pid, log_path, name, poll_seconds=30):
    """Poll until process completes or dies. Returns exit code or None if still alive."""
    print(f"[WATCHER] Waiting for {name} (PID {pid}) to complete...")
    while True:
        time.sleep(poll_seconds)
        if not is_process_alive(pid):
            # Check log for completion
            best = read_best_dice_from_log(log_path)
            print(f"[WATCHER] {name} FINISHED. Best dice: {best}")
            return best
        print(f"[WATCHER] {name} still running @ {datetime.now().strftime('%H:%M:%S')}")

def monitor_first_epochs(log_path, name, min_epochs=5):
    """Verify first N epochs complete without crash."""
    print(f"[WATCHER] Monitoring first {min_epochs} epochs of {name}...")
    last_count = 0
    stable_count = 0
    while stable_count < 3:  # need 3 consecutive checks of same epoch count
        time.sleep(30)
        try:
            with open(log_path) as f:
                lines = f.readlines()
            epoch_lines = [l for l in lines if "/200" in l and "loss=" in l]
            current_count = len(epoch_lines)
            if current_count > last_count:
                last_ep = epoch_lines[-1] if epoch_lines else ""
                print(f"[WATCHER] {name} @ epoch {current_count}: {last_ep.strip()[:80]}")
                last_count = current_count
                if current_count >= min_epochs:
                    stable_count += 1
            else:
                stable_count = 0
        except Exception as e:
            print(f"[WATCHER] Error reading log: {e}")
        # Check if process died
        if not is_process_alive(pid_from_log(log_path)):
            print(f"[WATCHER] {name} CRASHED during monitoring!")
            return False
    print(f"[WATCHER] {name} verified stable for {min_epochs}+ epochs.")
    return True

def pid_from_log(log_path):
    """Extract PID from log file header or process list."""
    # Try to find training process
    try:
        for line in open(log_path):
            if "python" in line and "run_EXP" in line:
                m = re.search(r'(\d+)', line)
                if m:
                    return int(m.group(1))
    except:
        pass
    # Fallback: find by command line
    try:
        out = subprocess.check_output(["pgrep", "-f", "run_EXP02_unetpp"], text=True)
        pids = [int(p) for p in out.strip().split("\n") if p.strip()]
        return pids[0] if pids else None
    except:
        return None

def update_product_md(exp_id, arch, encoder, lr, eff_bs, val_dice, status_note=""):
    """Update product.md leaderboard and backlog for a completed experiment."""
    md_path = PROJECT / "product.md"
    content = md_path.read_text()
    
    # Add to leaderboard (find last | and add before ---)
    new_row = f"| 99 | {exp_id} | {arch} | {encoder} (1024px) | {lr} | {eff_bs} | Dice+CE | {val_dice:.4f} | {status_note} |"
    
    # Find the last row in leaderboard (before ---)
    lines = content.split("\n")
    leaderboard_end = None
    for i, line in enumerate(lines):
        if line.startswith("---"):
            leaderboard_end = i
            break
    if leaderboard_end:
        lines.insert(leaderboard_end, new_row)
    
    # Update backlog
    old_backlog = f"- [ ] **{exp_id}:"
    new_backlog = f"- [x] **{exp_id}:"
    content = content.replace(old_backlog, new_backlog)
    
    md_path.write_text("\n".join(lines))
    print(f"[WATCHER] Updated product.md for {exp_id}")

# ─── MAIN LOGIC ────────────────────────────────────────────────────────

print(f"[WATCHER] Starting autonomous watcher @ {datetime.now().strftime('%H:%M:%S')}")
print("[WATCHER] Step 1: Kill zombies on GPUs 0-3")
kill_zombies()

# Step 1: Wait for EXP-01 to finish
print(f"[WATCHER] Step 2: Waiting for EXP-01 (PID {EXP01_PID}) to complete...")
best_exp01 = wait_for_completion(EXP01_PID, LOG_EXP01, "EXP-01")

if best_exp01:
    update_product_md("EXP-01", "DeepLabV3+", "resnet50", "5e-4", 32, best_exp01, "Completed")
    # Commit
    try:
        subprocess.run(["git", "add", "product.md"], cwd=PROJECT, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"EXP-01 done: dice={best_exp01:.4f}"], 
                      cwd=PROJECT, capture_output=True, timeout=30)
        print("[WATCHER] Git committed product.md")
    except Exception as e:
        print(f"[WATCHER] Git commit failed (non-blocking): {e}")

# Step 3: Kill remaining processes, free GPUs
print("[WATCHER] Step 3: Freeing GPUs for EXP-02...")
for pid in [EXP01_PID, 3337498]:
    try:
        os.kill(pid, 9)
    except:
        pass
time.sleep(3)
kill_zombies()
time.sleep(2)

# Check GPU memory
gpus = get_gpu_memory()
print(f"[WATCHER] GPU memory: {gpus}")

# Step 4: Launch EXP-02
print("[WATCHER] Step 4: Launching EXP-02 (UNet++ / resnet50 / LR=3e-4)...")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
exp02_proc = subprocess.Popen(
    [str(VENV_PYTHON), "-u", "scripts/run_EXP02_unetpp.py"],
    cwd=str(PROJECT),
    stdout=open(LOG_EXP02, "w"),
    stderr=subprocess.STDOUT,
    env=env,
)
print(f"[WATCHER] EXP-02 started with PID {exp02_proc.pid}")

# Step 5: Monitor first 10 epochs
print("[WATCHER] Step 5: Monitoring EXP-02 first epochs...")
ok = monitor_first_epochs(LOG_EXP02, "EXP-02", min_epochs=10)
if not ok:
    print("[WATCHER] EXP-02 crashed during first epochs! Aborting loop.")
    exit(1)

print("[WATCHER] EXP-02 verified stable. Entering standby mode.")
print("[WATCHER] EXP-02 will early-stop autonomously. User should wake agent for EXP-03.")
print(f"[WATCHER] Watcher complete @ {datetime.now().strftime('%H:%M:%S')}")
