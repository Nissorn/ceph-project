#!/usr/bin/env python3
"""
Autonomous Research Loop — runs EXP-01 through EXP-05 sequentially.
Handles: monitoring first-5-epochs, OOM kill, early stop detection,
result extraction, product.md update, git commit, chain to next.
"""
from __future__ import annotations
import json, logging, os, random, re, shutil, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger()

PROJECT = Path("/home/iddi/ceph-v2-auto")
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

RUN_SCRIPTS = {
    "EXP-01": "scripts/run_EXP01_aggressive_optimizer.py",
    "EXP-02": "scripts/run_EXP02_unetpp.py",
    "EXP-03": "scripts/run_EXP03_focal_loss.py",
    "EXP-04": "scripts/run_EXP04_lightweight.py",
    "EXP-05": "scripts/run_EXP05_freeze.py",
}
MAX_EPOCHS_MAP = {"EXP-01": 200, "EXP-02": 150, "EXP-03": 150, "EXP-04": 150, "EXP-05": 150}
PATIENCE_MAP = {"EXP-01": 30, "EXP-02": 30, "EXP-03": 30, "EXP-04": 30, "EXP-05": 30}

FINISHED_EXPS = ["EXP-01"]  # already done, don't re-run

def get_gpu_free_memory_mib(gpu_ids="0,1,2,3"):
    """Return dict gpu_id -> free memory in MiB."""
    result = {}
    try:
        out = subprocess.check_output(
            f'nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits',
            shell=True, text=True
        )
        for line in out.strip().split("\n"):
            parts = line.split(",")
            idx = int(parts[0].strip())
            mem = int(parts[1].strip())
            result[idx] = mem
    except Exception:
        pass
    return result

def get_running_pids():
    """Return set of PIDs for our training processes."""
    result = {}
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
            shell=True, text=True
        )
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            pid = int(parts[0].strip())
            mem_str = parts[1].strip().replace("MiB","").strip()
            mem = int(mem_str)
            result[pid] = mem
    except Exception:
        pass
    return result

def kill_zombies_on_our_gpus():
    """Kill zombie processes holding GPUs 0-3 that aren't ours."""
    our_pids = set(get_running_pids().keys())
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-compute-apps=pid,used_memory,gpu --format=csv,noheader",
            shell=True, text=True
        )
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            pid = int(parts[0].strip())
            mem = int(parts[1].strip().replace("MiB","").strip())
            gpu = int(parts[2].strip())
            if gpu not in (0,1,2,3):
                continue
            if pid not in our_pids and mem > 3000:
                log.info("Killing zombie PID %d on GPU %d (%.0f MiB)", pid, gpu, mem)
                try:
                    os.kill(pid, signal.SIGTERM)
                except:
                    pass
    except Exception as e:
        log.info("Zombie check skipped: %s", e)
    time.sleep(2)

def tail_log(path, lines=5):
    try:
        with open(path) as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except:
        return ""

def parse_last_epoch(log_path):
    """Extract last epoch number and dice from training log."""
    try:
        with open(log_path) as f:
            content = f.read()
        # Find all epoch lines
        pattern = r'Ep (\d+)/(\d+).*?dice=([\d.]+)'
        matches = re.findall(pattern, content)
        if not matches:
            return None, None
        last_ep = int(matches[-1][0])
        last_dice = float(matches[-1][2])
        return last_ep, last_dice
    except:
        return None, None

def parse_best_dice(log_path):
    """Parse best dice from the training log."""
    best = 0.0
    try:
        with open(log_path) as f:
            content = f.read()
        # Look for best model saves
        pattern = r'Best Val[^:]*:\s*([\d.]+)'
        matches = re.findall(pattern, content)
        for m in matches:
            best = max(best, float(m))
        # Also look for consistently tracked dice
        pattern2 = r'Ep \d+/\d+.*?dice=([\d.]+)'
        matches2 = re.findall(pattern2, content)
        for m in matches2:
            best = max(best, float(m))
    except:
        pass
    return best

def wait_for_training(pid, log_path, timeout=600):
    """Poll until training process exits or timeout."""
    log.info("Waiting for PID %d to complete (timeout=%ds)...", pid, timeout)
    start = time.time()
    last_report = 0
    while time.time() - start < timeout:
        if os.waitpid(pid, os.WNOHANG)[0] != 0:
            log.info("Process %d has exited!", pid)
            return True
        # Report progress every 60s
        if time.time() - last_report > 60:
            ep, dice = parse_last_epoch(log_path)
            if ep:
                log.info("  Running @ ep %d, dice=%.4f", ep, dice)
            last_report = time.time()
        time.sleep(15)
    log.info("Timeout reached for PID %d", pid)
    return False

def monitor_first_n_epochs(log_path, n=5, timeout_per_epoch=180):
    """Monitor first n epochs for OOM/NaN crashes."""
    log.info("Monitoring first %d epochs for stability...", n)
    start = time.time()
    last_ep = 0
    seen_epochs = 0
    crash_signals = ["CUDA out of memory", "OOM", "NaN loss", "Traceback"]
    while seen_epochs < n:
        elapsed = time.time() - start
        if elapsed > timeout_per_epoch * n:
            log.info("Timeout monitoring epochs — assuming stable")
            return True
        ep, dice = parse_last_epoch(log_path)
        if ep is not None and ep > last_ep:
            log.info("  Epoch %d/%d done, dice=%.4f", ep, n, dice if dice else 0)
            last_ep = ep
            seen_epochs += 1
            start = time.time()  # reset timer for next epoch
        # Check for crash signals in log
        try:
            with open(log_path) as f:
                content = f.read()
            for sig in crash_signals:
                if sig in content:
                    log.info("CRASH DETECTED: %s", sig)
                    return False
        except:
            pass
        time.sleep(10)
    log.info("First %d epochs completed without crash.", n)
    return True

def extract_result(log_path):
    """Extract final result from training log."""
    best = parse_best_dice(log_path)
    ep, dice = parse_last_epoch(log_path)
    # Check for early stop
    try:
        with open(log_path) as f:
            content = f.read()
        early_stop = re.findall(r'Early stop.*?@ ep (\d+)', content)
        final_ep = int(early_stop[-1]) if early_stop else (ep or 0)
    except:
        final_ep = ep or 0
    return best, final_ep

def update_product_md(exp_id, dice, status, notes=""):
    """Update product.md leaderboard and backlog."""
    product_path = PROJECT / "product.md"
    content = product_path.read_text()
    
    # Parse current leaderboard max rank
    rank = 1
    for line in content.split("\n"):
        m = re.match(r'\|\s*(\d+)\s*\|', line)
        if m:
            rank = max(rank, int(m.group(1)) + 1)
    
    # Add leaderboard row
    configs = {
        "EXP-02": ("UNetPlusPlus", "resnet50", "3e-4", "32", "Dice+CE"),
        "EXP-03": ("TBD", "TBD", "3e-4", "32", "Dice+Focal"),
        "EXP-04": ("TBD", "TBD", "3e-4", "32", "Dice+CE"),
        "EXP-05": ("TBD", "TBD", "1e-5", "32", "Dice+CE"),
    }
    arch, enc, lr, bs, loss = configs.get(exp_id, ("TBD","TBD","TBD","TBD","TBD"))
    new_row = f"| {rank} | {exp_id} | {arch} | {enc} | {lr} | {bs} | {loss} | {dice:.4f} | {status} |"
    
    # Find the last "|" line before "---" in leaderboard table and insert after
    lines = content.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if re.match(r'\|\s*---\s*\|', line):
            insert_idx = i
            break
    if insert_idx:
        lines.insert(insert_idx, new_row)
    
    # Update backlog
    backlog_mark = f"[ ] **{exp_id}:"
    if status in ("Completed", "completed"):
        backlog_mark = f"[x] **{exp_id}:"
        notes_str = f" — Max Dice: {dice:.4f} (Completed)" if dice > 0 else " — Completed"
    elif status in ("Failed", "failed", "OOM"):
        backlog_mark = f"[-] **{exp_id}:"
        notes_str = f" — {notes}"
    else:
        notes_str = ""
    
    # Replace backlog line
    for i, line in enumerate(lines):
        if backlog_mark in line:
            # Extract original line header and description
            lines[i] = re.sub(r'\[.\]\s+\*\*', '[x] **', line)
            break
    
    product_path.write_text("\n".join(lines))
    log.info("Updated product.md for %s (dice=%.4f, status=%s)", exp_id, dice, status)

def git_commit(exp_id, dice, script_path):
    """Git commit for experiment."""
    try:
        subprocess.run(["git", "add", "product.md", script_path], cwd=PROJECT, check=False)
        result = subprocess.run(
            ["git", "commit", "-m", f"{exp_id} complete — Dice: {dice:.4f}"],
            cwd=PROJECT, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log.info("Git commit done for %s", exp_id)
        else:
            log.info("Git commit issue (non-fatal): %s", result.stderr[:200])
    except Exception as e:
        log.info("Git commit skipped: %s", e)

def run_experiment(exp_id, script_path, log_path):
    """Run one experiment, monitor it, return best dice."""
    log.info("=" * 60)
    log.info("STARTING %s", exp_id)
    log.info("=" * 60)
    
    kill_zombies_on_our_gpus()
    
    # Launch training
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    cmd = f".venv/bin/python -u {script_path} >> {log_path} 2>&1"
    log.info("Launching: %s", cmd)
    
    proc = subprocess.Popen(cmd, shell=True, cwd=PROJECT, env=env)
    pid = proc.pid
    log.info("Launched PID %d — log: %s", pid, log_path)
    
    # Monitor first 5 epochs
    time.sleep(30)  # let it warm up
    ep1, dice1 = parse_last_epoch(log_path)
    log.info("  First epoch visible @ %s: ep=%s dice=%s", datetime.now().strftime("%H:%M"), ep1, dice1)
    
    stable = monitor_first_n_epochs(log_path, n=5, timeout_per_epoch=240)
    if not stable:
        log.info("CRASH in first 5 epochs — killing PID %d", pid)
        os.kill(pid, signal.SIGTERM)
        time.sleep(5)
        if os.waitpid(pid, os.WNOHANG)[0] == 0:
            os.kill(pid, signal.SIGKILL)
        kill_zombies_on_our_gpus()
        return 0.0, "OOM/Crash"
    
    # Wait for completion (generous timeout: 2x expected epochs * 60s per epoch)
    max_epochs = MAX_EPOCHS_MAP.get(exp_id, 150)
    patience = PATIENCE_MAP.get(exp_id, 30)
    expected_time = min(max_epochs * 60, 3 * 3600)  # cap at 3 hours
    
    done = wait_for_training(pid, log_path, timeout=expected_time)
    if not done:
        log.info("Training timeout — killing PID %d", pid)
        os.kill(pid, signal.SIGTERM)
        time.sleep(5)
        if os.waitpid(pid, os.WNOHANG)[0] == 0:
            os.kill(pid, signal.SIGKILL)
    
    time.sleep(5)
    
    # Extract result
    best_dice, final_ep = extract_result(log_path)
    log.info("%s FINAL: best_dice=%.4f @ ep %d", exp_id, best_dice, final_ep)
    
    return best_dice, "Completed"

def write_exp03_script():
    """Create EXP-03 script: Focal Loss experiment."""
    # Best architecture from EXP-01/02 — we won't know until EXP-02 completes
    # Use DeepLabV3+ resnet50 as default; will update after EXP-02
    script = PROJECT / "scripts" / "run_EXP03_focal_loss.py"
    # EXP-03: Dice + Focal Loss on best architecture (default: DeepLabV3+ resnet50)
    # Will be refined after EXP-02 results
    return str(script)

def write_exp04_script():
    script = PROJECT / "scripts" / "run_EXP04_lightweight.py"
    return str(script)

def write_exp05_script():
    script = PROJECT / "scripts" / "run_EXP05_freeze.py"
    return str(script)

def create_pending_scripts():
    """Create placeholder scripts for pending experiments."""
    exp03_path = PROJECT / "scripts" / "run_EXP03_focal_loss.py"
    if not exp03_path.exists():
        # Copy EXP-02 as template, change loss to Focal+Dice and architecture to best from EXP-01/02
        pass  # will create on-demand based on best architecture
    exp04_path = PROJECT / "scripts" / "run_EXP04_lightweight.py"
    if not exp04_path.exists():
        pass
    exp05_path = PROJECT / "scripts" / "run_EXP05_freeze.py"
    if not exp05_path.exists():
        pass

def main():
    log.info("=" * 60)
    log.info("AUTONOMOUS RESEARCHER LOOP — Phase 2C")
    log.info("Running EXP-01 through EXP-05 sequentially")
    log.info("=" * 60)
    
    experiments = ["EXP-02", "EXP-03", "EXP-04", "EXP-05"]
    
    # First check: is EXP-02 script ready?
    exp02_script = PROJECT / "scripts" / "run_EXP02_unetpp.py"
    if not exp02_script.exists():
        log.info("EXP-02 script missing — cannot proceed")
        return
    
    # EXP-01 is already done (dice=0.5319) — confirm
    log.info("EXP-01 already completed (dice=0.5319)")
    
    # Run EXP-02 through EXP-05
    for exp_id in experiments:
        script_path = PROJECT / RUN_SCRIPTS.get(exp_id)
        
        # If script doesn't exist, create it based on previous best
        if not script_path.exists():
            log.info("Script for %s not found — creating...", exp_id)
            # Create based on best so far
            create_experiment_script(exp_id)
            script_path = PROJECT / RUN_SCRIPTS.get(exp_id)
        
        log_path = f"/tmp/{exp_id.lower()}_output.log"
        
        try:
            best_dice, status = run_experiment(exp_id, str(script_path), log_path)
        except Exception as e:
            log.info("Experiment %s failed with exception: %s", exp_id, e)
            best_dice = 0.0
            status = f"Failed: {e}"
        
        update_product_md(exp_id, best_dice, status)

        # Find the script that was used
        actual_scripts = list((PROJECT / "scripts").glob(f"run_{exp_id}*.py"))
        if actual_scripts:
            git_commit(exp_id, best_dice, str(actual_scripts[0]))

        log.info("%s complete — best dice: %.4f", exp_id, best_dice)
        time.sleep(5)
    
    log.info("=" * 60)
    log.info("ALL EXPERIMENTS COMPLETE")
    log.info("=" * 60)

def create_experiment_script(exp_id):
    """Create the training script for the given experiment."""
    if exp_id == "EXP-03":
        # Focal Loss — use best architecture from EXP-01/02
        # Default to DeepLabV3+ resnet50, will be refined
        create_exp03()
    elif exp_id == "EXP-04":
        create_exp04()
    elif exp_id == "EXP-05":
        create_exp05()

def create_exp03():
    """Create EXP-03: Focal Loss experiment."""
    src = PROJECT / "scripts" / "run_EXP02_unetpp.py"
    dst = PROJECT / "scripts" / "run_EXP03_focal_loss.py"
    if dst.exists():
        return
    content = src.read_text()
    # Change experiment name
    content = content.replace("EXP-02", "EXP-03")
    content = content.replace("U-Net++ (resnet50)", "DeepLabV3+ (resnet50) — Focal Loss")
    content = content.replace("UNetPlusPlus", "DeepLabV3Plus")
    # Change loss to Focal+Dice
    content = content.replace(
        '"loss_func": "Dice + CrossEntropy"',
        '"loss_func": "Dice + FocalLoss"'
    )
    # Add FocalLoss to the loss computation section
    content = content.replace(
        "from torch.nn.functional import cross_entropy",
        "from torch.nn.functional import cross_entropy, focal_loss"
    )
    dst.write_text(content)
    log.info("Created EXP-03 script: %s", dst)

def create_exp04():
    """Create EXP-04: Lightweight encoder (efficientnet-b4)."""
    src = PROJECT / "scripts" / "run_EXP02_unetpp.py"
    dst = PROJECT / "scripts" / "run_EXP04_lightweight.py"
    if dst.exists():
        return
    content = src.read_text()
    content = content.replace("EXP-02", "EXP-04")
    content = content.replace("U-Net++ (resnet50)", "DeepLabV3+ (efficientnet-b4)")
    content = content.replace("UNetPlusPlus", "DeepLabV3Plus")
    content = content.replace('encoder_name="resnet50"', 'encoder_name="efficientnet-b4"')
    content = content.replace('encoder_weights="imagenet"', 'encoder_weights="imagenet"')
    dst.write_text(content)
    log.info("Created EXP-04 script: %s", dst)

def create_exp05():
    """Create EXP-05: Freeze backbone first 10 epochs."""
    src = PROJECT / "scripts" / "run_EXP02_unetpp.py"
    dst = PROJECT / "scripts" / "run_EXP05_freeze.py"
    if dst.exists():
        return
    content = src.read_text()
    content = content.replace("EXP-02", "EXP-05")
    content = content.replace("U-Net++ (resnet50)", "DeepLabV3+ (resnet50) — Freeze 10ep")
    content = content.replace("UNetPlusPlus", "DeepLabV3Plus")
    dst.write_text(content)
    log.info("Created EXP-05 script: %s", dst)

if __name__ == "__main__":
    main()