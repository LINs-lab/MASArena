#!/usr/bin/env python3
"""Monitor evoagent log files — alerts on quota/Jina/deadlock/process death, reports progress."""

import os, sys, time, re, signal
from pathlib import Path
from datetime import datetime

LOG_DIR       = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/evoagent_gaia")
PIDS          = [int(p) for p in sys.argv[2:] if p.isdigit()]
STALE_MINUTES = 10
POLL_SECS     = 15
DEDUP_SECS    = 300

PATTERNS = [
    ("💥 API_QUOTA",   re.compile(r"insufficient_quota|You exceeded your current quota", re.I)),
    ("🔴 JINA_ERROR",  re.compile(r"jina\.ai.*(?:4\d\d|error)|(?:4\d\d).*jina\.ai", re.I)),
    ("🔴 JINA_QUOTA",  re.compile(r"jina.*(quota|limit exceeded|insufficient)", re.I)),
    ("💀 OOM",         re.compile(r"MemoryError|Out of memory|Killed", re.I)),
    ("🐛 EXCEPTION",   re.compile(r"Traceback \(most recent call last\)")),
]

IGNORE_AFTER_TRACEBACK = re.compile(
    r"get_child_watcher.*not activated|subprocess support is not installed|Event loop is closed",
    re.I
)

# Lines that look like OOM/Killed but are benign (e.g. agent init print)
OOM_FALSE_POSITIVE = re.compile(r"System Prompt", re.I)

LIVE_RESULT_RE = re.compile(
    r"LIVE_RESULT progress=(\d+)/(\d+) correct=(\d+) accuracy=([\d.]+%)"
)

RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ts():
    return datetime.now().strftime("%H:%M:%S")

def alert(tag, msg, color=RED):
    print(f"{color}{BOLD}[{ts()}] {tag}{RESET}  {msg}", flush=True)

def find_logs():
    logs = {}
    if not LOG_DIR.exists():
        return logs
    for f in sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        for model in ("qwen3-32b", "glm-5", "qwen35b"):
            if model in f.name and model not in logs:
                logs[model] = f
    return logs

class LogWatcher:
    def __init__(self, path, model):
        self.path  = path
        self.model = model
        self.pos   = path.stat().st_size
        self.last_dedup: dict[str, float] = {}
        self.last_progress = ""

    def _can_alert(self, key):
        now = time.time()
        if now - self.last_dedup.get(key, 0) > DEDUP_SECS:
            self.last_dedup[key] = now
            return True
        return False

    def poll(self):
        try:
            st = self.path.stat()
        except FileNotFoundError:
            alert("❓ FILE_GONE", str(self.path))
            return

        stale_secs = time.time() - st.st_mtime
        if stale_secs > STALE_MINUTES * 60:
            if self._can_alert("STALE:" + self.path.name):
                mins = int(stale_secs // 60)
                alert("🔒 DEADLOCK?",
                      f"{self.model} — no new lines for {mins} min", YELLOW)

        if st.st_size <= self.pos:
            return

        with open(self.path, errors="replace") as fh:
            fh.seek(self.pos)
            new_text = fh.read()
            self.pos = fh.tell()

        lines = new_text.splitlines()

        # progress: print last LIVE_RESULT in this chunk
        for line in reversed(lines):
            m = LIVE_RESULT_RE.search(line)
            if m:
                done, total, correct, acc = m.group(1), m.group(2), m.group(3), m.group(4)
                progress_str = f"{done}/{total}"
                if progress_str != self.last_progress:
                    self.last_progress = progress_str
                    print(f"{GREEN}[{ts()}] 📊 {self.model:<12}{RESET}"
                          f"  {done}/{total}  ✅ {correct}  acc {acc}", flush=True)
                break

        # error patterns
        for i, line in enumerate(lines):
            for tag, rx in PATTERNS:
                if rx.search(line):
                    if tag == "🐛 EXCEPTION":
                        window = "\n".join(lines[i:i+25])
                        if IGNORE_AFTER_TRACEBACK.search(window):
                            break
                    if tag == "💀 OOM" and OOM_FALSE_POSITIVE.search(line):
                        break
                    if self._can_alert(tag + self.path.name):
                        alert(tag, f"[{self.model}] {line.strip()[-160:]}")
                    break

def pid_alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError:    return True

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

print(f"{CYAN}{BOLD}Monitor started — {LOG_DIR} | PIDs: {PIDS or 'auto-detect'}{RESET}")
print(f"Alerts: quota, Jina, >{STALE_MINUTES}min stale, process death  |  Progress: every new step  (Ctrl+C)\n")

watchers: dict[str, LogWatcher] = {}
dead_pids: set[int] = set()

while True:
    for model, path in find_logs().items():
        if model not in watchers or watchers[model].path != path:
            print(f"{CYAN}[{ts()}] Watching  {path.name}{RESET}", flush=True)
            watchers[model] = LogWatcher(path, model)

    for w in watchers.values():
        w.poll()

    for pid in PIDS:
        if pid not in dead_pids and not pid_alive(pid):
            dead_pids.add(pid)
            alert("💀 PROCESS_DIED", f"PID {pid} is gone")

    time.sleep(POLL_SECS)
