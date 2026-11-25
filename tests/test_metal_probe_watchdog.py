#!/usr/bin/env python3
"""Tests for tools/metal_probe_watchdog.py.

A dummy child allocates memory gradually so the watchdog's memory kill can be
exercised with a 0.2 GiB limit. Peak memory of the whole module stays well
under 300 MB. Run from the repository root: pytest tests/test_metal_probe_watchdog.py
"""

import csv
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "metal_probe_watchdog.py"
GIB = 1 << 30
GRANDCHILD_MARKER = "metal_probe_watchdog_test_grandchild"
# Spawns a grandchild, then grows its own memory by step_mb every pause_s
# seconds. Every page is touched, otherwise macOS leaves calloc'd memory
# non-resident and RSS never rises. argv[3], when given, is a marker that
# ends up in the grandchild's command line so tests can find it with ps.
ALLOC_CHILD = """
import subprocess, sys, time
step_mb, pause_s = int(sys.argv[1]), float(sys.argv[2])
marker = sys.argv[3] if len(sys.argv) > 3 else ""
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)  # " + marker])
print("GRANDCHILD", grandchild.pid, flush=True)
held = []
while True:
    chunk = bytearray(step_mb << 20)
    view = memoryview(chunk)
    for offset in range(0, len(view), 4096):
        view[offset] = 1
    held.append(chunk)
    time.sleep(pause_s)
"""


def find_family():
    """Return [(pid, command)] for every live process started from ALLOC_CHILD."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    family = []
    for line in out.splitlines():
        pid, _, command = line.strip().partition(" ")
        if "alloc_child.py" in command or GRANDCHILD_MARKER in command:
            family.append((int(pid), command))
    return family


def wait_until_gone(timeout=5):
    """Wait for the dummy family to disappear; returns whatever is still alive."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and find_family():
        time.sleep(0.1)
    return find_family()


@pytest.fixture(autouse=True)
def no_family_left_behind():
    # A failed test must not leak an unbounded allocator onto the host.
    yield
    for pid, _ in find_family():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.fixture
def alloc_child(tmp_path):
    path = tmp_path / "alloc_child.py"
    path.write_text(ALLOC_CHILD)
    return path


def run_watchdog(tmp_path, command, *watchdog_args):
    csv_path = tmp_path / "series.csv"
    log_path = tmp_path / "child.log"
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--csv", str(csv_path), "--log", str(log_path), *watchdog_args, "--", *command],
        capture_output=True, text=True, timeout=90,
    )
    return proc, csv_path, log_path


def read_rows(csv_path):
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def grandchild_pid(log_path):
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "GRANDCHILD":
            return int(parts[1])
    return None


def is_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return not out or out.startswith("Z")


def test_alloc_child_source_compiles():
    # A dummy child that dies at startup makes every kill test below pass vacuously.
    compile(ALLOC_CHILD, "alloc_child.py", "exec")


def test_memory_kill_reaches_whole_process_group(tmp_path, alloc_child):
    proc, csv_path, log_path = run_watchdog(
        tmp_path, [sys.executable, str(alloc_child), "2", "0.03", GRANDCHILD_MARKER],
        "--max-gb", "0.2", "--interval", "0.05", "--max-seconds", "60",
    )

    assert proc.returncode == 3, proc.stderr
    rows = read_rows(csv_path)
    assert rows and rows[-1]["event"] == "killed_memory"

    samples = [int(r["rss_bytes"]) for r in rows if not r["event"]]
    assert samples, "no samples were recorded before the kill"
    assert samples[-1] > 0.2 * GIB, "kill fired below the limit"
    assert max(samples) < 300 * 2**20, "test overshoot exceeded its memory budget"

    pid = grandchild_pid(log_path)
    assert pid, "child did not report its grandchild"
    deadline = time.monotonic() + 5
    while not is_gone(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert is_gone(pid), f"grandchild {pid} survived the group kill"


def test_time_kill(tmp_path, alloc_child):
    proc, csv_path, log_path = run_watchdog(
        tmp_path, [sys.executable, str(alloc_child), "1", "1", GRANDCHILD_MARKER],
        "--max-gb", "10", "--interval", "0.1", "--max-seconds", "1",
    )

    assert proc.returncode == 4, proc.stderr
    assert grandchild_pid(log_path), "child was not alive to be killed"
    rows = read_rows(csv_path)
    assert rows[-1]["event"] == "killed_time"
    assert float(rows[-1]["t_s"]) >= 1.0
    assert not wait_until_gone(), "command survived the time kill"


def test_sigterm_to_watchdog_does_not_orphan_the_command(tmp_path, alloc_child):
    csv_path, log_path = tmp_path / "series.csv", tmp_path / "child.log"
    watchdog = subprocess.Popen(
        [sys.executable, str(TOOL), "--csv", str(csv_path), "--log", str(log_path), "--interval", "0.05",
         "--max-seconds", "120", "--", sys.executable, str(alloc_child), "1", "1", GRANDCHILD_MARKER],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 20
        pid = None
        while pid is None and time.monotonic() < deadline:
            pid = grandchild_pid(log_path)
            time.sleep(0.05)
        assert pid is not None, "child and grandchild never got far enough to be guarded"
        assert watchdog.poll() is None, "watchdog exited before it was signalled"

        watchdog.send_signal(signal.SIGTERM)
        assert watchdog.wait(timeout=30) == 130
    finally:
        if watchdog.poll() is None:
            watchdog.kill()
            watchdog.wait(timeout=30)

    assert not wait_until_gone(), "command outlived the watchdog"


def test_completed_command_passes_through(tmp_path):
    proc, csv_path, log_path = run_watchdog(
        tmp_path, [sys.executable, "-c", "print('done')"],
        "--max-gb", "10", "--max-seconds", "30",
    )

    assert proc.returncode == 0, proc.stderr
    rows = read_rows(csv_path)
    assert rows and all(not r["event"] for r in rows)
    assert rows[0]["n_procs"] == "1", "the command was not sampled while alive"
    assert "done" in log_path.read_text()


def test_command_exit_code_is_propagated(tmp_path):
    proc, _, _ = run_watchdog(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"], "--max-gb", "10")

    assert proc.returncode == 7, proc.stderr


def test_footprint_sampler_covers_this_process():
    spec = importlib.util.spec_from_file_location("metal_probe_watchdog", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.make_footprint_sampler()

    if sys.platform != "darwin":
        assert sampler is None
        return
    assert sampler(os.getpid()) > 0
    assert sampler(2147483646) == 0, "an impossible pid must read as 0, not raise"
    own = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    assert sampler(os.getpid()) < int(own) * 1024 + 2**30, "footprint drifted impossibly far from RSS"


def test_footprint_metric_can_trigger_the_kill(tmp_path, alloc_child):
    if sys.platform != "darwin":
        pytest.skip("libproc is macOS only")
    proc, csv_path, log_path = run_watchdog(
        tmp_path, [sys.executable, str(alloc_child), "2", "0.03", GRANDCHILD_MARKER],
        "--max-gb", "0.2", "--interval", "0.05", "--max-seconds", "60", "--metric", "footprint",
    )

    assert proc.returncode == 3, proc.stderr
    assert grandchild_pid(log_path), "child was not alive to be killed"
    rows = read_rows(csv_path)
    assert rows[-1]["event"] == "killed_memory"
    assert int(rows[-2]["footprint_bytes"]) > 0.2 * GIB
    assert not wait_until_gone(), "command survived the memory kill"


def test_help_does_not_start_anything(tmp_path):
    csv_path = tmp_path / "unused.csv"
    proc = subprocess.run([sys.executable, str(TOOL), "--help", "--csv", str(csv_path)], capture_output=True, text=True)

    assert proc.returncode == 0
    assert "Exit codes" in proc.stdout
    assert not csv_path.exists()
