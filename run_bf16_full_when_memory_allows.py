#!/usr/bin/env python3
"""Run the full-precision BF16 Dory oracle once the host has memory to spare.

Why this exists: the BF16 acceptance PASS measured for the KoboldCpp Dory port
was assembled from an interrupted run (positions 0-8 reconstructed from a log at
three decimals).  The replacement measurement needs about 8 GB of resident model
pages, and this host sits near its swap ceiling for days at a time, so "run it
now" either thrashes the machine or never happens.

This runner waits - for a bounded time - until two measured conditions hold,
makes sure a working llama-server exists, runs the oracle exactly once, and
leaves a result file that says what happened.

Gate (both must hold):
    swap used <= --max-swap-used-mb
    (free + inactive + speculative pages) * page size >= --min-available-gb

llama-server: the oracle resolves it as the sibling of --binary, i.e.
/private/tmp/kobold-dory/llamaserver, after resolving symlinks.  If it is
missing the runner restores it from a spare copy, or builds it.  Both
KoboldCpp targets are always remade and a signalled make DELETES its target,
which is how the binaries have twice vanished from that tree; so the build is
the only thing here that must never be interrupted, and a spare copy is kept.

Exit codes: 0 oracle ran; 3 the gate never opened before the deadline;
4 a prerequisite is missing, evidence already exists, or no working server could
be produced; 5 another runner holds the lock; 6 the oracle exceeded its
wall-clock budget and was killed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

ORACLE_DIR = Path("/private/tmp/kobold-dory")
SERVER = ORACLE_DIR / "llamaserver"
PYTHON = "/Users/dmitri.arkhipov/.pyenv/versions/MultiAgency/bin/python3"
BIN_DIR = Path("/private/tmp/kobold-dory-oracle-bin")
MODEL = Path("/private/tmp/dory-bf16-loops4.gguf")
ORACLE = Path("/Volumes/Widekeep/oracle/discriminating-prompt.json")
REFERENCE = Path("/Volumes/Widekeep/oracle/discriminating-prompt-top5.json")
EARLIER = Path("/private/tmp/kobold-dory-oracle-bf16-observed.json")
OBSERVED = Path("/private/tmp/kobold-dory-oracle-bf16-full-observed.json")
REPORT = Path("/tmp/kobold-dory-oracle-bf16-full-report.log")
# Only servers started from this port's own tree match; nothing else is touched.
SERVER_PATTERN = r"kobold-dory-oracle-bin/llama-server|kobold-dory/llamaserver|kobold-dory/llama-server"

EXIT_RAN = 0
EXIT_GATE_CLOSED = 3
EXIT_PREREQUISITE = 4
EXIT_LOCKED = 5
EXIT_TIMED_OUT = 6


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def read_swap_used_mb() -> float | None:
    text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=30).stdout
    found = re.search(r"used = ([\d.]+)M", text)
    return float(found.group(1)) if found else None


def read_available_gb() -> float | None:
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=30).stdout
    page_size = re.search(r"page size of (\d+) bytes", text)
    if not page_size:
        return None
    pages = 0
    for name in ("free", "inactive", "speculative"):
        found = re.search(rf"Pages {name}:\s+(\d+)\.", text)
        if not found:
            return None
        pages += int(found.group(1))
    return pages * int(page_size.group(1)) / 1e9


def readings() -> dict[str, float | None]:
    return {"swap_used_mb": read_swap_used_mb(), "available_gb": read_available_gb()}


def gate_open(current: dict[str, float | None], args: argparse.Namespace) -> bool:
    swap_used, available = current["swap_used_mb"], current["available_gb"]
    if swap_used is None or available is None:
        return False
    return swap_used <= args.max_swap_used_mb and available >= args.min_available_gb


def describe_gate(current: dict[str, float | None], args: argparse.Namespace) -> str:
    swap_used, available = current["swap_used_mb"], current["available_gb"]
    if swap_used is None or available is None:
        return f"unreadable (swap={swap_used}, available={available})"
    return (f"swap used {swap_used:.0f} MB, available {available:.1f} GB; "
            f"need swap used <= {args.max_swap_used_mb:.0f} MB and available >= {args.min_available_gb:.0f} GB")


def missing_prerequisites() -> list[str]:
    problems = []
    for path in (ORACLE_DIR / "scripts/dory_oracle_check.py", BIN_DIR / "llama-cli", MODEL, ORACLE, REFERENCE):
        if not path.is_file():
            problems.append(f"missing: {path}")
    if (BIN_DIR / "llama-cli").is_file() and not os.access(BIN_DIR / "llama-cli", os.X_OK):
        problems.append(f"not executable: {BIN_DIR / 'llama-cli'}")
    for path in (OBSERVED, REPORT):
        if path.exists():
            problems.append(f"refusing to overwrite existing evidence: {path}")
    return problems


def acquire_lock(lock_path: Path) -> bool:
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip() or 0)
        except ValueError:
            pid = 0
        if pid and pid != os.getpid():
            try:
                os.kill(pid, 0)
                return False
            except OSError:
                log(f"removing stale lock left by pid {pid}")
    lock_path.write_text(f"{os.getpid()}\n")
    return True


def kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError as exc:
        log(f"killpg({pgid}) failed: {exc}")
    found = subprocess.run(["pgrep", "-f", SERVER_PATTERN], capture_output=True, text=True, timeout=30).stdout.split()
    for raw in found:
        pid = int(raw)
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"killed leftover llama-server pid {pid}")
        except OSError as exc:
            log(f"kill({pid}) failed: {exc}")


def server_answers(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        return subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"{path.name} --version failed: {exc}")
        return False


def ensure_server(spare: Path, build_log: Path, build_timeout_s: float) -> tuple[bool, str, float]:
    """Return (ok, what was done, seconds spent building)."""
    if server_answers(SERVER):
        return True, "present", 0.0
    if spare.is_file():
        shutil.copy2(spare, SERVER)
        if server_answers(SERVER):
            return True, "restored from the spare copy", 0.0
        log("the spare copy does not run; rebuilding")
    log("building llamaserver - do NOT interrupt this make: a signalled make deletes its target")
    started = time.monotonic()
    with build_log.open("a", encoding="utf-8") as sink:
        sink.write(f"# build started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        sink.flush()
        process = subprocess.Popen(
            ["make", "-C", str(ORACLE_DIR), "llamaserver", "LLAMA_METAL=1", "-j4"],
            stdout=sink, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code: int | None = process.wait(timeout=build_timeout_s)
        except subprocess.TimeoutExpired:
            kill_group(process.pid)
            return False, f"build exceeded {build_timeout_s / 60:.0f} minutes and was stopped (log: {build_log})", \
                time.monotonic() - started
    spent = time.monotonic() - started
    if code != 0 or not server_answers(SERVER):
        return False, f"make exited {code} and left no working llamaserver (log: {build_log})", spent
    spare.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SERVER, spare)
    return True, "built", spent


def run_oracle(command: list[str], timeout_s: float) -> tuple[int | None, float]:
    started = time.monotonic()
    with REPORT.open("a", encoding="utf-8") as sink:
        sink.write(f"# full-precision BF16 run started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        sink.flush()
        process = subprocess.Popen(command, cwd=ORACLE_DIR, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code: int | None = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_leftovers(process.pid)
            code = None
    with REPORT.open("a", encoding="utf-8") as sink:
        sink.write(f"# ORACLE_EXIT={code}\n")
    return code, time.monotonic() - started


def kill_leftovers(pgid: int) -> None:
    kill_group(pgid)
    found = subprocess.run(["pgrep", "-f", SERVER_PATTERN], capture_output=True, text=True, timeout=30).stdout.split()
    for raw in found:
        pid = int(raw)
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"killed leftover llama-server pid {pid}")
        except OSError as exc:
            log(f"kill({pid}) failed: {exc}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top_map(entry: dict) -> dict[int, float]:
    return {int(token_id): float(value) for token_id, value in entry["top"]}


def compare_with_earlier() -> object:
    """Per position: how far the exact values sit from the earlier merged artifact."""
    if not EARLIER.is_file() or not OBSERVED.is_file():
        return {"error": f"need both {EARLIER} and {OBSERVED}"}
    fresh = json.loads(OBSERVED.read_text())["positions"]
    old = json.loads(EARLIER.read_text())["positions"]
    per_position = []
    for new_entry in sorted(fresh, key=lambda entry: int(entry["position"])):
        index = int(new_entry["position"])
        match = next((entry for entry in old if int(entry["position"]) == index), None)
        if match is None:
            per_position.append({"position": index, "earlier": "absent"})
            continue
        now, before = top_map(new_entry), top_map(match)
        deltas = [abs(now[token] - before[token]) for token in now.keys() & before.keys()]
        per_position.append({
            "position": index,
            "max_abs_delta_vs_earlier": round(max(deltas), 6) if deltas else None,
            "only_in_fresh": sorted(now.keys() - before.keys()),
            "only_in_earlier": sorted(before.keys() - now.keys()),
        })
    return {"per_position": per_position}


def oracle_command() -> list[str]:
    return [
        PYTHON, "scripts/dory_oracle_check.py",
        "--route", "logits",
        "--binary", str(BIN_DIR / "llama-cli"),
        "--model", str(MODEL),
        "--oracle", str(ORACLE),
        "--reference", str(REFERENCE),
        "--teacher-forced",
        "--accept-bf16-noise",
        "--n", "16",
        "--ctx", "64",
        "--ngl", "0",
        "--dump-observed", str(OBSERVED),
        "--keep-logs",
    ]


def write_result(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"result written to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full-precision BF16 Dory oracle when memory allows.")
    parser.add_argument("--max-wait-minutes", type=float, default=480.0, help="give up if the gate never opens")
    parser.add_argument("--poll-seconds", type=float, default=300.0, help="seconds between gate checks")
    parser.add_argument("--max-swap-used-mb", type=float, default=16000.0, help="gate: swap used must not exceed this")
    parser.add_argument("--min-available-gb", type=float, default=25.0, help="gate: free+inactive+speculative at least this")
    parser.add_argument("--run-timeout-minutes", type=float, default=75.0, help="wall-clock budget for the oracle")
    parser.add_argument("--build-timeout-minutes", type=float, default=120.0, help="wall-clock budget for the server build")
    parser.add_argument("--workdir", type=Path, default=Path("/private/tmp/dory-agent-fullprecision"))
    args = parser.parse_args(argv)

    result_path = args.workdir / "bf16-full-run-result.json"
    lock_path = args.workdir / "bf16-full-runner.lock"
    if not acquire_lock(lock_path):
        log("another runner holds the lock; exiting")
        return EXIT_LOCKED
    try:
        problems = missing_prerequisites()
        if problems:
            for problem in problems:
                log(problem)
            write_result(result_path, {"outcome": "prerequisite", "problems": problems})
            return EXIT_PREREQUISITE
        write_result(result_path, {"outcome": "waiting-for-gate", "since": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

        deadline = time.monotonic() + args.max_wait_minutes * 60
        current = readings()
        while True:
            while not gate_open(current, args):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log(f"gave up waiting: {describe_gate(current, args)}")
                    write_result(result_path, {"outcome": "gate-never-opened", "last_readings": current})
                    return EXIT_GATE_CLOSED
                log(f"gate closed: {describe_gate(current, args)}; {remaining / 60:.0f} min left")
                time.sleep(min(args.poll_seconds, remaining))
                current = readings()

            ok, status, spent = ensure_server(args.workdir / "spare" / "llamaserver",
                                              args.workdir / "server-build.log",
                                              args.build_timeout_minutes * 60)
            log(f"llamaserver: {status}" + (f" ({spent / 60:.1f} min)" if spent else ""))
            if not ok:
                write_result(result_path, {"outcome": "server-unavailable", "server": status, "gate_at_attempt": current})
                return EXIT_PREREQUISITE
            after_build = readings()
            if gate_open(after_build, args):
                current = after_build
                break
            if time.monotonic() >= deadline:
                log(f"gave up waiting: {describe_gate(after_build, args)}")
                write_result(result_path, {"outcome": "gate-never-opened", "last_readings": after_build, "server": status})
                return EXIT_GATE_CLOSED
            log("gate closed again while the server was being prepared; back to waiting")

        log(f"gate OPEN: {describe_gate(current, args)} - launching the oracle")
        code, duration = run_oracle(oracle_command(), args.run_timeout_minutes * 60)
        lines = REPORT.read_text(errors="replace").splitlines() if REPORT.is_file() else []
        outcome = "ran" if code is not None else "timed-out"
        write_result(result_path, {
            "outcome": outcome,
            "exit_code": code,
            "duration_s": round(duration, 1),
            "server": status,
            "gate_at_launch": current,
            "readings_after": readings(),
            "server_line": next((line.strip() for line in reversed(lines) if line.startswith("server :")), None),
            "verdict_lines": [line.strip() for line in lines if "VERDICT" in line][-6:],
            "observed": str(OBSERVED) if OBSERVED.is_file() else None,
            "observed_sha256": sha256_of(OBSERVED) if OBSERVED.is_file() else None,
            "compare_with_earlier_merged": compare_with_earlier() if OBSERVED.is_file() else None,
            "report": str(REPORT),
        })
        return EXIT_RAN if code is not None else EXIT_TIMED_OUT
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
