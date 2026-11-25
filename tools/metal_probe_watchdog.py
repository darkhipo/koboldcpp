#!/usr/bin/env python3
"""Run a command under a memory / wall-clock kill switch and record its footprint.

The command starts in its own session, so it leads a fresh process group. Every
--interval seconds the memory of its whole process tree (the group plus any
descendant that left it) is sampled with ps and appended to a CSV time series.
A sample above --max-gb gets the tree SIGKILLed (exit 3); a run longer than
--max-seconds gets exit 4. If the watchdog itself fails or is signalled, the
tree is killed first: a live command is never left unguarded.

Metrics (--metric):
    rss        Sum of ps RSS over the tree. Portable. Compressed or swapped
               pages are not counted.
    footprint  macOS only: sum of ri_phys_footprint (libproc), i.e. what the
               kernel bills each process for.

Exit codes: 0 completed, 3 killed for memory, 4 killed for time. Any other
value is the command's own exit status, so a command that exits 3 or 4 by
itself is indistinguishable from a kill -- check the CSV event column.

Only processes started here are ever signalled.
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time

EXIT_KILLED_MEMORY = 3
EXIT_KILLED_TIME = 4
CSV_HEADER = ("t_s", "n_procs", "rss_bytes", "footprint_bytes", "event")


def make_footprint_sampler():
    """Build a function mapping a pid to its physical footprint in bytes.

    Returns:
        The sampler, or None where libproc is unavailable (any non-macOS host).
    """
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    libproc_path = ctypes.util.find_library("proc")
    if not libproc_path:
        return None
    libproc = ctypes.CDLL(libproc_path)
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    buf = ctypes.create_string_buffer(256)  # rusage_info_v2, flavor 2

    def footprint(pid):
        # uuid[16] then uint64 fields: ri_resident_size is 7th (byte 64),
        # ri_phys_footprint 8th (byte 72).
        if libproc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:
            return 0
        return int.from_bytes(buf.raw[72:80], sys.byteorder)

    return footprint


def sample_tree(root, footprint):
    """Measure the process tree hanging off ``root``.

    Returns:
        (tree_pids, group_pids, rss_bytes, metric_bytes), or None if ``ps``
        failed. ``metric_bytes`` is the footprint sum when ``footprint`` is
        given, otherwise it equals ``rss_bytes``.
    """
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,rss="], capture_output=True, text=True,
                             timeout=10.0, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [r for r in (line.split() for line in out.splitlines()) if len(r) == 4]
    tree = {root} if any(int(r[0]) == root for r in rows) else set()
    group = {int(r[0]) for r in rows if int(r[2]) == root}
    while True:
        more = {int(r[0]) for r in rows if int(r[0]) not in tree and int(r[1]) in tree}
        if not more:
            break
        tree |= more
    rss = sum(int(r[3]) for r in rows if int(r[0]) in tree) * 1024
    metric = sum(footprint(p) for p in tree) if footprint else rss
    return tree, group, rss, metric


def kill_tree(proc, footprint):
    """SIGKILL the command's process group, then any descendant that left it."""
    snap = sample_tree(proc.pid, footprint)  # before the kill: parent links still intact
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    for pid in (snap[0] - snap[1]) if snap else ():
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _interrupt(signum, frame):
    # SIGTERM to the watchdog must not leave an unguarded command behind.
    raise KeyboardInterrupt


def main(argv=None):
    """Run the watchdog. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-gb", type=float, default=40.0, help="memory limit in GiB (default: 40)")
    parser.add_argument("--max-seconds", type=float, default=300.0, help="wall-clock limit in seconds (default: 300)")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between samples (default: 0.5)")
    parser.add_argument("--metric", choices=("rss", "footprint"), default="rss")
    parser.add_argument("--csv", required=True, help="output path for the time series")
    parser.add_argument("--log", help="file for the command's stdout+stderr (default: pass through)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command to run, after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; pass it after --")
    if min(args.max_gb, args.max_seconds, args.interval) <= 0:
        parser.error("limits and --interval must be > 0")
    footprint = None
    if args.metric == "footprint":
        footprint = make_footprint_sampler()
        if footprint is None:
            parser.error("--metric footprint needs libproc, which is macOS only")

    signal.signal(signal.SIGTERM, _interrupt)
    log = open(args.log, "w") if args.log else None
    sink = {"stdout": log, "stderr": subprocess.STDOUT} if log else {}
    try:
        proc = subprocess.Popen(command, start_new_session=True, **sink)
    except OSError as exc:
        parser.exit(1, f"metal_probe_watchdog: cannot start {command[0]!r}: {exc}\n")

    limit_bytes = args.max_gb * (1 << 30)
    start = time.monotonic()
    peak, reason = 0, None
    with open(args.csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        fh.flush()
        try:
            while True:
                snap = sample_tree(proc.pid, footprint)
                now = time.monotonic() - start
                if snap:
                    peak = max(peak, snap[3])
                    writer.writerow((round(now, 3), len(snap[0]), snap[2], snap[3], ""))
                    fh.flush()
                if proc.poll() is not None:
                    break
                if snap and snap[3] > limit_bytes:
                    reason = "memory"
                elif now > args.max_seconds:
                    reason = "time"
                if reason:
                    kill_tree(proc, footprint)
                    writer.writerow((round(time.monotonic() - start, 3), 0, snap[2] if snap else 0, peak,
                                     f"killed_{reason}"))
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            reason = "interrupt"
            kill_tree(proc, footprint)
        except Exception:
            kill_tree(proc, footprint)  # fail closed
            raise
        fh.flush()

    if reason == "memory":
        code = EXIT_KILLED_MEMORY
    elif reason == "time":
        code = EXIT_KILLED_TIME
    elif reason == "interrupt":
        code = 130
    elif proc.returncode is None:
        code = 1
    else:
        code = 128 - proc.returncode if proc.returncode < 0 else proc.returncode
    print(f"metal_probe_watchdog: {reason or 'completed'} after {time.monotonic() - start:.1f}s, "
          f"peak {peak / 2**20:.0f} MiB ({args.metric}), csv {args.csv}", file=sys.stderr)
    if log:
        log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
