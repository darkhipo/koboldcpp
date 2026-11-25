#!/usr/bin/env python3
"""Count MADV_WILLNEED calls (madvise advice 3) made while a model is loaded.

Why lldb and not a DYLD interposer: on macOS 27.2.1 dyld binds a plain clang-built
binary's imports straight to libsystem_kernel even with DYLD_INSERT_LIBRARIES set
(DYLD_PRINT_BINDINGS shows `<main/bind#221> -> <libsystem_kernel.dylib/_posix_madvise>`),
so libmadvise_trace.dylib stays silent for llama.cpp tools. A debugger breakpoint sees the
real calls. The loader issues POSIX_MADV_WILLNEED through posix_madvise() only
(src/llama-mmap.cpp); posix_madvise is an alias of madvise in libsystem_kernel, so one
breakpoint on `madvise` covers both.

Prerequisite: make main llamaserver -j4   (Xcode command line tools provide lldb)

    python3 tools/madvise_runtime_proof.py --model /path/model.gguf
    python3 tools/madvise_runtime_proof.py --model /path/model.gguf --server --port 29611

Default: ./main (llama-completion) loads the model through the same
common_init_from_params() -> llama_model_loader::init_mappings() chain as llamaserver and
exits by itself. --server: ./llamaserver under an lldb script, killed once the log says
"model loaded". Each run is skipped unless swap used <= --max-swap-mb.
Exit codes: 0 expectation met, 1 not met or load failed, 3 memory gate closed.
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODES = ("mmap", "mmap,noprefetch")
PORTS_NEVER_USE = (8095, 19102)  # other minds' model servers on this host
LLDB = "/usr/bin/lldb"
def q(text):
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


WILLNEED_LENS = []  # filled by the lldb breakpoint callback in server mode
BP_COMMAND = "breakpoint command add 2 -o " + q(
    "script print('WILLNEED len=%d' % lldb.frame.FindRegister('x1').GetValueAsUnsigned())"
)
LOAD_ARGS = ["-ngl", "0", "-c", "512", "--no-warmup"]


def swap_used_mb():
    sysctl = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True)
    return float(re.search(r"used = ([\d.]+)M", sysctl.stdout).group(1))


def free_port():
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    for _ in range(50):
        port = random.randint(20000, 60000)
        if port in PORTS_NEVER_USE:
            continue
        busy = subprocess.run([lsof, "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN"], capture_output=True, text=True)
        if not busy.stdout.strip():
            return port
    raise SystemExit("no free TCP port found")


def clean_env():
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "USER", "LANG") if k in os.environ}
    env.setdefault("PATH", "/usr/bin:/bin")
    return env



def run_lldb(commands, outdir, tag, timeout):
    """Run lldb in batch mode; returns its output, also saved to <outdir>/lldb-<tag>.log."""
    cmd = [LLDB, "--batch"] + [part for command in commands for part in ("-o", command)]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=clean_env(),
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # lldb, debugserver and the process it launched
        out, _ = proc.communicate()
        out += "\nTIMEOUT after %ss" % timeout
    with open(os.path.join(outdir, "lldb-%s.log" % tag), "w") as fh:
        fh.write(out)
    return out


def error_lines(out):
    return [line.strip()[:200] for line in out.splitlines() if "error" in line.lower()][:4]


def run_self_exiting(tool, model, mode, outdir, timeout):
    """./main: load the model, generate one token, exit. lldb returns when the process exits."""
    tag = mode.replace(",", "-")
    args = ["-m", model, "--load-mode", mode] + LOAD_ARGS + ["-n", "1", "-p", "The"]
    commands = ["target create " + q(tool), "breakpoint set --name madvise"]
    commands += ['breakpoint set --name madvise --condition "$x2 == 3"', BP_COMMAND]
    commands += ["breakpoint modify --auto-continue true 1 2", "process launch -- " + " ".join(q(a) for a in args)]
    commands += ["breakpoint list 1 2"]
    out = run_lldb(commands, outdir, tag, timeout + 60)
    counts = {
        int(m.group(1)): int(m.group(2)) for m in re.finditer(r"^(\d): name = 'madvise'.*?hit count = (\d+)", out, re.M)
    }
    status = re.search(r"exited with status = (-?\d+)", out)
    return {
        "all_madvise_calls": counts.get(1),
        "willneed_calls": counts.get(2),
        "willneed_lens": [int(x) for x in re.findall(r"WILLNEED len=(\d+)", out)],
        "exit_status": int(status.group(1)) if status else None,
        "errors": error_lines(out),
    }


def run_server(tool, model, mode, port, outdir, timeout):
    """./llamaserver: lldb's embedded Python (lldb_session) launches it and kills it after the load."""
    tag = mode.replace(",", "-")
    log = os.path.join(outdir, "server-%s.log" % tag)
    if os.path.exists(log):
        os.remove(log)
    spec = {
        "args": [tool, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--load-mode", mode] + LOAD_ARGS,
        "log": log,
        "timeout": timeout,
        "settle": 3,
    }
    script = (
        "import importlib.util as u, json, os, sys; "
        "os.environ['MADVISE_PROOF_SPEC'] = json.dumps(%r); "
        "s = u.spec_from_file_location('madvise_runtime_proof', %r); "
        "m = u.module_from_spec(s); sys.modules['madvise_runtime_proof'] = m; s.loader.exec_module(m); "
        "m.WILLNEED_LENS.clear(); m.lldb_session()" % (json.dumps(spec), os.path.abspath(__file__))
    )
    out = run_lldb(["target create " + q(tool), "script " + script], outdir, tag, timeout + 90)
    found = re.search(r"MADVISE_PROOF_RESULT (\{.*\})", out)
    if not found:
        return {
            "all_madvise_calls": None,
            "willneed_calls": None,
            "willneed_lens": [],
            "exit_status": None,
            "outcome": "no-result",
            "errors": error_lines(out),
        }
    result = json.loads(found.group(1))
    result["errors"] = error_lines(out)
    return result


def lldb_session():
    """Body of server mode, executed inside lldb's Python by the -o 'script ...' command."""
    import lldb

    spec = json.loads(os.environ["MADVISE_PROOF_SPEC"])
    target = lldb.debugger.GetSelectedTarget()
    every = target.BreakpointCreateByName("madvise")
    every.SetAutoContinue(True)
    willneed = target.BreakpointCreateByName("madvise")
    willneed.SetCondition("$x2 == 3")
    willneed.SetScriptCallbackBody(
        "import madvise_runtime_proof as m; "
        "m.WILLNEED_LENS.append(frame.FindRegister('x1').GetValueAsUnsigned()); return False"
    )
    info = lldb.SBLaunchInfo(None)
    info.SetArguments(spec["args"], False)
    info.AddOpenFileAction(1, spec["log"], False, True)
    info.AddOpenFileAction(2, spec["log"], False, True)
    info.SetEnvironmentEntries(["MADVISE_PROOF_SPEC=" + os.environ["MADVISE_PROOF_SPEC"]], True)
    error = lldb.SBError()
    result = {
        "outcome": "launch-failed",
        "all_madvise_calls": 0,
        "willneed_calls": 0,
        "willneed_lens": [],
        "exit_status": None,
    }
    process = target.Launch(info, error)
    if not process.IsValid():
        result["launch_error"] = error.GetCString()
    else:
        lldb.debugger.SetAsync(True)
        listener, event = lldb.debugger.GetListener(), lldb.SBEvent()
        deadline = time.time() + spec["timeout"]
        while time.time() < deadline:
            listener.WaitForEvent(1, event)  # keeps lldb's view of the process current
            if process.GetState() == lldb.eStateExited:
                result["outcome"], result["exit_status"] = "exited", process.GetExitStatus()
                break
            log_text = open(spec["log"], errors="replace").read() if os.path.exists(spec["log"]) else ""
            if "model loaded" in log_text:
                time.sleep(spec["settle"])  # let advice calls made right after the load land
                result["outcome"] = "loaded"
                break
        else:
            result["outcome"] = "timeout"
        process.Kill()  # only the process launched above
    result["all_madvise_calls"] = every.GetHitCount()
    result["willneed_calls"] = willneed.GetHitCount()
    result["willneed_lens"] = list(WILLNEED_LENS)
    print("MADVISE_PROOF_RESULT " + json.dumps(result))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GGUF to load (read-only)")
    ap.add_argument("--server", action="store_true", help="use ./llamaserver instead of ./main")
    ap.add_argument("--port", type=int, help="listen port for --server (default: a free one, checked with lsof)")
    ap.add_argument("--max-swap-mb", type=float, default=18000.0, help="skip every run if swap used is above this")
    ap.add_argument("--timeout", type=int, default=180, help="seconds to wait for one load")
    ap.add_argument("--outdir", default=os.path.join(ROOT, ".build", "madvise-proof"))
    args = ap.parse_args()

    tool = os.path.join(ROOT, "llamaserver" if args.server else "main")
    model = os.path.abspath(args.model)
    missing = [p for p in (tool, LLDB, model) if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing file: %s" % ", ".join(missing))
    os.makedirs(args.outdir, exist_ok=True)
    port = args.port or free_port()
    ready = "python3 %s --model %s%s" % (
        os.path.abspath(__file__),
        model,
        " --server --port %d" % port if args.server else "",
    )
    rows = []
    for mode in MODES:
        swap = swap_used_mb()
        if swap > args.max_swap_mb:
            print("MEMORY GATE CLOSED: swap used %.0f MB > %.0f MB; nothing was launched." % (swap, args.max_swap_mb))
            print("READY-TO-RUN: " + ready)
            for row in rows:
                print(
                    "already done: --load-mode %-16s willneed=%s all=%s"
                    % (row["mode"], row["willneed_calls"], row["all_madvise_calls"])
                )
            return 3
        started = time.time()
        if args.server:
            res = run_server(tool, model, mode, port, args.outdir, args.timeout)
        else:
            res = run_self_exiting(tool, model, mode, args.outdir, args.timeout)
        res.update(mode=mode, tool=os.path.basename(tool), swap_used_mb=swap, seconds=round(time.time() - started, 1))
        rows.append(res)
        print(
            "--load-mode %-16s willneed=%s all_madvise=%s lens=%s exit=%s outcome=%s %.1fs (swap %.0f MB)"
            % (
                mode,
                res["willneed_calls"],
                res["all_madvise_calls"],
                res["willneed_lens"],
                res["exit_status"],
                res.get("outcome", "ran"),
                res["seconds"],
                swap,
            )
        )
        for err in res["errors"]:
            print("   ", err)
        time.sleep(1)
    with open(os.path.join(args.outdir, "madvise-proof-result.json"), "w") as fh:
        json.dump({"model": model, "port": port, "runs": rows}, fh, indent=2)
    noprefetch_ok = all(r["willneed_calls"] == 0 for r in rows if r["mode"].endswith("noprefetch"))
    prefetch_ok = all(r["willneed_calls"] >= 1 for r in rows if not r["mode"].endswith("noprefetch"))
    finished = all(r["exit_status"] == 0 or r.get("outcome") == "loaded" for r in rows)
    ok = len(rows) == len(MODES) and noprefetch_ok and prefetch_ok and finished
    print("RESULT:", "PASS" if ok else "FAIL", "(mmap must issue >=1 WILLNEED, mmap,noprefetch must issue 0)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
