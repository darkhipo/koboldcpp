"""The noprefetch runtime-proof tool must import before anything else.

Its breakpoint command was built with a helper that was defined thirty lines
below the first use, so the module raised NameError at import and the proof had
never executed once. An import is the cheapest guard against that class.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "madvise_runtime_proof.py"


def _load():
    spec = importlib.util.spec_from_file_location("madvise_runtime_proof", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proof_tool_imports_and_builds_its_breakpoint_command() -> None:
    module = _load()

    assert module.BP_COMMAND.startswith("breakpoint command add 2 -o ")
    assert "x1" in module.BP_COMMAND  # the advice-3 call's length register
    assert module.MODES == ("mmap", "mmap,noprefetch")


def test_quote_helper_escapes_for_lldb() -> None:
    module = _load()

    assert module.q('say "hi" \\') == '"say \\"hi\\" \\\\"'
