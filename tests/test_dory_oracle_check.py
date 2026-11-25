from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "dory_oracle_check.py"
SPEC = importlib.util.spec_from_file_location("dory_oracle_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
oracle_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle_check)


def make_reference(tmp_path: Path, **overrides) -> dict:
    payload = {
        "quantity": "logprobs",
        "prompt": "What is 17 multiplied by 23? The answer is",
        "prompt_ids": [7493, 1395],
        "ids_loops4": [1032, 1051],
        "positions": [{"position": 0, "top": [[1032, -0.1], [2007, -5.0]]}],
    }
    payload.update(overrides)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return oracle_check.load_reference(path)


def test_reference_pairing_rejects_a_different_prompt(tmp_path: Path) -> None:
    reference = make_reference(tmp_path)

    reason = oracle_check.reference_oracle_mismatch(reference, "The capital of France is", [1784, 8961], [6993])

    assert "oracle prompt_ids" in reason
    assert "reference prompt" in reason


def test_reference_pairing_rejects_different_expected_ids(tmp_path: Path) -> None:
    reference = make_reference(tmp_path)

    reason = oracle_check.reference_oracle_mismatch(
        reference,
        "What is 17 multiplied by 23? The answer is",
        [7493, 1395],
        [1032, 9999],
    )

    assert "position 1" in reason
    assert "9999" in reason


def test_reference_pairing_accepts_matching_data(tmp_path: Path) -> None:
    reference = make_reference(tmp_path)

    reason = oracle_check.reference_oracle_mismatch(
        reference,
        "What is 17 multiplied by 23? The answer is",
        [7493, 1395],
        [1032, 1051],
    )

    assert reason == ""


def report_reference() -> dict:
    return {
        "path": "reference.json",
        "quantity": "logprobs",
        "positions": {0: [(1032, -0.1), (2007, -5.0)]},
        "provenance": "pytest",
        "self_observed": False,
    }


def test_disjoint_top_k_is_loud_data_not_a_crash(capsys: pytest.CaptureFixture[str]) -> None:
    rows = [{"position": 0, "actual": [(6993, -0.2), (2549, -4.0)], "why": ""}]

    rc = oracle_check.report_logits(
        rows,
        report_reference(),
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [[(6993, -0.2), (2549, -4.0)]],
        None,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING == 3
    assert "EMPTY COMPARISON" in captured.out
    assert "no shared token ids" in captured.out
    assert "Traceback" not in captured.err


def test_zero_rows_cannot_be_a_silent_pass(capsys: pytest.CaptureFixture[str]) -> None:
    rc = oracle_check.report_logits(
        [],
        report_reference(),
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [],
        None,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING == 3
    assert "no position produced a non-empty numerical comparison" in captured.out


def test_nonempty_matching_comparison_still_passes(capsys: pytest.CaptureFixture[str]) -> None:
    actual = [(1032, -0.1), (2007, -5.0)]
    rows = [{"position": 0, "actual": actual, "why": ""}]

    rc = oracle_check.report_logits(
        rows,
        report_reference(),
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [actual],
        None,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_MATCH == 0
    assert "VERDICT: PASS -- 1 positions" in captured.out


def test_bf16_mode_allows_order_noise_but_prints_strict_failure(capsys: pytest.CaptureFixture[str]) -> None:
    reference = report_reference()
    reference["provenance"] = "BF16 pytest"
    reference["positions"] = {0: [(1032, -0.1), (2007, -5.0), (1659, -5.5)]}
    actual = [(1032, -0.2), (1659, -5.6), (2007, -5.1)]
    rows = [{"position": 0, "actual": actual, "why": ""}]

    rc = oracle_check.report_logits(
        rows,
        reference,
        0.5,
        "pytest",
        Path("oracle.json"),
        "ids_loops4",
        [actual],
        None,
        allow_bf16_noise=True,
    )

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_MATCH == 0
    assert "VERDICT: FAIL" in captured.out
    assert "BF16 VERDICT: PASS" in captured.out


def test_bf16_mode_allows_a_near_tie_argmax_flip() -> None:
    expected = [(5675, -2.8758), (9032, -2.8844), (1032, -2.9129)]
    actual = [(9032, -2.880), (5675, -2.893), (1032, -2.991)]

    relaxed, _ = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logprobs", 0.5)

    assert relaxed


def test_bf16_mode_rejects_a_large_delta() -> None:
    expected = [(1032, -0.1), (2007, -5.0)]
    actual = [(1032, -0.7), (2007, -5.0)]

    relaxed, _ = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logprobs", 0.5)

    assert not relaxed


def test_bf16_mode_rejects_a_non_tie_argmax_flip() -> None:
    expected = [(1032, -0.1), (2007, -1.0)]
    actual = [(2007, -0.9), (1032, -0.2)]

    relaxed, _ = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logprobs", 0.5)

    assert not relaxed



def make_executable(path: Path, body: str = "#!/bin/sh\nexit 3\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_server_resolution_refuses_a_path_server_and_names_the_expected_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = tmp_path / "build" / "bin"
    binary = make_executable(build / "llama-cli")
    elsewhere = tmp_path / "elsewhere"
    path_server = make_executable(elsewhere / "llama-server")
    monkeypatch.setenv("PATH", str(elsewhere))

    got = oracle_check.resolve_server(binary)

    assert got["path"] is None
    assert str((build / "llama-server").resolve()) in got["reason"]
    assert str(path_server.resolve()) in got["reason"]
    assert "REFUSED" in got["reason"]
    assert "--server" in got["reason"]


def test_server_resolution_uses_the_sibling_of_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build = tmp_path / "build" / "bin"
    binary = make_executable(build / "llama-cli")
    sibling_server = make_executable(build / "llama-server")
    monkeypatch.setenv("PATH", str(tmp_path))

    got = oracle_check.resolve_server(binary)

    assert got["path"] == sibling_server.resolve()
    assert got["source"] == "sibling of --binary"
    assert got["reason"] == ""


def test_server_override_is_deliberate_and_logged_once_at_startup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    build = tmp_path / "build" / "bin"
    binary = make_executable(build / "llama-cli")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real gguf")
    other_server = make_executable(tmp_path / "other-build" / "llama-server")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"prompt": "hi", "prompt_ids": [1], "ids_loops4": [2]}), encoding="utf-8")

    rc = oracle_check.main([
        "--route", "logits", "--no-reference", "--boot-timeout", "2",
        "--binary", str(binary), "--model", str(model), "--oracle", str(oracle),
        "--server", str(other_server),
    ])

    captured = capsys.readouterr()
    logged = [line for line in captured.out.splitlines() if line.startswith("server : ")]
    assert logged == [f"server : {other_server.resolve()}   (--server override)"]
    assert rc == oracle_check.EXIT_CANNOT_RUN  # the stand-in exits at once: never healthy
