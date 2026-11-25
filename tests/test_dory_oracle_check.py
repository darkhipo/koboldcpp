from __future__ import annotations

import http.server
import importlib.util
import json
import math
import threading
import types
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
        "--route", "logits", "--no-reference", "--boot-timeout", "2", "--checkpoint", "",
        "--binary", str(binary), "--model", str(model), "--oracle", str(oracle),
        "--server", str(other_server),
    ])

    captured = capsys.readouterr()
    logged = [line for line in captured.out.splitlines() if line.startswith("server : ")]
    assert logged == [f"server : {other_server.resolve()}   (--server override)"]
    assert rc == oracle_check.EXIT_CANNOT_RUN  # the stand-in exits at once: never healthy



# ---------------------------------------------------------------------------
# checkpoint / resume  (register 2125613720)
# ---------------------------------------------------------------------------

PROMPT_IDS = [7493, 1395, 1032]
EXPECTED = [1032, 1051, 1057, 1049]
# 17 significant digits: anything short of an exact float round-trip shows up
FAKE_TOPS = [
    [(1032, -0.12345678901234567), (2007, -5.0179495811462401), (1659, -5.4067463874816895)],
    [(1051, -0.015484467148780823), (1049, -4.4138092994689941)],
    [(1057, -0.0010000000000000000218), (1049, -9.4299999999999997)],
    [(1049, -0.00099999999999999), (1051, -8.1280000000000001)],
]


class FakeLlamaServer:
    """In-process stand-in for llama-server: /health, /tokenize, 1-token /completion.

    It listens on a real local port, so the script's HTTP path runs unchanged.
    The process ServerSession launches is only a sleeper, so a run can also fail
    before anything is collected.
    """

    def __init__(self, prompt_ids: list[int], tops: list[list[tuple[int, float]]]) -> None:
        self.prompt_ids = list(prompt_ids)
        self.tops = tops
        self.requests: list[list[int]] = []
        self.health_hits = 0
        self.fail_at: set[int] = set()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send(self, payload: object, code: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                outer.health_hits += 1
                self._send({"status": "ok"})

            def do_POST(self) -> None:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if self.path == "/tokenize":
                    self._send({"tokens": outer.prompt_ids})
                    return
                outer.requests.append(list(request["prompt"]))
                position = len(request["prompt"]) - len(outer.prompt_ids)
                if position in outer.fail_at:
                    self._send({"error": {"message": "injected failure"}}, 500)
                    return
                top = outer.tops[position]
                self._send({"completion_probabilities": [{
                    "id": top[0][0],
                    "top_logprobs": [{"id": tid, "logprob": val} for tid, val in top],
                }]})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "FakeLlamaServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def logits_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Everything --route logits needs with no real binary; run(name, *extra) drives main()."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no host llama-tokenize: the live /tokenize is the route
    build = tmp_path / "build"
    make_executable(build / "llama-cli")
    make_executable(build / "llama-server", "#!/bin/sh\nexec sleep 60\n")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf stand-in, version A")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({
        "prompt": "What is 17 multiplied by 23? The answer is",
        "prompt_ids": PROMPT_IDS,
        "ids_loops4": EXPECTED,
    }), encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "quantity": "logprobs",
        "source": "pytest",
        "positions": [[[tid, val] for tid, val in top] for top in FAKE_TOPS],
    }), encoding="utf-8")
    fake = FakeLlamaServer(PROMPT_IDS, FAKE_TOPS).start()

    def run(name: str, *extra: str, teacher_forced: bool = True) -> int:
        argv = [
            "--route", "logits", "--boot-timeout", "20",
            "--binary", str(build / "llama-cli"), "--model", str(model),
            "--oracle", str(oracle), "--server", str(build / "llama-server"),
            "--port", str(fake.port), "--reference", str(reference),
            "--checkpoint", str(tmp_path / f"{name}.checkpoint.json"),
            "--dump-observed", str(tmp_path / f"{name}.observed.json"),
        ]
        return oracle_check.main(argv + (["--teacher-forced"] if teacher_forced else []) + list(extra))

    return {"build": build, "model": model, "fake": fake, "run": run}


def test_checkpoint_is_written_as_each_position_arrives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt_path = tmp_path / "ckpt.json"
    on_disk_at_request: list[list[int]] = []

    def fake_http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
        assert url.endswith("/completion"), url
        stored = json.loads(ckpt_path.read_text(encoding="utf-8"))["positions"]
        on_disk_at_request.append([entry["position"] for entry in stored])
        top = FAKE_TOPS[len(payload["prompt"]) - len(PROMPT_IDS)]
        return {"completion_probabilities": [{
            "id": top[0][0], "top_logprobs": [{"id": tid, "logprob": val} for tid, val in top],
        }]}

    monkeypatch.setattr(oracle_check, "http_json", fake_http_json)
    ckpt = oracle_check.Checkpoint(ckpt_path, {"prompt_ids": PROMPT_IDS})
    ckpt.flush()
    session = types.SimpleNamespace(base="http://stand-in")  # never contacted: http_json is faked

    collected, ids, err, code = oracle_check.collect_teacher_forced(
        session, PROMPT_IDS, EXPECTED, 3, 5.0, sink=ckpt.add)

    assert (err, code) == ("", oracle_check.EXIT_MATCH)
    assert on_disk_at_request == [[], [0], [0, 1], [0, 1, 2]]  # every earlier position is already on disk
    doc = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert doc["format"] == oracle_check.CHECKPOINT_FORMAT
    assert [entry["sampled"] for entry in doc["positions"]] == [top[0][0] for top in FAKE_TOPS]
    assert [entry["top"] for entry in doc["positions"]] == [[[tid, val] for tid, val in top] for top in FAKE_TOPS]
    assert [q.name for q in tmp_path.iterdir() if q.name != "ckpt.json"] == []  # no temp file left behind


def test_identity_mismatch_names_every_differing_field(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"AAAA")
    server = make_executable(tmp_path / "llama-server")
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")

    def identity(server_bin: Path = server, reference_path: Path | None = None, **mode: object) -> dict:
        return oracle_check.run_identity(model, server_bin, reference_path, PROMPT_IDS, EXPECTED,
                                         {"teacher_forced": True, "n": 4, "n_probs": 5, "tol": 0.5, **mode})

    saved = identity()
    assert oracle_check.identity_mismatches(saved, identity()) == []

    model.write_bytes(b"BBBB")  # same path and size: only the bytes differ
    diffs = oracle_check.identity_mismatches(
        saved,
        identity(server_bin=make_executable(tmp_path / "other" / "llama-server", "#!/bin/sh\nexit 4\n"),
                 reference_path=reference, n_probs=6),
    )

    text = "\n".join(diffs)
    for field in ("model.sha256", "server.path", "server.sha256", "reference", "mode.n_probs"):
        assert field in text, text
    for field in ("model.path", "model.size", "prompt_ids", "expected_ids", "mode.teacher_forced", "mode.tol"):
        assert field not in text, text
    assert "mtime" not in text, text  # a rewritten file is caught by its bytes, not its timestamp


def test_resume_collects_only_the_missing_positions_and_reproduces_a_fresh_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    h = logits_harness(tmp_path, monkeypatch)
    fake = h["fake"]
    try:
        assert h["run"]("fresh") == oracle_check.EXIT_MATCH

        fake.fail_at.add(2)
        assert h["run"]("interrupted") == oracle_check.EXIT_CANNOT_RUN
        assert "--resume" in capsys.readouterr().err
        partial = json.loads((tmp_path / "interrupted.checkpoint.json").read_text(encoding="utf-8"))
        assert [entry["position"] for entry in partial["positions"]] == [0, 1]

        fake.fail_at.clear()
        fake.requests.clear()
        assert h["run"]("resumed", "--resume", str(tmp_path / "interrupted.checkpoint.json")) == oracle_check.EXIT_MATCH
    finally:
        fake.stop()

    # only the missing positions were requested, each on the saved prefix + expected suffix
    assert fake.requests == [PROMPT_IDS + EXPECTED[:2], PROMPT_IDS + EXPECTED[:3]]
    fresh = (tmp_path / "fresh.observed.json").read_text(encoding="utf-8")
    resumed = (tmp_path / "resumed.observed.json").read_text(encoding="utf-8")
    assert list(json.loads(resumed)) == list(json.loads(fresh))  # same keys in the same order
    assert json.dumps(json.loads(resumed)) == json.dumps(json.loads(fresh))  # same floats, digit for digit


def test_resume_is_refused_before_any_server_launch_when_identity_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    h = logits_harness(tmp_path, monkeypatch)
    fake = h["fake"]
    try:
        fake.fail_at.add(3)
        assert h["run"]("interrupted") == oracle_check.EXIT_CANNOT_RUN
        hits, requests = fake.health_hits, list(fake.requests)

        h["model"].write_bytes(b"gguf stand-in, version B")  # same size, different bytes
        rc = h["run"]("resumed", "--resume", str(tmp_path / "interrupted.checkpoint.json"))
    finally:
        fake.stop()

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING
    assert "IDENTITY MISMATCH" in captured.err and "model.sha256" in captured.err
    assert "VERDICT: PASS" not in captured.out and "BF16 VERDICT" not in captured.out
    assert (fake.health_hits, fake.requests) == (hits, requests)  # nothing launched, nothing collected


def test_resume_refuses_a_single_pass_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    h = logits_harness(tmp_path, monkeypatch)
    fake = h["fake"]
    try:
        fake.fail_at.add(2)
        assert h["run"]("interrupted") == oracle_check.EXIT_CANNOT_RUN
        single_pass = tmp_path / "single-pass.checkpoint.json"
        doc = json.loads((tmp_path / "interrupted.checkpoint.json").read_text(encoding="utf-8"))
        doc["identity"]["mode"]["teacher_forced"] = False  # what a run without --teacher-forced writes
        single_pass.write_text(json.dumps(doc), encoding="utf-8")
        hits = fake.health_hits

        rc = h["run"]("single", "--resume", str(single_pass), teacher_forced=False)
    finally:
        fake.stop()

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING
    assert "teacher-forced" in captured.err and "VERDICT: PASS" not in captured.out
    assert fake.health_hits == hits  # refused before anything was launched


def test_resume_from_an_unreadable_checkpoint_is_a_loud_plumbing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    h = logits_harness(tmp_path, monkeypatch)
    fake = h["fake"]
    try:
        garbage = tmp_path / "garbage.json"
        garbage.write_text("not json", encoding="utf-8")

        with pytest.raises(SystemExit) as raised:  # same convention as load_oracle() and load_reference()
            h["run"]("garbage", "--resume", str(garbage))
        hits = fake.health_hits
    finally:
        fake.stop()

    assert raised.value.code == oracle_check.EXIT_PLUMBING
    assert "checkpoint" in capsys.readouterr().err
    assert hits == 0



# ---------------------------------------------------------------------------
# BF16 substitution bound  (register 2125613721)
# ---------------------------------------------------------------------------


def test_bf16_mode_rejects_an_unexpected_candidate_with_real_probability() -> None:
    # Shared tokens move by 0.1 and the argmax agrees, so before the bound this PASSED:
    # id 12 sits in the port's top-5 with half the top token's probability while the
    # reference does not rank it at all.
    expected = [(10, -0.1), (11, -4.0), (13, -7.2)]
    actual = [(10, -0.2), (12, -0.9), (11, -4.1)]

    cmp = oracle_check.compare_position(actual, expected, "logprobs", 0.5)
    relaxed, info = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logprobs", 0.5)

    assert cmp["max_delta"] <= 0.5 and cmp["argmax_match"]  # the hole the bound closes
    assert not relaxed
    assert [entry["id"] for entry in info["unshared"] if not entry["within_bound"]] == [12, 13]


def test_bf16_mode_rejects_a_missing_reference_candidate_with_real_probability() -> None:
    # The reference gives id 11 90% of the top token's probability; the port drops it from its top-5.
    expected = [(10, -2.0), (11, -2.1), (12, -5.0)]
    actual = [(10, -2.1), (12, -5.0), (13, -7.5)]  # id 13 is a tail-level stand-in: 0.45% of the top token

    relaxed, info = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logprobs", 0.5)

    assert not relaxed
    assert [entry["id"] for entry in info["unshared"] if not entry["within_bound"]] == [11, 13]


def test_bf16_mode_still_allows_tail_level_substitutions() -> None:
    # Both sides swap their last-listed token, each far below 1% of the top token.
    expected = [(10, -0.1), (11, -4.0), (13, -7.2)]
    actual = [(10, -0.2), (11, -4.1), (14, -7.3)]

    relaxed, info = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logprobs", 0.5)

    assert relaxed
    assert [(entry["id"], entry["side"], entry["counterpart"]) for entry in info["unshared"]] == [(14, "port only", 13)]
    assert all(entry["within_bound"] for entry in info["unshared"])


def test_bf16_bound_is_the_acceptance_tolerance_between_paired_candidates() -> None:
    # A boundary swap is acceptable while the entrant sits within tol of the leaver;
    # a hair beyond tol makes the same swap an intrusion.
    expected = [(10, -0.1), (11, -4.0), (13, -2.0)]
    inside = [(10, -0.2), (11, -4.1), (14, -2.5)]
    outside = [(10, -0.2), (11, -4.1), (14, -2.6)]

    assert oracle_check.bf16_relaxed_position_verdict(inside, expected, "logprobs", 0.5)[0]
    assert not oracle_check.bf16_relaxed_position_verdict(outside, expected, "logprobs", 0.5)[0]


def test_bf16_substitution_is_measured_against_a_logits_reference_too() -> None:
    # Centred values are the same number in logits and logprobs, so the bound needs no units.
    expected = [(10, 12.5), (11, 8.6), (13, 5.3)]
    actual = [(10, 12.4), (12, 11.6), (11, 8.5)]

    relaxed, info = oracle_check.bf16_relaxed_position_verdict(actual, expected, "logits", 0.5)

    assert not relaxed
    unshared = {entry["id"]: entry for entry in info["unshared"]}
    assert unshared[12]["counterpart"] == 13 and unshared[12]["gap"] > 0.5


def test_bf16_mode_reports_a_substitution_failure_through_report_logits(capsys: pytest.CaptureFixture[str]) -> None:
    reference = report_reference()
    reference["provenance"] = "BF16 pytest"
    reference["positions"] = {0: [(1032, -0.1), (2007, -5.0), (1659, -5.5)]}
    actual = [(1032, -0.2), (9999, -0.6), (2007, -5.1)]  # 9999: half the top token's probability, unknown to the reference
    rows = [{"position": 0, "actual": actual, "why": ""}]

    rc = oracle_check.report_logits(
        rows, reference, 0.5, "pytest", Path("oracle.json"), "ids_loops4", [actual], None, allow_bf16_noise=True)

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_MISMATCH == 1
    assert "VERDICT: FAIL -- at least one position breaks the strict SET/ORDER acceptance" in captured.out
    assert "BF16 VERDICT: FAIL" in captured.out and "9999" in captured.out
    assert "BF16 VERDICT: PASS" not in captured.out



# ---------------------------------------------------------------------------
# BF16 flag plumbing  (register 2125613731)
# ---------------------------------------------------------------------------


def test_bf16_flag_with_a_non_bf16_reference_is_a_loud_plumbing_error(capsys: pytest.CaptureFixture[str]) -> None:
    actual = [(1032, -0.1), (2007, -5.0)]
    rows = [{"position": 0, "actual": actual, "why": ""}]

    rc = oracle_check.report_logits(
        rows, report_reference(), 0.5, "pytest", Path("oracle.json"), "ids_loops4", [actual], None, allow_bf16_noise=True)

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING == 3
    assert "PLUMBING" in captured.err and "--accept-bf16-noise" in captured.err
    assert "reference.json" in captured.err and "provenance" in captured.err and "'pytest'" in captured.err
    assert "BF16 VERDICT" not in captured.out  # the relaxed rule is never evaluated


def test_bf16_flag_is_refused_before_the_server_is_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    h = logits_harness(tmp_path, monkeypatch)
    fake = h["fake"]
    try:
        rc = h["run"]("bf16flag", "--accept-bf16-noise")  # the harness reference is declared source=pytest, not BF16
        hits = fake.health_hits
    finally:
        fake.stop()

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING
    assert "PLUMBING" in captured.err and "--accept-bf16-noise" in captured.err
    assert hits == 0 and fake.requests == []  # refused before anything was launched or collected


def test_bf16_flag_on_an_id_route_is_refused_rather_than_ignored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build = tmp_path / "build"
    binary = make_executable(build / "llama-cli")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real gguf")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"prompt": "hi", "prompt_ids": [1], "generated_ids": [2]}), encoding="utf-8")

    rc = oracle_check.main([
        "--route", "server", "--accept-bf16-noise", "--boot-timeout", "2",
        "--binary", str(binary), "--model", str(model), "--oracle", str(oracle),
        "--server", str(make_executable(build / "llama-server")),
    ])

    captured = capsys.readouterr()
    assert rc == oracle_check.EXIT_PLUMBING
    assert "--accept-bf16-noise" in captured.err and "--route logits" in captured.err



# ---------------------------------------------------------------------------
# follow-ups found by the CLI smoke run
# ---------------------------------------------------------------------------


def test_help_renders_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse %-formats help text, so a bare percent sign in any help string breaks --help.
    with pytest.raises(SystemExit) as raised:
        oracle_check.main(["--help"])

    captured = capsys.readouterr()
    assert raised.value.code == 0
    for flag in ("--server", "--checkpoint", "--resume", "--accept-bf16-noise"):
        assert flag in captured.out
    assert "1% of the top token" in captured.out


def test_foreign_tool_note_names_a_tokenizer_that_is_not_the_build_s(tmp_path: Path) -> None:
    build = tmp_path / "build"
    binary = make_executable(build / "llama-cli")
    own = make_executable(build / "llama-tokenize")
    foreign = make_executable(tmp_path / "homebrew" / "llama-tokenize")

    assert oracle_check.foreign_tool_note(binary, None, "llama-tokenize") == ""
    assert oracle_check.foreign_tool_note(binary, own, "llama-tokenize") == ""
    note = oracle_check.foreign_tool_note(binary, foreign, "llama-tokenize")

    assert "found on PATH" in note and str(foreign) in note and "not part of" in note


# ---------------------------------------------------------------------------
# Optional server flags across forks
# ---------------------------------------------------------------------------


def _fake_server(path: Path, help_text: str, exit_code: int = 0) -> Path:
    script = "#!/bin/sh\necho \"" + help_text + "\"\nexit " + str(exit_code) + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def _launch_line(capsys: pytest.CaptureFixture[str]) -> str:
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("launching:")]
    assert len(lines) == 1
    return lines[0]


def test_server_session_omits_a_flag_the_build_does_not_advertise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # ik_llama's server has no --no-webui and rejects it; the launch must not carry it.
    fake = _fake_server(tmp_path / "llama-server", "usage: --host --port --override-kv")
    session = oracle_check.ServerSession(fake, tmp_path / "m.gguf", 0, 0.2, tmp_path / "s.log", ctx=64)
    with session:
        pass
    line = _launch_line(capsys)
    assert "--no-webui" not in line
    assert "--override-kv" in line


def test_server_session_keeps_the_flags_a_build_advertises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _fake_server(tmp_path / "llama-server", "usage: --no-webui --override-kv --host")
    session = oracle_check.ServerSession(fake, tmp_path / "m.gguf", 0, 0.2, tmp_path / "s.log", ctx=64)
    with session:
        pass
    line = _launch_line(capsys)
    assert "--no-webui" in line
    assert "--override-kv" in line


def test_server_flag_probe_failure_keeps_the_flag(tmp_path: Path) -> None:
    # A binary that cannot even answer --help gets the previous unconditional behaviour.
    missing = tmp_path / "no-such-server"
    assert missing.exists() is False
    assert oracle_check.server_flag_supported(missing, "--no-webui") is True


def test_server_session_reads_help_even_when_the_server_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # ik_llama's server prints a full usage text on --help and then exits 1.  The
    # text is still the answer: an unadvertised flag must not reach the launch line.
    fake = _fake_server(tmp_path / "llama-server", "usage: --host --override-kv", exit_code=1)
    session = oracle_check.ServerSession(fake, tmp_path / "m.gguf", 0, 0.2, tmp_path / "s.log", ctx=64)
    with session:
        pass
    line = _launch_line(capsys)
    assert "--no-webui" not in line
    assert "--override-kv" in line


# ---------------------------------------------------------------------------
# Legacy probability shape (ik_llama): token strings and probabilities
# ---------------------------------------------------------------------------


def test_parse_positions_converts_the_legacy_probs_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # ik_llama's /completion answers with {"content", "probs": [{"tok_str", "prob"}]}:
    # token strings and post-softmax probabilities, no ids and no logprobs.
    monkeypatch.setattr(oracle_check, "_VOCAB", {"5": 1032, "1": 1044, "3": 1055})
    body = {"completion_probabilities": [{"content": "5", "probs": [
        {"tok_str": "5", "prob": 0.5}, {"tok_str": "1", "prob": 0.25}, {"tok_str": "3", "prob": 0.125}]}]}

    parsed, err = oracle_check.parse_positions(body, 3)

    assert err == "" and parsed is not None
    assert parsed[0]["sampled"] == 1032
    assert [tid for tid, _ in parsed[0]["top"]] == [1032, 1044, 1055]
    assert [lp for _, lp in parsed[0]["top"]] == pytest.approx([math.log(0.5), math.log(0.25), math.log(0.125)])


def test_parse_positions_refuses_a_legacy_candidate_it_cannot_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle_check, "_VOCAB", {"5": 1032})
    body = {"completion_probabilities": [{"content": "5", "probs": [
        {"tok_str": "5", "prob": 0.5}, {"tok_str": "?", "prob": 0.1}]}]}

    parsed, err = oracle_check.parse_positions(body, 2)

    assert parsed is None and "maps to no token id" in err
