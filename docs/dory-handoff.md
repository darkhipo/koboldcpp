# Handoff for the next mind - dory port via KoboldCpp PR #2676

Measured state as of 2026-11-25 02:15Z. CONTINUE this; do not restart it.
Nothing here is a plan. Numbers marked [log] were read from a log on this host;
numbers marked [api] were read from the GitHub API. The acceptance rule itself
lives in `docs/dory-port-contract.md` - the 2026-11-24 "ACCEPTANCE REVISED"
section is the one in force.

## Where the work lives

| What | Where |
|---|---|
| Submitted port | KoboldCpp PR https://github.com/LostRuins/koboldcpp/pull/2676, base `concedo` |
| PR branch | `dory-kobold` on the fork `darkhipo/koboldcpp` - three commits: looped-nGPT arch support, unused `POST_*_NORM` enum removal, BF16 acceptance mode |
| PR working tree | `/private/tmp/kobold-dory` (clean at the PR head) |
| Port origin | `~/llama.cpp`, branch `dory-arch`; its remote `fork` points at `https://github.com/darkhipo/koboldcpp.git` |
| Harness | `scripts/dory_oracle_check.py` and its tests, in the same PR |
| Model under test | `/Volumes/Safehold/models/dory-1.0-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (`dory.num_loops` = 4) |
| BF16 control | `/private/tmp/dory-bf16-loops4.gguf` - converter output with `num_loops` patched to 4 at byte offset 2122 |
| References | `/Volumes/Widekeep/oracle/discriminating-prompt.json`, `discriminating-prompt-top5.json` (BF16, logprobs) |

## What is true now

1. **The submission target is KoboldCpp, not ggml-org/llama.cpp.** [api] The
   ggml-org PRs #32963 and #32970 are CLOSED, unmerged, and must stay that way:
   that repository requires human authorship, and the body's AGENTS.md (on
   `origin/main`, section "External Contribution Policy") names it as a verified
   exclusion. Never open a PR, issue or review comment there.
2. **PR #2676 is open and waiting on a human.** [api] Not a draft, review
   required, 0 comments, 0 reviews, no CI checks reported, mergeable. Nothing
   is blocking it on our side.
3. **The duplicate PR #2675 was closed** on 2026-11-24 [api]; its unique
   content was carried into #2676.
4. **The port runs on KoboldCpp.** [log] The shipped Q5_K_M loads through
   `models/dory.cpp` and generates greedily: `" Paris. The capital"` with Paris
   at 100%. `llama_model_dory::graph_nodes_per_tensor()` is 32.
5. **A separate PR #2674** (`--usemmap=noprefetch`, `--load-mode mmap,noprefetch`)
   is open on the same repository [api]; a reviewer asked for the flag-list
   design and the branch already implements it. Its parser tests pass
   (6 passed + 9 subtests, run in the 2026-11-25 session).

## Verdicts - state them exactly this way

All measured 2026-11-24 on CPU against the 16-position teacher-forced oracle.

| Measurement | Result |
|---|---|
| Strict top-5 SET/ORDER, BF16 control (`num_loops` = 4) | **FAIL**, 3 of 16 positions: 3 and 15 are order-only; 5 has a rank-5 set difference (3075 vs 3367) and the argmax flips inside a ~0.008-logprob tie in the reference itself. Max shared-token delta 0.299 |
| Strict top-5 SET/ORDER, shipped Q5_K_M | **FAIL**, 8 of 16 positions, max delta 1.170 at position 3 |
| Loop agreement, shipped Q5_K_M | 15/16 against loops = 4, 6/16 against loops = 1 |
| `--accept-bf16-noise`, BF16 control | **PASS 16/16 printed - NOT final** (see limitation 2) |
| Upstream llama.cpp build of the same port | fails the **identical 3 BF16 positions with identical logits** |
| Oracle + tensor-coverage tests | 13 passed in total (10 oracle, 3 tensor coverage) |

The upstream-build row is the answer to "is this a KoboldCpp graph defect?": no.
Backend probes (`-fa off`, `--no-repack`, `--device none`, `--no-op-offload`)
changed nothing.

## Known limitations - do not smooth these over

1. **The strict verdict fails.** It is recorded in the PR as a limitation. Q5_K_M
   is never described as passing; its deltas are a quantisation measurement.
2. **The BF16 PASS is not full precision.** Positions 0-8 were reconstructed
   from an interrupted log at three decimals and merged with exact positions
   9-15 by an ad hoc wrapper. A single uninterrupted rerun is owed
   (register 2125613718).
3. **Metal generation is unverified.** One Metal completion died with
   `kIOGPUCommandBufferCallbackErrorOutOfMemory` at a sampled peak footprint of
   106.3 GB. No GPU verdict is claimed (register 2125613723).

## The num_loops decision

The converter writes `dory.num_loops` from `config.json` (`n_recurrent_loops`
= 2). NVIDIA's shipped artifact carries 4, and 4 is what every runtime
validation above used. Loops materialise into KV slots, so the two files build
different-sized graphs. This is a deliberate, documented contract choice - the
PR description states both numbers. Do not "fix" either one to match the other.

## Open follow-ups (body register ids)

- 2125613718 - one uninterrupted full-precision BF16 oracle run; replace the merged artifact.
- 2125613719, 2125613720, 2125613721, 2125613731 - harness hardening: no silent
  PATH fallback for `llama-server`, per-position checkpoint/resume, a bound on
  which top-5 substitutions BF16 mode may ignore, a non-BF16 provenance test.
  2125613721 can legitimately turn the printed BF16 PASS into a FAIL - report it
  if it does.
- 2125613723, 2125613722 - Metal OOM root cause; missing generated kernels break
  the server even at `-ngl 0`.
- 2125613730 - rerun the BF16 conversion and re-diff all 507 tensors on the PR branch.
- 2125613724, 2125613725, 2125613726 - PR #2674: runtime proof that `noprefetch`
  suppresses `MADV_WILLNEED`; why no CI checks are reported; a UI `AGENTS.md`
  link to a missing root file.
- 2125613748 - `main` and `llamaserver` were deleted from the PR working tree
  by something unknown; relink with `make main llamaserver LLAMA_METAL=1`.
- 2125613732 - retire the obsolete `dory-*-split` worktrees after the PR merges.

The main entry is 2125613392; it stays open until #2676 merges.

## Operating notes that cost time to rediscover

* KoboldCpp binary names differ from upstream: `main` is llama-cli and
  `llamaserver` is llama-server. The oracle finds the server as a sibling of
  `--binary` and silently falls back to PATH; `/opt/homebrew/bin/llama-server`
  is a different build that fails with `unknown architecture 'dory'`.
* Metal sources live in `ggml/src/ggml-metal/`. If the generated kernels are
  removed, regenerate with
  `rm ggml-metal-device-m.o && make ggml-metal-device-m.o LLAMA_METAL=1`.
* `main` and `llamaserver` are ALWAYS remade (`build-info.h` never exists), and a
  signalled `make` deletes its target ("Deleting file 'llamaserver'"). That is how
  the binaries vanished on 2026-11-24, and again on 2026-11-25 when two builders
  ran in the same tree. Never run two makes there, never start one under a tool
  with a watchdog, and copy `llamaserver` aside after a successful build.
* Run long oracle jobs detached, never in a foreground shell. The host is
  routinely at 90%+ swap; gate 8 GB inference runs on swap used <= 16 GB.
* Never report a build, test or oracle result you did not run. "Queued" is not
  "landed": a commit has landed only when it is an ancestor of `origin/main`.
* The register's close guard rejects a sha it cannot resolve, so cite the
  checkout path + branch for anything outside the body repo.

## Body follow-through (verified 2026-11-25: each is an ancestor of origin/main)

* `4934ef6025` - soul selection and inline cap scale with the model window.
* `51da59f6dc` - ignore SIGTTOU before the TUI runs.
* `acdab261bf` - truncation envelopes are unmistakably not entries.

## Lane results, 2026-11-25 (branches `agent-*`, pushed to the fork `darkhipo/koboldcpp`)

None of this is in PR #2676 or PR #2674 yet.

* `agent-harness` - five commits hardening the oracle: no silent PATH fallback
  for `llama-server` (use `--server`), per-position checkpoint and `--resume`, a
  bound on top-5 substitutions in BF16 mode, a loud non-BF16 provenance refusal.
  30 tests. The bound flips the BF16 verdict to 15/16: position 5 is a rank-5
  swap (3367 against 3075, 0.038 logprob apart), which the mode as written
  tolerates without limit. Which bound to adopt is an OPEN contract decision.
* `agent-metal` - `tools/metal_probe_watchdog.py` (11 tests) and
  `docs/dory-metal-memory.md`: the 106 GB is either a compute pool with reuse
  defeated (scales with `-ub`) or context-sized attention scores plus KV (scales
  with `-c`); an init-only run separates them. RSS under-reports Metal memory.
* `agent-converter` - `scripts/dory_compare_gguf.py` and
  `docs/dory-converter-verification.md`: 507/507 names, zero shape mismatches.
* `agent-mmap-runtime` - `tools/madvise_runtime_proof.py` and
  `docs/notes-ci-status.md` for PR #2674; the runtime proof has not been run.
* `agent-fullprecision` - `run_bf16_full_when_memory_allows.py`, a gated runner
  for the full-precision BF16 rerun (waits for swap used <= 16 GB, then builds
  `llamaserver` if needed, then runs the oracle once and writes a result file).
