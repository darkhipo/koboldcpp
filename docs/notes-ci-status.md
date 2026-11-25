# PR #2674 notes: CI status, the ui/apps/chat AGENTS.md link, runtime proof

Verified 2026-11-25 against the tree at 4e04d5b and the GitHub API.

## Why both PRs show "no checks reported"

Expected, not a problem. This repository has 13 workflows (the
`kcpp-build-release-*.yaml` family and `test-unofficial-arm64.yaml`) and every one
looked at is triggered by `workflow_dispatch`: a manually started release build
that takes an optional `commit_hash`. None runs on `pull_request` or on a push to
a fork branch, so a PR from a fork never gets a check row. Review is the
maintainer's; verification is local by design. Parser tests for this PR:
`pytest -q tests/test_mmap_flags.py` gives 6 passed and 9 subtests passed.

## The `ui/apps/chat/AGENTS.md` link is really broken

`ui/apps/chat/AGENTS.md` is the only tracked AGENTS.md. Its third line begins
"These add to the repository's [AGENTS.md](../../../AGENTS.md) for work in
`ui/apps/chat`". That resolves to a root `AGENTS.md` which does not exist in this
tree and has no commit in this clone's history. The file is a vendored copy of
upstream's web-client instructions, and the document it says it adds to was not
vendored with it. The file's own 23 lines carry no authorship or AI restriction;
they are design and prose guidance for `ui/apps/chat`, which this PR does not
touch.

Proposed fix (documentation only, NOT part of this PR): drop "add to the
repository's AGENTS.md and", or point the sentence at the directory's own files.

## Runtime proof that `noprefetch` suppresses MADV_WILLNEED - not yet run

`tools/madvise_runtime_proof.py` is written and self-gating. It loads a model
through `./main` (default) or `./llamaserver` (`--server`) under an lldb
breakpoint on `madvise` and reports the `advice=3` (WILLNEED) count. Every run is
skipped unless swap used is at most `--max-swap-mb`. Exit codes: 0 expectation
met, 1 not met or load failed, 3 memory gate closed.

It uses a debugger because a DYLD interposer does NOT work here: on macOS 27.2.1
dyld binds a plain binary's `posix_madvise` straight to libsystem_kernel even
with `DYLD_INSERT_LIBRARIES` set, so `tools/madvise_trace.c` and
`libmadvise_trace.dylib` stay silent for these tools. They are kept as the
recorded negative result. `main` and `llamaserver` are built in this worktree.

When swap used is under the gate (a small non-Dory GGUF is enough):

    python3 tools/madvise_runtime_proof.py --model <small non-Dory GGUF>
    python3 tools/madvise_runtime_proof.py --model <small non-Dory GGUF> --server

Until that has run and its output is attached to the PR, the runtime claim of
PR #2674 rests on parser tests and code reading, not on an observed syscall.
