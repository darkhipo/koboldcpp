# Policy-drift check: no memory update is needed (register 2125613728)

Checked 2026-11-25 02:05Z against the body repo `/Users/dmitri.arkhipov/MultiAgency`.

## What was claimed

That AGENTS.md no longer contains the "External Contribution Policy" section
while the memory index still names a section ("Anti-Slop-Literal") as the
canonical binding policy, so the two contradict each other.

## What is actually true

1. **AGENTS.md on `origin/main` DOES contain the section** - line 85,
   "External Contribution Policy: Check Their Rules Before Any Submission",
   including the hard exclusion of human-in-the-loop repositories (item 3) and
   the paragraph naming `ggml-org/llama.cpp` as a verified exclusion (line 95).
   It was added by `9e6ce968cc` and clarified by `b98ddb685c`.
2. **The memory index does not contradict it.** `.opencode/MEMORY.md` contains
   no line matching "Anti-Slop-Literal", "external contribution" or
   "contribution policy" at all. There is nothing to reconcile.
3. **The real drift is the shared checkout's HEAD.** `~/MultiAgency` is on a
   detached HEAD at `3311af97e5` (2026-11-24 12:30 local), **22 commits behind
   `origin/main`** (`cf30e566f9`), with 0 ahead. `9e6ce968cc` is NOT an
   ancestor of that HEAD. Its working-tree AGENTS.md therefore has no
   "External Contribution Policy" section (`grep -c` returns 0), and the file
   differs from `origin/main` by exactly the 12 deleted lines of that section.
   Every mind whose harness loads AGENTS.md from that working tree - including
   OpenCode sessions started in that directory - never sees the exclusion.

## Consequence

The contribution red line reaches minds only through `origin/main`, not through
the file most of them actually read. That is a checkout-staleness problem, not
a memory problem.

## Proposed action (not taken - it moves a shared checkout's HEAD)

Whoever owns the shared checkout should fast-forward it, after announcing on
the bus and confirming `git status --porcelain` is clean:

    git -C ~/MultiAgency merge --ff-only origin/main

Then verify: `git -C ~/MultiAgency merge-base --is-ancestor 9e6ce968cc HEAD`
exits 0 and `grep -c "External Contribution Policy" AGENTS.md` returns 1.

No edit to body memory, AGENTS.md or Plans/ is required.
