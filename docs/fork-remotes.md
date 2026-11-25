# Fork remotes and worktrees - dory / koboldcpp lanes

Verified 2026-11-25 02:10Z with `git remote -v`, `gh repo view`, `git ls-remote`.

## The rename, stated precisely

The GitHub fork that used to be `darkhipo/llama.cpp` is now named
**`darkhipo/koboldcpp`**. Both URLs answer (GitHub redirects) and
`git ls-remote` returns identical refs for each. Despite the new name it is
still a fork of **`ggml-org/llama.cpp`** (`gh repo view` reports that parent).
It is NOT a fork of LostRuins/koboldcpp - the name collides, the repository
does not. PRs #2676 and #2674 are opened against LostRuins/koboldcpp from this
fork's branches (`dory-kobold`, `mmap-prefetch`). I could not determine who
renamed the fork or when.

Branches on the fork that matter:

| branch | tip | role |
|---|---|---|
| `dory-kobold` | 2bb0217 | PR #2676 head |
| `mmap-prefetch` | 4e04d5b | PR #2674 head |
| `dory-arch` | 8a96c3626 | port origin + acceptance contract docs |
| `dory-1.0-support` | 9c71a79b8 | earlier upstream PR branch (closed) |
| `enum-names` | 979da2fea | earlier enum-only PR branch (closed; content carried into #2676) |
| `mmap`, `mmap-pages-stats` | - | earlier mmap experiments |

## Remote mapping

| checkout | remote | URL | state |
|---|---|---|---|
| `~/llama.cpp` | `origin` | `https://github.com/ggml-org/llama.cpp.git` | correct - FETCH ONLY, never push |
| `~/llama.cpp` | `fork` | `https://github.com/darkhipo/koboldcpp.git` | **updated 2026-11-25** (was `darkhipo/llama.cpp`) |
| `/private/tmp/kobold-dory` | `origin` | `https://github.com/LostRuins/koboldcpp.git` | correct - pull/compare only |
| `/private/tmp/kobold-dory` | `fork` | `https://github.com/darkhipo/koboldcpp.git` | correct |

The update that was applied:

    git -C ~/llama.cpp remote set-url fork https://github.com/darkhipo/koboldcpp.git

It is shared by every worktree of that repository and is reversible with the
same command and the old URL. A `git fetch fork dory-arch --dry-run` succeeded
afterwards.

Open decision for the Emissary: the fork's new name is misleading for anyone
who reads "koboldcpp" as "a KoboldCpp fork". It is not changed here.

## Worktree survey (nothing deleted - see register 2125613732)

| path | branch @ tip | class |
|---|---|---|
| `/private/tmp/kobold-dory` | `dory-kobold-pr2676` @ 2bb0217 | active - PR #2676 working tree |
| `/private/tmp/kobold-mmap` | `mmap-prefetch` @ 4e04d5b | active - PR #2674 checkout |
| `~/llama.cpp` | `dory-arch` @ 8a96c3626 | active - port origin, contract docs |
| `/private/tmp/dory-agent-harness` | `agent-harness` | agent lane (oracle hardening) |
| `/private/tmp/dory-agent-fullprecision` | `agent-fullprecision` | agent lane (BF16 rerun) |
| `/private/tmp/dory-agent-metal` | `agent-metal` | agent lane (Metal OOM) |
| `/private/tmp/dory-agent-converter` | `agent-converter` | agent lane (converter) |
| `/private/tmp/mmap-agent-runtime` | `agent-mmap-runtime` | agent lane (PR #2674 runtime proof) |
| `/private/tmp/dory-agent-docs` | `agent-docs` | agent lane (this document) |
| `/private/tmp/dory-converter-split` | `dory-converter-split` @ 12d47ba2f | obsolete split |
| `/private/tmp/dory-runtime-split` | `dory-runtime-split` @ 12d47ba2f | obsolete split |
| `/private/tmp/dory-validation-split` | `dory-validation-split` @ 12d47ba2f | obsolete split |
| `/private/tmp/dorycheck` | detached @ 709f76020 | obsolete check |
| `/private/tmp/mastercheck` | detached @ 12d47ba2f | obsolete check |
| `/private/tmp/enum-pr` | `enum-names` @ 979da2fea | obsolete (closed PR) |
| `/private/tmp/llamacpp-master` | detached @ 6f41ac59e | git reports prunable (directory gone) |

Before removing anything classed obsolete: confirm the branch tip is on the
fork (`git ls-remote --heads https://github.com/darkhipo/koboldcpp.git`), run
`git -C <path> status --porcelain` in it, and remove the worktree only - never
the branch - until #2676 has merged.
