# Dory converter verification - existing artifacts (2026-11-25)

Status: the three complete BF16 files already on disk were compared against the
shipped Q5_K_M. The conversion was NOT re-run on this branch: the host sat at
37-80 GB of swap during the session and that step is gated on memory (see end).

## Where the converter lives

- This PR (#2676) carries the converter: `conversion/dory.py`,
  `conversion/__init__.py` (`"DoryForCausalLM": "dory"`), `gguf-py/gguf/constants.py`
  and `gguf-py/gguf/tensor_mapping.py`.
- The files compared below were produced on 2026-11-24 by the port-origin branch
  (`dory-arch`, commit "convert: support the dory architecture in gguf-py"), before
  the code moved into this tree. They are evidence about that converter, not yet
  about this branch's copy of it.

## Tool

`scripts/dory_compare_gguf.py` reads GGUF headers and the tensor directory only
(never tensor data); about 40 seconds for five files.

    python scripts/dory_compare_gguf.py --limit 4 <shipped Q5_K_M> <bf16 files...>

Exit 0 only when every comparison is clean apart from dtype. A file whose header
promises more bytes than are on disk is reported UNREADABLE instead of aborting.
Full output of the run summarised here:
`/private/tmp/dory-agent-converter-out/compare-existing.txt`.

## Files

| file | tensors | tensor bytes | dory.num_loops | state |
|---|---|---|---|---|
| shipped `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` | 507 | 3,665,387,520 | 4 | reference |
| `/private/tmp/dory-converted-bf16.gguf` | 507 | 7,952,711,680 | 2 | complete |
| `/private/tmp/dory-bf16-loops4.gguf` (num_loops patched to 4) | 507 | 7,952,711,680 | 4 | complete |
| `/private/tmp/dory-successor-20261124-090121-bf16.gguf` | 507 | 7,952,711,680 | 2 | complete |
| `/private/tmp/dory-converted-current-bf16.gguf` | - | 5,893,564,160 on disk | - | TRUNCATED, unreadable |

## Result against the shipped file (each complete BF16 file)

- Tensors: 507 vs 507, 0 missing, 0 extra, 0 shape mismatches. The earlier
  "507/507 names, zero shape mismatches" claim reproduces.
- dtype only: 252 tensors Q5_K -> BF16 and 36 tensors Q6_K -> BF16 (the weight
  matrices); 217 tensors BF16 -> F32 (the 1-D scale/alpha vectors, upcast; lossless).
- `dory.num_loops`: 2 in the conversion (from `config.json`), 4 in the shipped
  file. Deliberate and documented; the patched control with 4 compares CLEAN.
- Keys only in the shipped file: `general.basename`, `general.finetune`.
- Keys only in the conversion: the `general.dataset.*` provenance and one other
  key outside `dory.*` / `general.*`.
- `general.file_type` 17 vs 32 and `general.name` differ, as expected for a
  different quantisation and a converter-assigned name.
- `tokenizer.chat_template` differs, and it is NOT a converter defect: the
  conversion embeds the HF repository's `chat_template.jinja` byte for byte
  (verified), while the shipped GGUF carries a different revision of the template
  (it emits `<|im_start|>assistant` inside each branch and a newline after
  `<|im_end|>`; the HF template hoists the header into a `turn` block).

## Why `dory-converted-current-bf16.gguf` is 2 GB smaller

It is an interrupted write: a full header (507 tensors) over 5,893,564,160 of the
7,960,614,656 bytes a complete file needs. The GGUF reader fails on the first
tensor whose offset lies past EOF. Do not use it for anything.

## Not done here

The fresh conversion on this branch, then the same comparison against both the
shipped file and `dory-converted-bf16.gguf`. Commands, to run from this tree when
swap used is at most 16 GB:

    python convert_hf_to_gguf.py --outtype bf16 --outfile /private/tmp/dory-agent-converter-out/dory-1.0-4b-bf16.gguf /Volumes/Widekeep/oracle/dory-bf16
    python scripts/dory_compare_gguf.py --limit 8 <shipped> /private/tmp/dory-agent-converter-out/dory-1.0-4b-bf16.gguf /private/tmp/dory-converted-bf16.gguf

Acceptance for the rerun: 507/507 names, 0 shape mismatches, every strict key
equal to `dory-converted-bf16.gguf`, and identical tensor bytes.
