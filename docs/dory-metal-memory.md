# Dory on Metal: where the 106 GB comes from

Lane C, 2026-11-24. Analysis only: no model was loaded or run here (host at ~93 % swap). Tickets
2125613723 (Metal OOM), 2125613722 (missing kernels). Hn = hypothesis, U = unverified.

## Inputs
- Measured earlier, not repeated: a 16-token Metal completion failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` at a peak footprint of 106.3 GB. CPU generation works.
- GGUF header (read, no tensor data): `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`, 3503 MiB, 507 tensors.
  n_embd 2560, n_ff 7168, 16 heads x 256, 4 KV heads x 256, sliding window 1024, training context 262144.
- 16 input + 40 recurrent + 16 output = 72 physical modules and `num_loops` 4 (src/models/dory.cpp:177-190): 192 slots, 96 attention (72 sliding, 24 full), 96 MLP.
- Metal memory, measured here (macOS 27.2.1, one 256 MiB MTLBuffer; RSS / footprint): untouched allocation 11.6 / 4.3 MiB; after a GPU fill 12.9 / 422 MiB (Private or Shared); after a CPU memset 267 / 260 MiB.
  Buffers are lazily committed, GPU-written pages never enter RSS, and footprint counts them 1:1 plus a fixed ~161 MiB for the first GPU command (8 to 512 MiB runs).

## (a) What a loop pass allocates (dory.cpp:281-366)
T = tokens in the microbatch; A = [2560, T] F32 = 10 KB/token.
- Attention slot, ~61 nodes by static count: 8 A-sized, 7 gate-width [4096, T] and 8 KV-width [1024, T] activations, about 229 KB/token. MLP slot, ~31 nodes: 8 A-sized and 7 [7168, T], 283 KB/token.
- Per pass (20 + 20 slots) 10.2 MB/token; whole graph about 49 MB/token of requests: 25.2 GB at T=512, 100.7 GB at T=2048, if nothing were reused.
- Alive across the whole graph only `input_hidden` (dory.cpp:374), the attention masks ((n_ctx + 1024) x 4 B/token) and `inp_pos`; nothing is created per loop beyond the slot's own activations and KV view (dory.cpp:326).
- ggml-alloc frees a tensor at its last consumer (ggml/src/ggml-alloc.c:790-816), so peak live memory is ~60 KB/token (attention) or ~105 KB/token (MLP) per slot. "Free the per-loop temporaries" is not an available fix: they are already transient.
- Static node count ~8.9k (92 per slot pair; measured ~6.5k); capacity 32 x 507 (src/models/models.h:2287 ff.).

## (b) How the pool is sized
`llama_context::graph_reserve` (src/llama-context.cpp:2998) builds the graph at the largest n_tokens; `ggml_gallocr_reserve_n` (ggml-alloc.c:959) places each tensor with a best-fit free list (:199).
Each backend then gets a buffer sized to the peak of simultaneously live tensors (U: possibly multiplied by scheduler hunks or copies), logged at llama-context.cpp:929-931 as `compute buffer size = N MiB` before any token is generated.
- Metal: the default buffer type is the shared one on unified memory (ggml/src/ggml-metal/ggml-metal.cpp:809-816): host memory wrapped with `newBufferWithBytesNoCopy` (ggml-metal-device.m:2212-2216). It exists from init but is resident only where CPU or GPU touched it.
- CPU: malloc-backed, paged in on first touch (allocation code not read here). Neither backend has the pool resident from the start, so ~100 GB really was written.

## (c) Arithmetic for 106.3 GB (= 99.0 GiB), T = 512
KV per slot is 4 KiB per context token (K + V, f16, 1024 wide each).

| source | formula | n_ctx 8192 | n_ctx 262144 |
|---|---|---|---|
| KV cache | 72 x 4 MiB + 24 x 4 KiB x n_ctx | 1.03 GiB | 24.3 GiB |
| attention masks | (n_ctx + 1024) x T x 4 | 18 MiB | 513 MiB |
| activations (reuse working) | ~105 KB x T | 54 MiB | 54 MiB |
| scores if flash-attention is off (U) | 2 x 16 x n_kv x T x 4 | 0.50 GiB | 16 GiB |
| weights | file size | 3.4 GiB | 3.4 GiB |
| total | | 4.5 / 5.0 GiB | 28.3 / 44.3 GiB |

No standard configuration reaches 100 GB. Each hypothesis below fits within about 5 % using one free parameter:
- H1, pool without cross-node reuse: 49 MB/token x T, which is 106 GB at T of about 2160.
- H2, context-sized tensors at a large ubatch: the same table at T = 2048 gives 93.9 GiB = 100.8 GB.

## (d) Why -ngl 0 still needs the kernels
- `llama_prepare_model_devices` (src/llama.cpp:182) adds every registered GPU to `model.devices` (:246-296, :302-303) without consulting `n_gpu_layers`; -ngl only places layer weights. `llama_context` then calls `ggml_backend_dev_init` on each and throws on failure (src/llama-context.cpp:465-471).
- Metal init compiles kernels from source when no `.metallib` exists (ggml-metal-device.m:548-602), from `<cwd>/kernels/` (:565-577) or `$GGML_METAL_PATH_RESOURCES` (:557). Only the `ggml-metal-device-m.o` recipe creates that directory (Makefile:410-415) and its prerequisites do not list the kernels: after a delete no timestamp changes, so `make` is a no-op.

## Ranked hypotheses
1. H2: score tensors plus the 192-slot KV cache at a large -c with T near 2048. The KV term follows from the GGUF dims; needs the failed run's -c, -ub and -fa.
2. H1: no cross-node reuse in the pool. Needs the free at ggml-alloc.c:802-815 to be defeated, for example by scheduler hunks (U).
3. KV alone (24.3 GiB at ctx 262144, 1.0 GiB at 8192), weights (3.4 GiB) and graph metadata: contributors at most, never the whole.

## Unknowns
- The failed run's -c, batch, ubatch, KV type, flash-attention state, `compute buffer size`, KV buffer size and number of scheduler splits.
- Whether flash-attention is eligible at head width 256 on Metal; whether the scheduler multiplies the pool by hunks or copies.
- Whether GPU-written pages of a host-memory buffer wrapped with `newBufferWithBytesNoCopy` count in RSS (measured only for `newBufferWithLength`: they do not).

## Bounded experiment, for when memory allows
Precondition: `sysctl vm.swapusage` well under 2 GB used, no other model server running.
1. Init only, zero generation, with the failed run's exact flags, under the watchdog. Read `n_ctx`, `compute buffer size`, KV buffer size, flash-attention and split lines from the log.
2. Repeat with `-c 64 -ub 64`, then `-c 64 -ub 2048`, then the old -c with `-ub 64`. H1 scales with -ub only; H2 scales with -c and changes with -fa.
3. Only then generate 4 tokens on Metal, plus the -ngl 0 control, each also with `-fa off` and `-fa on`. Exit 3 means killed for memory; the CSV `footprint_bytes` column shows when.

    W=tools/metal_probe_watchdog.py; G=/Volumes/Safehold/models/dory-1.0-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
    python3 $W --max-gb 12 --max-seconds 180 --csv dory-ngl99.csv -- <bin> -m $G -ngl 99 -c 64 -b 64 -ub 64 -n 4 -p Hello
    python3 $W --max-gb 12 --max-seconds 180 --csv dory-ngl0.csv  -- <bin> -m $G -ngl 0  -c 64 -b 64 -ub 64 -n 4 -p Hello

Init is not known to be cheap: clearing the KV cache may be a CPU memset (U), which is why step 1 keeps the 12 GB limit.

## Smallest fix (proposals, nothing implemented)
- Refuse above a computed bound: `ggml_gallocr_reserve_n_size` (ggml-alloc.c:948) sizes the pool without allocating. Compare it with free memory and fail with the number and a -c / -ub hint. About 15 lines near llama-context.cpp:2998 plus a test; needs a full build to verify.
- Kernels: make `kernels/` a stamp target the link step depends on, and make a GPU that fails to init non-fatal when nothing is offloaded (llama-context.cpp:465-471).
- Not worth doing: freeing per-loop temporaries (see (a)); lowering n_batch alone (H2 needs -c too).
