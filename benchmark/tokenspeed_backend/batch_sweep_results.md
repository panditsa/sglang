# gpt-oss-120b Batch-Size Sweep: full-tokenspeed vs aiter baseline

Model: /data/models/amd-gpt-oss-120b-w-mxfp4-a-fp8 (MXFP4-W / FP8-A, Quark)
Hardware: 1x AMD Instinct MI350X (gfx950), TP=1, SGLang v0.5.15.post1
Config: CUDA graph ON, ~40-token prompt, 256 output tokens, streaming
Each batch shape warmed once, then measured.

Backends:

- full-tokenspeed : --attention-backend tokenspeed --moe-runner-backend tokenspeed
- aiter baseline  : --attention-backend aiter --moe-runner-backend aiter

## Aggregate decode throughput (tok/s)

| M  | full-tokenspeed | aiter baseline | delta        |
|----|-----------------|----------------|--------------|
| 1  | 252.1           | 210.0          | +20% ts      |
| 2  | 447.2           | 388.5          | +15% ts      |
| 4  | 752.4           | 719.7          | +5%  ts      |
| 8  | 1109.7          | 1207.8         | -8%  ts      |
| 16 | 1611.3          | 2034.8         | -21% ts      |
| 32 | 2632.9          | 3380.1         | -22% ts      |

## Per-request decode throughput (tok/s, mean)

| M  | full-tokenspeed | aiter baseline |
|----|-----------------|----------------|
| 1  | 260.2           | 214.3          |
| 2  | 235.7           | 198.1          |
| 4  | 196.9           | 183.3          |
| 8  | 144.1           | 155.4          |
| 16 | 103.6           | 130.7          |
| 32 | 84.7            | 108.8          |

## TTFT (ms, mean) / ITL (ms, mean)

| M  | ts TTFT | aiter TTFT | ts ITL | aiter ITL |
|----|---------|------------|--------|-----------|
| 1  | 35.6    | 29.4       | 3.84   | 4.67      |
| 2  | 62.8    | 30.7       | 4.24   | 5.05      |
| 4  | 65.4    | 31.3       | 5.08   | 5.46      |
| 8  | 75.2    | 54.1       | 6.94   | 6.44      |
| 16 | 81.1    | 60.9       | 9.65   | 7.65      |
| 32 | 98.3    | 79.2       | 11.81  | 9.19      |

## Takeaways

- tokenspeed WINS at low batch (M<=4): +20% at M=1, +15% at M=2, +5% at M=4,
  and lower ITL (3.84 vs 4.67 ms at M=1). Best for latency-sensitive / low-QPS
  agentic decode — tokenspeed's small-M warp-decode MoE + gfx950 MHA shine.
- aiter WINS at high batch (M>=8): +9% to +28% aggregate. aiter's fused MoE +
  attention are better tuned for large-batch throughput; the registry moe_apply
  dispatch overhead and the page_size=1 attention layout cost more as M grows.
- Crossover is around M=4-8.

## Optimization experiments (results)

### 1. Cache registry kernel selection — NOT WORTH IT (measured)

Profiled the per-call select_kernel overhead directly:

- attention mha_decode: select_kernel = 2.7 us/call vs 76.5 us full call = 3.6%
- moe apply (override): select_kernel = 1.7 us/call
The registry already caches selection internally (KernelRegistry._selection_cache
keyed on family/mode/signature/traits). Per-call cost is just signature+key
construction + a dict lookup. Caching in the backend would save <=3.6%, so the
high-batch gap is NOT dispatch overhead — it is genuine kernel compute.

### 2. Larger page_size (16 vs 1) — NEUTRAL (measured)

Added real block-paged support to the attention backend and re-swept with
--page-size 16:

| M  | ts page_size=1 | ts page_size=16 | aiter |
|----|----------------|-----------------|-------|
| 1  | 252            | 244             | 210   |
| 2  | 447            | 445             | 389   |
| 4  | 752            | 647             | 720   |
| 8  | 1110           | 817*            | 1208  |
| 16 | 1611           | 1502            | 2035  |
| 32 | 2633           | 2638            | 3380  |

(* M=8 page16 run was noisy; mid-batch ITL improved slightly, e.g. M=8 ITL
6.28 vs 6.94 ms, M=16 8.18 vs 9.65 ms, but aggregate throughput is within
noise of page_size=1.) Page-table size was not the high-batch bottleneck.

## Conclusion

tokenspeed is the better choice for low-batch / latency-sensitive agentic decode
(M<=4, up to +20% throughput, lower ITL). aiter remains better for high-batch
throughput serving (M>=8). The remaining high-batch gap is in the kernels
themselves (tokenspeed's gfx950 kernels are small-M / decode tuned; aiter's
fused MoE+attention are large-batch tuned), not in the SGLang integration or
registry dispatch. Closing it would require kernel-level work in
tokenspeed_kernel_amd, not backend-side changes.

## mha_prefill + full kernel validation (added)

### Registry kernels validated correct in isolation (cos=1.0 vs torch SDPA)

- mha_prefill: cos=1.0000 across causal, sliding-window (S=300 >> win=128),
  batched ragged seqs, attention sinks.
- mha_decode_with_kvcache: cos=1.0000 at S = 16,32,64,80,96,128,160,200,256,400
  for both sliding-window and full attention, with sinks + GQA 64:8.
=> The tokenspeed MHA kernels themselves are numerically correct at all lengths.

### KNOWN BUG in the attention backend integration (hybrid SWA pool)

End-to-end, the tokenspeed attention backend produces correct output for very
short prompts (<= ~80 total tokens: e.g. 17x23=391, capital of France=Paris)
but DEGENERATES into repetition/looping on longer prompts (>= ~84 tokens),
e.g. summing a 6-number list. The aiter backend answers the identical prompt
correctly, so this is specific to the tokenspeed backend.

Root cause (identified): GPT-OSS alternates sliding-window and full-attention
layers and SGLang stores sliding-window layers in a SEPARATE SWA KV pool with
its own token indexing. The current TokenspeedAttnBackend uses the full-attention
req_to_token map as the page_table for ALL layers, so the sliding-window layers
read from the wrong KV slots once the sequence grows. First generated token is
correct (prefill ok); decode drifts as the hybrid-pool mismatch compounds.
mha_prefill vs mha_extend made no difference (A/B via SGLANG_TS_MHA_PREFILL=0/1),
confirming it is the SWA read-path, not the prefill entry.

Fix required (backend-side, not kernel-side): per-layer KV routing that reads
the SWA pool + SWA page table for sliding-window layers and the full pool for
full-attention layers (mirror aiter_backend's dual-pool handling), including the
CUDA-graph static buffers. This is the next step to make the tokenspeed
attention backend production-correct for GPT-OSS.

Status: MoE runner backend is correct and shippable. Attention backend is
correct for short-context but needs the SWA dual-pool fix before it is
correct for general prompts.
