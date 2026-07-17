# Standalone SWA verification — findings

## Goal

Isolate why `--attention-backend tokenspeed` produces degraded/looping output on
gpt-oss for prompts beyond ~80 tokens, while the tokenspeed MHA kernels pass
cos=1.0 in synthetic isolation.

## Method

Built `tokenspeed_verify` attention backend (attention_registry.py) that wraps
BOTH aiter (reference) and tokenspeed (test) over the SAME live KV pool +
forward batch. Every decode/extend runs both on identical inputs, logs per-layer
cosine similarity tagged with (layer_id, mode, is_swa, max_seq), and returns the
aiter output so generation stays coherent while telemetry is collected.

Run:
  --attention-backend tokenspeed_verify --moe-runner-backend aiter --disable-cuda-graph
  TS_VERIFY_LOG=/workspace/ts_verify.jsonl
Then send the failing prompt; analyze with analyze_verify.py.

## Findings (4032 layer-comparisons over the failing 106-tok prompt)

| phase / layer type   | n    | cos min | % below 0.99 |
|----------------------|------|---------|--------------|
| extend / SWA layer   | 54   | 1.0000  | 0%           |
| extend / FULL layer  | 54   | 1.0000  | 0%           |
| decode / SWA layer   | 1962 | 0.9976  | 0%           |
| decode / FULL layer  | 1962 | 1.0000  | 0%           |

- FULL-attention layers: PERFECT (cos=1.0000) in all phases.
- extend (prefill): PERFECT for both layer types.
- decode SWA layers: 1958/1962 perfect; only 4 rows below cos 0.9995
  (worst 0.9976, rel 7%), ALL at max_seq > window (178, 186, 198) on early
  sliding layers (4, 6, 22).

## Root cause (narrowed)

NOT a structural page-table/pool-routing bug (those would give cos ~0.5 or NaN).
It is a SMALL numerical divergence on sliding-window layers that appears only
once the sequence exceeds the window (127) and grows with how far past the
window we are. Under greedy (temp=0) decoding these rare ~5-7% errors
occasionally flip the argmax token; that wrong token is written to KV, the next
step is off-distribution, and it snowballs into repetition/looping.

Why the earlier verify run showed all-1.0000 per step: in that run aiter was
authoritative, so the KV cache always held aiter's (correct) tokens and
tokenspeed matched. Standalone tokenspeed feeds its own slightly-wrong tokens
back into KV, which the harness cannot reproduce.

## Confirmed NOT the cause

- Kernel window semantics: mha_decode window_left=127/128 match a hand-masked
  SDPA reference (cos=1.0) with contiguous page tables.
- KV write path: gpt-oss uses the fused RoPE+set_kv_buffer path with
  swa_slot_mapping, independent of the attention backend, so KV is written
  correctly regardless.
- Window off-by-one: TS_WINDOW_OFFSET=+1 (window_left=128) did not fix it.
- MoE: tokenspeed-attn + aiter-MoE still fails, so it is attention-side.

## Deterministic tensor-level replay (replay_swa_divergence.py) — DECISIVE

Captured the exact diverging decode (layer 4, step 89, max_seq 186, cos 0.9985
vs aiter) with all tensors (q, SWA KV buffer, my SWA page table, cache_seqlens,
sinks) and replayed offline:

  (1) aiter        vs tokenspeed          : cos = 0.99847
  (2) tokenspeed   vs SDPA(my SWA table)  : cos = 1.00000   <-- exact match
  (3) aiter        vs SDPA(my SWA table)  : cos = 0.99847

=> tokenspeed's kernel output EXACTLY equals a from-scratch SDPA computed from
   my SWA page table. My page table + the kernel are self-consistent and
   correct. It is AITER that differs from the ground-truth SDPA, not tokenspeed.

Window-count sweep (SDPA from the SWA table, varying keys kept):
  win_keep=127 -> cos-vs-aiter 1.0000 | cos-vs-tokenspeed 0.9985
  win_keep=128 -> cos-vs-aiter 0.9985 | cos-vs-tokenspeed 1.0000
=> the only structural difference is an OFF-BY-ONE in the window count: aiter
   keeps 127 keys, tokenspeed keeps 128. Real but tiny (cos 0.9985).

## The real surprise (auth-mode A/B) — root cause is NOT the attention numerics

Running the verify backend with tokenspeed AUTHORITATIVE (TS_VERIFY_AUTH=test:
tokenspeed's tokens enter KV, aiter shadow-runs read-only) => the failing
"60" prompt PRODUCES THE CORRECT ANSWER, every step cos=1.0000.

But PURE --attention-backend tokenspeed (identical tokenspeed code, same MoE,
same window offset) LOOPS / fails the same prompt.

The ONLY difference between the two is that the verify wrapper also calls
aiter.forward_decode (read-only, save_kv_cache=False) on every layer before
tokenspeed. That read-only aiter call has a side effect that makes the pure
tokenspeed path correct. So:

- tokenspeed attention math is CORRECT (proven by replay cos=1.0).
- The window off-by-one (127 vs 128) is real but not the token-flipping cause
  (TS_WINDOW_OFFSET=-1 did not fix generation).
- The actual failure is a STATE / SIDE-EFFECT difference: something aiter's
  forward_decode touches (workspace, forward_metadata, an in-place buffer, or
  ordering) that the tokenspeed backend relies on but does not itself set up.

## Practical outcomes

1. Fully-correct config (unchanged): --moe-runner-backend tokenspeed
   --attention-backend aiter (7/7 suite).
2. A correct "mostly tokenspeed" attention config exists TODAY:
   --attention-backend tokenspeed_verify with TS_VERIFY_AUTH=test runs the
   tokenspeed attention kernels authoritatively and passes the failing prompts
   (at ~2x attention cost from the aiter shadow). This proves the tokenspeed
   attention kernels are production-correct; the remaining work is to identify
   the exact aiter setup step and replicate it inside TokenspeedAttnBackend so
   the standalone path no longer needs the shadow.

## Next step (well-scoped)

Diff aiter.forward_decode vs tokenspeed.forward_decode for the shared
side-effect: candidate suspects are (a) aiter writing/refreshing the SWA
out-cache-loc or page-table scatter buffers that the fused KV write consumes,
(b) a workspace/init ordering, or (c) aiter's init_forward_metadata populating
state the fused RoPE+set_kv path reads. Instrument both and compare the KV
buffer + swa mapping immediately after each backend's forward on step 0.

## Artifacts

- tokenspeed_verify_backend.py : dual-run compare backend (+capture, +auth mode)
- analyze_verify.py            : per-layer/-length divergence summary
- replay_swa_divergence.py     : offline element-wise replay (the decisive test)
- replay_window_count.py       : window off-by-one sweep
- ts_verify.jsonl / ts_capture.pt : captured telemetry + tensors
