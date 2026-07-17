"""Standalone SWA divergence replay.

Loads a captured diverging sliding-window decode (from tokenspeed_verify) and
diffs, element by element:
  1. aiter output vs tokenspeed output (the recorded divergence)
  2. tokenspeed output vs a from-scratch SDPA reference computed DIRECTLY from
     the SWA page table + SWA KV buffer (ground truth for "did my page table
     point at the right KV?")
  3. the same SDPA reference vs aiter output

If (2) is large but (3) is small -> my SWA page table / window is wrong.
If (2) is small but (3) is large -> aiter and my table agree; divergence is
   elsewhere (kernel numerics).
Run from /root.
"""
import sglang  # noqa: F401
import torch, torch._dynamo  # noqa: F401
import triton.language as _tl  # noqa: F401
_ = _tl.dtype
import math

cap = torch.load("/workspace/ts_capture.pt", map_location="cuda")
print(f"layer={cap['layer_id']} step={cap['step']} window={cap['sliding_window_size']} "
      f"recorded_cos={cap['cos']:.5f} max_seq={cap['max_seq_len']}")

q = cap["q"].cuda()                       # [total_q, ...] flattened
NQ = cap["tp_q_head_num"]; NKV = cap["tp_k_head_num"]
HD = cap["qk_head_dim"]; VD = cap["v_head_dim"]
scale = cap["scaling"]
window = cap["sliding_window_size"]
k_cache = cap["k_cache"].cuda()           # [num_slots, NKV, HD]  (SWA pool buffer)
v_cache = cap["v_cache"].cuda()
swa_pt = cap["swa_page_table"].cuda()     # [B, S] SWA slot ids
cs = cap["cache_seqlens"].cuda()          # [B]
sinks = cap["sinks"].cuda() if cap["sinks"] is not None else None
ref_out = cap["ref_out"].cuda().float()   # aiter
test_out = cap["test_out"].cuda().float() # tokenspeed

B = swa_pt.shape[0]
q = q.view(B, NQ, HD)
print(f"B={B} NQ={NQ} NKV={NKV} HD={HD} k_cache={tuple(k_cache.shape)} "
      f"swa_pt={tuple(swa_pt.shape)} cache_seqlens={cs.tolist()[:4]}")

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()

# (1) recorded divergence
print(f"\n(1) aiter vs tokenspeed:  cos={cos(ref_out, test_out):.5f} "
      f"rel={((test_out-ref_out).norm()/ref_out.norm()).item():.4f}")

# (2)/(3) from-scratch SDPA reference from the SWA page table
g = NQ // NKV
sdpa = torch.zeros(B, NQ, VD, device="cuda", dtype=torch.float32)
for b in range(B):
    S = int(cs[b].item())
    slots = swa_pt[b, :S].long()          # SWA slot id per position 0..S-1
    valid = slots >= 0
    kk = k_cache[slots.clamp(min=0)].float()   # [S, NKV, HD]
    vv = v_cache[slots.clamp(min=0)].float()
    for h in range(NQ):
        kv = h // g
        lg = (kk[:, kv] @ q[b, h].float()) * scale     # [S]
        # sliding window: keep [S-1-window, S-1]
        lo = max(0, S - 1 - window)
        pos = torch.arange(S, device="cuda")
        mask = (pos < lo) | (~valid)
        lg = lg.masked_fill(mask, float("-inf"))
        if sinks is not None:
            sk = sinks[h].float()
            m = torch.maximum(lg.max(), sk)
            ex = torch.exp(lg - m); den = ex.sum() + torch.exp(sk - m); p = ex / den
        else:
            p = torch.softmax(lg, 0)
        sdpa[b, h] = p @ vv[:, kv]
sdpa_flat = sdpa.reshape(B, NQ * VD)

print(f"(2) tokenspeed vs SDPA(my SWA table): cos={cos(test_out, sdpa_flat):.5f} "
      f"rel={((test_out-sdpa_flat).norm()/sdpa_flat.norm()).item():.4f}")
print(f"(3) aiter      vs SDPA(my SWA table): cos={cos(ref_out, sdpa_flat):.5f} "
      f"rel={((ref_out-sdpa_flat).norm()/sdpa_flat.norm()).item():.4f}")

# also: how many -1 (evicted) slots in the read window?
for b in range(min(B, 2)):
    S = int(cs[b].item()); lo = max(0, S - 1 - window)
    win_slots = swa_pt[b, lo:S]
    print(f"  b={b} S={S} window=[{lo},{S}) evicted(-1) in window: "
          f"{int((win_slots < 0).sum().item())}/{S-lo}")
