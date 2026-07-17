"""Check whether aiter uses a different sliding-window count than tokenspeed.
Recompute SDPA from the SWA table for several window sizes and see which one
matches aiter best.
"""
import sglang  # noqa: F401
import torch, torch._dynamo  # noqa: F401
import triton.language as _tl  # noqa: F401
_ = _tl.dtype

cap = torch.load("/workspace/ts_capture.pt", map_location="cuda")
q = cap["q"].cuda()
NQ = cap["tp_q_head_num"]; NKV = cap["tp_k_head_num"]
HD = cap["qk_head_dim"]; VD = cap["v_head_dim"]
scale = cap["scaling"]; window = cap["sliding_window_size"]
k_cache = cap["k_cache"].cuda(); v_cache = cap["v_cache"].cuda()
swa_pt = cap["swa_page_table"].cuda(); cs = cap["cache_seqlens"].cuda()
sinks = cap["sinks"].cuda() if cap["sinks"] is not None else None
ref_out = cap["ref_out"].cuda().float()
test_out = cap["test_out"].cuda().float()
B = swa_pt.shape[0]; q = q.view(B, NQ, HD); g = NQ // NKV

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()

def sdpa_with_window(win_keep):
    """win_keep = number of key positions kept incl. current token."""
    out = torch.zeros(B, NQ, VD, device="cuda", dtype=torch.float32)
    for b in range(B):
        S = int(cs[b].item())
        slots = swa_pt[b, :S].long().clamp(min=0)
        kk = k_cache[slots].float(); vv = v_cache[slots].float()
        lo = max(0, S - win_keep)
        pos = torch.arange(S, device="cuda")
        for h in range(NQ):
            kv = h // g
            lg = (kk[:, kv] @ q[b, h].float()) * scale
            lg = lg.masked_fill(pos < lo, float("-inf"))
            if sinks is not None:
                sk = sinks[h].float()
                m = torch.maximum(lg.max(), sk)
                ex = torch.exp(lg - m); den = ex.sum() + torch.exp(sk - m); p = ex / den
            else:
                p = torch.softmax(lg, 0)
            out[b, h] = p @ vv[:, kv]
    return out.reshape(B, NQ * VD)

print(f"window config = {window}")
print(f"aiter vs tokenspeed: cos={cos(ref_out, test_out):.5f}")
print("\nwin_keep | cos-vs-aiter | cos-vs-tokenspeed")
for wk in [window, window + 1, window + 2, window - 1]:
    s = sdpa_with_window(wk)
    print(f"  {wk:4d}   |   {cos(ref_out, s):.5f}   |   {cos(test_out, s):.5f}")
