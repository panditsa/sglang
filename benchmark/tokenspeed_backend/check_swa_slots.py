import torch
cap = torch.load("/workspace/ts_capture.pt", map_location="cpu")
swa = cap["swa_page_table"]
cs = cap["cache_seqlens"]
win = cap["sliding_window_size"]
ksize = cap["k_cache"].shape[0]
for b in range(swa.shape[0]):
    S = int(cs[b]); lo = max(0, S - 1 - win)
    inwin = swa[b, lo:S]
    neg = int((inwin < 0).sum())
    oob = int((inwin >= ksize).sum())
    mn = int(inwin.min()); mx = int(inwin.max())
    print(f"b={b} S={S} window=[{lo},{S}) neg={neg} oob={oob} min={mn} max={mx} ksize={ksize}")
    # also check the FULL row (what my swa_cache_seqlens=S makes the kernel potentially touch)
    full = swa[b, :S]
    print(f"   full row [0,{S}): neg={int((full<0).sum())} (evicted positions the kernel must NOT read)")
