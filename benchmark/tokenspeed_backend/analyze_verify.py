import json
from collections import defaultdict

rows = [json.loads(l) for l in open("/workspace/ts_verify.jsonl") if l.strip()]
print(f"total rows: {len(rows)}")

# split by phase and layer type
def summarize(pred, label):
    sub = [r for r in rows if pred(r)]
    if not sub:
        print(f"{label}: (none)")
        return
    coss = [r["cos"] for r in sub]
    bad = [r for r in sub if r["cos"] < 0.99]
    print(f"{label}: n={len(sub)} cos[min={min(coss):.4f} mean={sum(coss)/len(coss):.4f}] "
          f"bad(<0.99)={len(bad)} ({100*len(bad)/len(sub):.0f}%)")

print("\n=== by phase x layer-type ===")
summarize(lambda r: r["mode"] == "extend" and r["is_swa"], "extend / SWA layer")
summarize(lambda r: r["mode"] == "extend" and not r["is_swa"], "extend / FULL layer")
summarize(lambda r: r["mode"] == "decode" and r["is_swa"], "decode / SWA layer")
summarize(lambda r: r["mode"] == "decode" and not r["is_swa"], "decode / FULL layer")

# decode: divergence vs sequence length (bucketed)
print("\n=== decode SWA-layer cos vs max_seq bucket ===")
buckets = defaultdict(list)
for r in rows:
    if r["mode"] == "decode" and r["is_swa"]:
        buckets[r["max_seq"] // 20 * 20].append(r["cos"])
for b in sorted(buckets):
    c = buckets[b]
    print(f"  max_seq {b:4d}-{b+19}: n={len(c):4d} cos_min={min(c):.4f} cos_mean={sum(c)/len(c):.4f}")

print("\n=== decode FULL-layer cos vs max_seq bucket ===")
buckets = defaultdict(list)
for r in rows:
    if r["mode"] == "decode" and not r["is_swa"]:
        buckets[r["max_seq"] // 20 * 20].append(r["cos"])
for b in sorted(buckets):
    c = buckets[b]
    print(f"  max_seq {b:4d}-{b+19}: n={len(c):4d} cos_min={min(c):.4f} cos_mean={sum(c)/len(c):.4f}")

# first few worst decode rows
print("\n=== 10 worst decode rows ===")
dec = sorted([r for r in rows if r["mode"] == "decode"], key=lambda r: r["cos"])[:10]
for r in dec:
    print(f"  step={r['step']:3d} layer={r['layer_id']:2d} swa={r['is_swa']} "
          f"max_seq={r['max_seq']:3d} cos={r['cos']:.4f} rel={r['rel']:.3f}")
