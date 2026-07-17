from __future__ import annotations

"""Verification attention backend for debugging the tokenspeed hybrid-SWA path.

Wraps TWO real attention backends over the SAME live KV pool / forward batch:
  - reference: aiter (known-correct)
  - test:      tokenspeed (under investigation)

On every forward_decode / forward_extend it runs BOTH on identical inputs,
records per-layer divergence (cos-sim, max abs err), tagged with whether the
layer is sliding-window or full-attention and the batch seq lengths, then
RETURNS THE REFERENCE (aiter) output so generation stays correct while we
collect divergence telemetry.

Select with --attention-backend tokenspeed_verify. Divergence rows are written
to $TS_VERIFY_LOG (default /workspace/ts_verify.jsonl) and a compact summary is
logged. This isolates exactly which layers / conditions the tokenspeed SWA
integration gets wrong, using the real SWA pool state (not synthetic tensors).
"""

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class TokenspeedVerifyBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        from sglang.srt.layers.attention.tokenspeed_attn_backend import (
            TokenspeedAttnBackend,
        )

        self.ref = AiterAttnBackend(model_runner)   # authoritative
        self.test = TokenspeedAttnBackend(model_runner)  # under test
        # Passthrough attrs SGLang reads directly off the active attn backend
        # (e.g. forward_context.get_token_to_kv_pool()).
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.device = model_runner.device
        self.log_path = os.environ.get("TS_VERIFY_LOG", "/workspace/ts_verify.jsonl")
        # {(layer_id, mode): [cos, ...]} rolling stats
        self._stats: dict = {}
        self._step = 0
        # truncate log at startup
        try:
            open(self.log_path, "w").close()
        except Exception:
            pass

    # --- metadata: both backends must see the batch ---
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.ref.init_forward_metadata(forward_batch)
        self.test.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # Verification runs eager only.
        self.ref.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.ref.get_cuda_graph_seq_len_fill_value()

    def _compare(self, mode, layer, forward_batch, ref_out, test_out):
        try:
            r = ref_out.float().flatten()
            t = test_out.float().flatten()
            cos = torch.nn.functional.cosine_similarity(r, t, dim=0).item()
            rel = ((t - r).norm() / (r.norm() + 1e-9)).item()
            sw = getattr(layer, "sliding_window_size", -1)
            is_swa = sw is not None and sw > 0
            seq_lens = forward_batch.seq_lens
            max_seq = int(seq_lens.max().item()) if seq_lens.numel() else 0
            row = {
                "step": self._step,
                "mode": mode,
                "layer_id": int(layer.layer_id),
                "is_swa": bool(is_swa),
                "window": int(sw) if is_swa else -1,
                "max_seq": max_seq,
                "cos": round(cos, 5),
                "rel": round(rel, 5),
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            key = (int(layer.layer_id), mode, bool(is_swa))
            self._stats.setdefault(key, []).append(cos)
        except Exception as e:
            logger.warning("ts_verify compare failed: %s", e)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        # TS_VERIFY_AUTH=test makes tokenspeed authoritative (its tokens enter
        # KV); default 'ref' keeps aiter authoritative.
        auth_test = os.environ.get("TS_VERIFY_AUTH", "ref") == "test"
        ref_out = self.ref.forward_decode(
            q, k, v, layer, forward_batch,
            save_kv_cache=(save_kv_cache and not auth_test), **kw
        )
        test_out = self.test.forward_decode(
            q, k, v, layer, forward_batch,
            save_kv_cache=(save_kv_cache and auth_test), **kw
        )
        self._compare("decode", layer, forward_batch, ref_out, test_out)
        self._maybe_capture(layer, forward_batch, q, ref_out, test_out, kw)
        if int(layer.layer_id) == 0:
            self._step += 1
        return test_out if auth_test else ref_out

    def _maybe_capture(self, layer, forward_batch, q, ref_out, test_out, kw):
        """When a sliding-window decode diverges beyond threshold, dump the exact
        tensors for offline element-wise replay (aiter vs tokenspeed). One-shot."""
        if getattr(self, "_captured", False):
            return
        sw = getattr(layer, "sliding_window_size", -1)
        if not (sw is not None and sw > 0):
            return
        cos = torch.nn.functional.cosine_similarity(
            ref_out.float().flatten(), test_out.float().flatten(), dim=0
        ).item()
        thr = float(os.environ.get("TS_CAPTURE_COS", "0.999"))
        if cos >= thr:
            return
        try:
            md = self.test.forward_metadata
            k_cache = self.test.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = self.test.token_to_kv_pool.get_value_buffer(layer.layer_id)
            path = os.environ.get("TS_CAPTURE_PATH", "/workspace/ts_capture.pt")
            torch.save(
                {
                    "layer_id": int(layer.layer_id),
                    "sliding_window_size": int(sw),
                    "step": self._step,
                    "cos": cos,
                    "q": q.detach().cpu(),
                    "sinks": (kw.get("sinks").detach().cpu()
                              if kw.get("sinks") is not None else None),
                    "k_cache": k_cache.detach().cpu(),
                    "v_cache": v_cache.detach().cpu(),
                    "full_page_table": md.page_table.detach().cpu(),
                    "swa_page_table": (md.swa_page_table.detach().cpu()
                                       if md.swa_page_table is not None else None),
                    "cache_seqlens": md.cache_seqlens.detach().cpu(),
                    "max_seq_len": md.max_seq_len,
                    "tp_q_head_num": layer.tp_q_head_num,
                    "tp_k_head_num": layer.tp_k_head_num,
                    "qk_head_dim": layer.qk_head_dim,
                    "v_head_dim": layer.v_head_dim,
                    "scaling": layer.scaling,
                    "ref_out": ref_out.detach().cpu(),
                    "test_out": test_out.detach().cpu(),
                },
                path,
            )
            self._captured = True
            logger.warning(
                "ts_verify: captured diverging SWA decode to %s "
                "(layer=%d step=%d cos=%.5f)",
                path, int(layer.layer_id), self._step, cos,
            )
        except Exception as e:
            logger.warning("ts_verify capture failed: %s", e)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        ref_out = self.ref.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kw
        )
        test_out = self.test.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=False, **kw
        )
        self._compare("extend", layer, forward_batch, ref_out, test_out)
        return ref_out

    def support_triton(self):
        return False
