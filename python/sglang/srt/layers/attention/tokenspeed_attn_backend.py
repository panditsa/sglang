from __future__ import annotations

"""Attention backend that dispatches to the tokenspeed_kernel registry MHA
kernels (mha_plan / mha_prefill / mha_extend_with_kvcache /
mha_decode_with_kvcache). Targets dense MHA models such as GPT-OSS (GQA,
sliding-window attention, attention sinks) on AMD gfx950.

Uses SGLang's KV pool as a paged cache: the req_to_token map is strided into
the registry page_table (page_size configurable).

HYBRID SWA STATUS (GPT-OSS): the model alternates sliding-window and
full-attention layers stored in separate KV pools. This backend routes each
layer to its pool (get_key_buffer(layer_id) is pool-aware) and translates the
sliding-window page table via full_to_swa_index_mapping (same approach as the
FA3 backend). This is CORRECT for full-attention layers and for short contexts,
and fixes many mid-length prompts, but a subtle SWA decode mismatch remains at
certain lengths (some prompts still degrade). Until that is resolved, the
recommended production config is:
    --moe-runner-backend tokenspeed --attention-backend aiter
which is fully correct (7/7 correctness suite). The full tokenspeed attention
path is functional for short/full-attention-dominant contexts and is the WIP
target for "gpt-oss fully through tokenspeed".
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


# Lazy registry handle (triton init-order safe: imported after full boot).
_tk = None


def _load_registry():
    global _tk
    if _tk is None:
        import tokenspeed_kernel as tk

        _tk = tk
    return _tk


@dataclass
class TokenspeedAttnMetadata:
    # Full-attention page_table for the batch: [batch, max_pages_per_seq].
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor  # int32 [batch]
    max_seq_len: int
    # Sliding-window-layer page_table (SWA-pool slot ids) and its own
    # cache_seqlens/max (capped at the window). Only built when the model uses a
    # hybrid SWA KV pool; None otherwise.
    swa_page_table: Optional[torch.Tensor] = None
    swa_cache_seqlens: Optional[torch.Tensor] = None
    swa_max_seq_len: int = 0


class TokenspeedAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata: Optional[TokenspeedAttnMetadata] = None
        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.use_sliding_window_kv_pool = (
            isinstance(self.token_to_kv_pool, SWAKVPool)
            and self.token_to_kv_pool.swa_layer_nums > 0
        )
        self.swa_out_cache_loc = None
        self.max_context_len = model_runner.model_config.context_len
        self.sliding_window_size = int(
            getattr(model_runner.model_config, "sliding_window", 0) or 0
        )
        # KV paging: registry MHA consumes [num_blocks, page_size, kv_heads, hd].
        # SGLang stores a flat [num_tokens, kv_heads, hd] buffer; with page_size>1
        # we view it as blocks and stride the token-level req_to_token map into a
        # block-level page table.
        self.page_size = int(getattr(model_runner.server_args, "page_size", 1) or 1)
        # Static buffers for CUDA-graph decode replay (allocated on demand).
        self._cg_page_table: Optional[torch.Tensor] = None
        self._cg_cache_seqlens: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
        else:
            self.swa_out_cache_loc = None

        seq_lens = forward_batch.seq_lens
        max_seq = int(seq_lens.max().item()) if seq_lens.numel() else 0
        req_to_token = self.req_to_token_pool.req_to_token
        token_table = req_to_token[forward_batch.req_pool_indices, :max_seq]
        swa_page_table = None
        if self.use_sliding_window_kv_pool:
            # Sliding-window layers read the SWA pool: translate the full-pool
            # token slots to SWA slots. Evicted (out-of-window) tokens map to -1;
            # the kernel's window_left masking never reads them, so the full
            # table + full seqlens are correct (validated: -1 outside the window
            # yields finite output).
            swa_tokens = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                token_table
            ).to(torch.int32)
            # Evicted tokens translate to -1; window_left masking excludes those
            # positions from the softmax, but a -1 page index would still gather
            # garbage (KV[-1]). Point evicted slots at 0 (a valid, masked slot).
            swa_tokens = torch.where(
                swa_tokens < 0, torch.zeros_like(swa_tokens), swa_tokens
            )
            swa_page_table = self._to_block_table(swa_tokens)
        self.forward_metadata = TokenspeedAttnMetadata(
            page_table=self._to_block_table(token_table),
            cache_seqlens=seq_lens.to(torch.int32),
            max_seq_len=max_seq,
            swa_page_table=swa_page_table,
            swa_cache_seqlens=seq_lens.to(torch.int32),
            swa_max_seq_len=max_seq,
        )

    @staticmethod
    def _is_swa_layer(layer: RadixAttention) -> bool:
        sw = getattr(layer, "sliding_window_size", -1)
        return sw is not None and sw > 0

    def _page_table_for_layer(self, layer: RadixAttention) -> torch.Tensor:
        """Return the page table this layer must use: the SWA-slot table for
        sliding-window layers (when a hybrid SWA pool is active), else full."""
        md = self.forward_metadata
        if (
            self.use_sliding_window_kv_pool
            and self._is_swa_layer(layer)
            and md.swa_page_table is not None
        ):
            return md.swa_page_table
        return md.page_table

    def _to_block_table(self, token_table: torch.Tensor) -> torch.Tensor:
        """Convert a token-level page table [B, S] into a block-level table
        [B, ceil(S/page_size)] of block ids. Identity when page_size==1."""
        if self.page_size == 1:
            return token_table.to(torch.int32)
        s = token_table.shape[1]
        strided = torch.arange(
            0, s, self.page_size, device=token_table.device, dtype=torch.int64
        )
        return (token_table[:, strided] // self.page_size).to(torch.int32)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # CUDA graph support (decode)
    # ------------------------------------------------------------------
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # Static block-level page_table + cache_seqlens that captured decode
        # graphs read; filled in-place each replay.
        max_blocks = (self.max_context_len + self.page_size - 1) // self.page_size
        self._cg_page_table = torch.zeros(
            (max_bs, max_blocks), dtype=torch.int32, device=self.device
        )
        self._cg_cache_seqlens = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

    def _fill_cuda_graph_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_to_token = self.req_to_token_pool.req_to_token
        max_seq = int(self.max_context_len)
        # Fill static buffers in place (addresses stay stable for graph replay).
        self._cg_cache_seqlens[:bs] = seq_lens.to(torch.int32)
        token_table = req_to_token[forward_batch.req_pool_indices, :max_seq]
        block_table = self._to_block_table(token_table)
        self._cg_page_table[:bs, : block_table.shape[1]] = block_table
        self.forward_metadata = TokenspeedAttnMetadata(
            page_table=self._cg_page_table[:bs],
            cache_seqlens=self._cg_cache_seqlens[:bs],
            max_seq_len=max_seq,
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
        else:
            self.swa_out_cache_loc = None
        self._fill_cuda_graph_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        # Graph-recordable static-shape metadata already lives in the static
        # buffers filled by init_forward_metadata_out_graph; nothing to do.
        pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _kv_caches_paged(self, layer_id: int):
        """Return (k_cache, v_cache) as paged tensors
        [num_blocks, page_size, num_kv_heads, head_dim]."""
        k = self.token_to_kv_pool.get_key_buffer(layer_id)
        v = self.token_to_kv_pool.get_value_buffer(layer_id)
        # SGLang stores [num_tokens, num_kv_heads, head_dim].
        if self.page_size == 1:
            return k.unsqueeze(1), v.unsqueeze(1)
        nt, kvh, hd = k.shape
        nblocks = nt // self.page_size
        usable = nblocks * self.page_size
        k = k[:usable].view(nblocks, self.page_size, kvh, hd)
        v = v[:usable].view(nblocks, self.page_size, kvh, hd)
        return k, v

    @staticmethod
    def _window_left(layer: RadixAttention) -> int:
        sw = layer.sliding_window_size
        if sw is not None and sw > 0:
            import os

            off = int(os.environ.get("TS_WINDOW_OFFSET", "0"))
            return int(sw) + off
        return -1

    @staticmethod
    def _logit_cap(layer: RadixAttention) -> float:
        return float(getattr(layer, "logit_cap", 0.0) or 0.0)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        tk = _load_registry()
        sinks = kwargs.get("sinks", None)

        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache and k is not None and v is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, self.swa_out_cache_loc), k, v
            )

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_cache, v_cache = self._kv_caches_paged(layer.layer_id)
        md = self.forward_metadata
        if (
            self.use_sliding_window_kv_pool
            and self._is_swa_layer(layer)
            and md.swa_page_table is not None
        ):
            page_table = md.swa_page_table
            cache_seqlens = md.swa_cache_seqlens
            max_seqlen_k = md.swa_max_seq_len
        else:
            page_table = md.page_table
            cache_seqlens = md.cache_seqlens
            max_seqlen_k = md.max_seq_len

        res = tk.mha_decode_with_kvcache(
            q_,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=1,
            window_left=self._window_left(layer),
            logit_cap=self._logit_cap(layer),
            sinks=sinks,
        )
        out = res.out if hasattr(res, "out") else res
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # extend / prefill
    # ------------------------------------------------------------------
    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        tk = _load_registry()
        sinks = kwargs.get("sinks", None)

        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache and k is not None and v is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, self.swa_out_cache_loc), k, v
            )

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        seq_lens = forward_batch.seq_lens
        extend_seq_lens = forward_batch.extend_seq_lens

        # Pure-prefill fast path: no cached prefix -> run mha_prefill directly on
        # the current q/k/v (no paged-cache gather). Requires k/v this call and
        # every request starting from position 0.
        import os

        _use_prefill = os.environ.get("SGLANG_TS_MHA_PREFILL", "1") == "1"
        prefix_lens = forward_batch.extend_prefix_lens
        no_prefix = prefix_lens is None or (
            int(prefix_lens.max().item()) == 0 if prefix_lens.numel() else True
        )
        if _use_prefill and no_prefix and k is not None and v is not None:
            k_ = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v_ = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
            cu = torch.zeros(
                extend_seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device
            )
            torch.cumsum(extend_seq_lens.to(torch.int32), dim=0, out=cu[1:])
            cu_cpu = cu.tolist()
            max_s = int(extend_seq_lens.max().item()) if extend_seq_lens.numel() else 0
            res = tk.mha_prefill(
                q_,
                k_,
                v_,
                cu,
                cu_cpu,
                max_seqlen=max_s,
                window_left=self._window_left(layer),
                logit_cap=self._logit_cap(layer),
                sinks=sinks,
            )
            out = res.out if hasattr(res, "out") else res
            return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

        k_cache, v_cache = self._kv_caches_paged(layer.layer_id)

        # cu_seqlens for query (extend) and kv (full visible cache)
        cu_q = torch.zeros(
            extend_seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device
        )
        torch.cumsum(extend_seq_lens.to(torch.int32), dim=0, out=cu_q[1:])
        cu_kv = torch.zeros(
            seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device
        )
        torch.cumsum(seq_lens.to(torch.int32), dim=0, out=cu_kv[1:])

        req_to_token = self.req_to_token_pool.req_to_token
        max_kv = int(seq_lens.max().item()) if seq_lens.numel() else 0
        token_table = req_to_token[forward_batch.req_pool_indices, :max_kv]
        if self.use_sliding_window_kv_pool and self._is_swa_layer(layer):
            # Sliding-window layer reads the SWA pool: translate slots.
            token_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                token_table
            ).to(torch.int32)
        page_table = self._to_block_table(token_table)
        max_q = int(extend_seq_lens.max().item()) if extend_seq_lens.numel() else 0

        res = tk.mha_extend_with_kvcache(
            q_,
            cu_q,
            cu_kv,
            k_cache,
            v_cache,
            page_table,
            seq_lens.to(torch.int32),
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            window_left=self._window_left(layer),
            logit_cap=self._logit_cap(layer),
            sinks=sinks,
        )
        out = res.out if hasattr(res, "out") else res
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def support_triton(self):
        return False
