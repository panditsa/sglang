# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.layers.quantization.utils import all_close_1d
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    round_up,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

_is_shuffle_moe_mxfp4 = is_gfx95_supported()

__all__ = ["QuarkW4A8MXFp4MoE"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter.ops.shuffle import (
        shuffle_scale,
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )

OCP_MX_BLOCK_SIZE = 32

# Opt-in: route this MoE scheme through tokenspeed's gluon MXFP4 kernels
# (tokenspeed_kernel_amd) instead of the AITER runner. Reversible via env var.
_use_tokenspeed_moe = get_bool_env_var("SGLANG_USE_TOKENSPEED_MOE") and _is_hip

# tokenspeed_kernel_amd pulls in tokenspeed_triton, whose native modules can
# trigger triton-custom's (fragile, circular) backend discovery if imported
# before SGLang has fully initialized `triton`. So the import is done LAZILY
# from within the scheme methods (which only run after full engine boot), not
# at module import time. Cached in these globals after first success.
_ts_gluon_fused_moe = None
_ts_preprocess_moe_weights = None


def _load_tokenspeed_moe():
    """Lazily import the tokenspeed MoE kernels. Returns True on success.
    Disables the tokenspeed path (falling back to AITER) on failure."""
    global _use_tokenspeed_moe, _ts_gluon_fused_moe, _ts_preprocess_moe_weights
    if _ts_gluon_fused_moe is not None:
        return True
    try:
        # FP8-activation routed fused MoE: uses the small-M warp-decode fast
        # path and matches SGLang's softmax-topk routing (validated cos=1.0).
        from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
            gluon_mxfp_fused_moe as _moe,
        )
        from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (
            preprocess_gluon_mxfp4_gfx950_moe_weights as _prep,
        )

        _ts_gluon_fused_moe = _moe
        _ts_preprocess_moe_weights = _prep
        logger.info(
            "SGLANG_USE_TOKENSPEED_MOE enabled: gpt-oss MoE will use "
            "tokenspeed_kernel_amd gluon MXFP4 kernels."
        )
        return True
    except Exception as _e:  # pragma: no cover - env-dependent
        _use_tokenspeed_moe = False
        logger.warning(
            "SGLANG_USE_TOKENSPEED_MOE set but tokenspeed_kernel_amd import "
            "failed (%s); falling back to AITER MoE.",
            _e,
        )
        return False


class _TokenspeedMoEModule(torch.nn.Module):
    """Carrier exposing the attribute names tokenspeed's
    preprocess_gluon_mxfp4_gfx950_moe_weights expects."""

    pass


class QuarkW4A8MXFp4MoE(QuarkMoEScheme):
    """Quark MoE scheme for MXFP4 weights with static FP8 activations."""

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        weight_dtype = self.weight_quant.get("dtype")
        input_dtype = self.input_quant.get("dtype")

        if not (
            weight_dtype == "fp4"
            and weight_qscheme == "per_group"
            and self.weight_quant.get("group_size") == OCP_MX_BLOCK_SIZE
            and not self.weight_quant.get("is_dynamic")
            and self.weight_quant.get("scale_format") == "e8m0"
        ):
            raise ValueError(
                "For W4A8 MXFP4-FP8 Fused MoE layers, weights must be "
                "static per-group FP4 with group_size=32 and e8m0 scales. "
                f"Found {self.weight_quant}."
            )

        if not (
            input_dtype in ("fp8_e4m3", "fp8_e4m3fn")
            and input_qscheme == "per_tensor"
            and not self.input_quant.get("is_dynamic")
        ):
            raise ValueError(
                "For W4A8 MXFP4-FP8 Fused MoE layers, activations must be "
                "static per-tensor fp8_e4m3/fp8_e4m3fn. "
                f"Found {self.input_quant}."
            )

        self.with_bias = False

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        self.with_bias = extra_weight_attrs.get("with_bias", False)
        if _use_aiter:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 256
            )
            hidden_size = round_up(hidden_size, 256)
            self.hidden_pad = hidden_size - layer.hidden_size
            self.intermediate_pad = (
                intermediate_size_per_partition_after_pad
                - layer.intermediate_size_per_partition
            )
        else:
            intermediate_size_per_partition_after_pad = intermediate_size_per_partition
            self.hidden_pad = 0
            self.intermediate_pad = 0

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition_after_pad,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=True,
        )
        self.intermediate_size_per_partition = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly.
        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                "weight_padded": weight_padded,
            },
        )

        weight_dtype = torch.uint8

        # WEIGHTS
        # MXFP4 weights are stored as uint8, with two FP4 values packed per
        # byte. The AITER path later views these buffers as float4_e2m1fn_x2.
        # Use ``zeros`` (not ``empty``) so the alignment padding (hidden
        # 2880->3072, intermediate 2880->3072 for GPT-OSS) dequantizes to
        # 0.0 if it ever reaches the matmul. The current AITER kernel
        # skips the padded tail via ``n_pad_zeros`` / ``k_pad_zeros`` so
        # this is defensive, but it matches ``Mxfp4MoEMethod``'s
        # convention for the same kernel.
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                w13_up_dim,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                w13_up_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        # MXFP4 uses one e8m0 scale per 32-value block. These scales are
        # loaded as uint8 and shuffled after loading for the kernel layout.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_up_dim,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        # 1. w2 scale is floor division of inter_dim by blockscale.
        # 2. w2 scale needs to scale up just as w2.
        # We combine 1. and 2. to keep the integer precision.
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                (w2_down_dim * 2) // OCP_MX_BLOCK_SIZE,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the activation scales are loaded in properly.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        # INPUT_SCALES
        # W4A8 checkpoints carry static per-tensor FP8 activation scales for
        # gate_up_proj and down_proj. These are separate from the MXFP4 weight
        # block scales above.
        w13_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _use_tokenspeed_moe and _load_tokenspeed_moe():
            self._process_weights_tokenspeed(layer)
            return
        # Mirror native MXFP4 post-load shuffling. The default
        # `SGLANG_USE_AITER_MOE_GU_ITLV=1` path uses the gate-up-aware
        # a16w4 layout; the `=0` fallback keeps the separated gate/up layout.
        # The Quark loader (`_load_quark_experts_weights` in
        # `python/sglang/srt/models/gpt_oss.py`) already writes the
        # SEPARATED-layout `[g0..g_{N-1}, u0..u_{N-1}]` buffer per expert,
        # which is exactly the starting state the native path is in after
        # its post-load `.view(e, n//2, 2, k).permute(0, 2, 1, 3)` step.
        if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get():
            if _is_shuffle_moe_mxfp4:
                layer.w13_weight.data = shuffle_weight_a16w4(
                    layer.w13_weight.contiguous(), 16, True
                )
                layer.w2_weight.data = shuffle_weight_a16w4(
                    layer.w2_weight.contiguous(), 16, False
                )
                layer.w13_weight.is_shuffled = True
                layer.w2_weight.is_shuffled = True
            shuffled_w13_scale = shuffle_scale_a16w4(
                layer.w13_weight_scale.view(-1, layer.w13_weight_scale.shape[-1]),
                self.num_experts,
                True,
            )
            shuffled_w2_scale = shuffle_scale_a16w4(
                layer.w2_weight_scale.view(-1, layer.w2_weight_scale.shape[-1]),
                self.num_experts,
                False,
            )
        else:
            if _is_shuffle_moe_mxfp4:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(),
                    is_guinterleave=False,
                    gate_up=True,
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(),
                    is_guinterleave=False,
                    gate_up=False,
                )
                layer.w13_weight.is_shuffled = True
                layer.w2_weight.is_shuffled = True
            shuffled_w13_scale = shuffle_scale(
                layer.w13_weight_scale.view(-1, layer.w13_weight_scale.shape[-1]),
                experts_cnt=self.num_experts,
                is_guinterleave=False,
                gate_up=True,
            )
            shuffled_w2_scale = shuffle_scale(
                layer.w2_weight_scale.view(-1, layer.w2_weight_scale.shape[-1]),
                experts_cnt=self.num_experts,
                is_guinterleave=False,
                gate_up=False,
            )

        layer.w13_weight_scale = torch.nn.Parameter(
            shuffled_w13_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            shuffled_w2_scale, requires_grad=False
        )

        # Static FP8 MoE kernels consume a single activation scale. Use the
        # maximum if expert-local checkpoint scales differ.
        if layer.w13_input_scale is None or layer.w2_input_scale is None:
            raise ValueError("W4A8 MXFP4-FP8 MoE requires static input scales.")
        if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
            layer.w2_input_scale
        ):
            logger.warning(
                "Found input_scales that are not equal for W4A8 MXFP4-FP8 "
                "MoE layer. Using the maximum across experts for each layer."
            )
        layer.w13_input_scale = torch.nn.Parameter(
            layer.w13_input_scale.max().to(torch.float32), requires_grad=False
        )
        layer.w2_input_scale = torch.nn.Parameter(
            layer.w2_input_scale.max().to(torch.float32), requires_grad=False
        )

        if hasattr(layer, "dispatcher"):
            # Weights are stored as torch.uint8 but semantically MXFP4
            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})

    def _process_weights_tokenspeed(self, layer: torch.nn.Module) -> None:
        """Convert the raw Quark MXFP4 MoE weights into tokenspeed's gluon
        layout and attach the tokenspeed runtime tensors to `layer`.

        Runs on the pre-AITER-shuffle weights: the Quark loader writes the
        SEPARATED gate/up layout `[g0.., u0..]` per expert, which tokenspeed
        treats as `w13_input_layout="concatenated"`.
        """
        if layer.w13_input_scale is None or layer.w2_input_scale is None:
            raise ValueError("W4A8 MXFP4-FP8 MoE requires static input scales.")

        carrier = _TokenspeedMoEModule()
        carrier.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.contiguous(), requires_grad=False
        )
        carrier.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.contiguous(), requires_grad=False
        )
        carrier.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.contiguous(), requires_grad=False
        )
        carrier.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.contiguous(), requires_grad=False
        )
        carrier.w13_input_scale = torch.nn.Parameter(
            layer.w13_input_scale.data.to(torch.float32), requires_grad=False
        )
        carrier.w2_input_scale = torch.nn.Parameter(
            layer.w2_input_scale.data.to(torch.float32), requires_grad=False
        )
        if getattr(layer, "w13_weight_bias", None) is not None:
            carrier.w13_weight_bias = torch.nn.Parameter(
                layer.w13_weight_bias.data.to(torch.float32), requires_grad=False
            )
        if getattr(layer, "w2_weight_bias", None) is not None:
            carrier.w2_weight_bias = torch.nn.Parameter(
                layer.w2_weight_bias.data.to(torch.float32), requires_grad=False
            )
        carrier.w13_input_layout = "concatenated"

        _ts_preprocess_moe_weights(plan={}, w=carrier, preshuffle=True)

        # Stash tokenspeed runtime tensors; free the original big params.
        layer.ts_w13_weight = carrier.w13_weight_triton_tensor
        layer.ts_w2_weight = carrier.w2_weight_triton_tensor
        layer.ts_w13_mx_scale = carrier.w13_precision_config.b_mx_scale
        layer.ts_w2_mx_scale = carrier.w2_precision_config.b_mx_scale
        # Per-tensor FP8 activation scales computed by the preprocess. Required
        # by the FP8-routed fused MoE (its small-M warp-decode fast path).
        layer.ts_w13_act_scale = getattr(carrier, "w13_act_scale", None)
        layer.ts_w2_act_scale = getattr(carrier, "w2_act_scale", None)
        layer.ts_w13_bias = getattr(carrier, "w13_weight_bias", None)
        layer.ts_w2_bias = getattr(carrier, "w2_weight_bias", None)
        layer.ts_swiglu_limit = float(
            getattr(self, "swiglu_limit", 0.0) or 0.0
        )
        layer._tokenspeed_moe_ready = True

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config(
                {"weight_dtype": torch.float4_e2m1fn_x2}
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
        )

        self.moe_runner_config = moe_runner_config
        if _use_tokenspeed_moe and _load_tokenspeed_moe():
            # tokenspeed path does route+dispatch+combine itself; no MoeRunner.
            self.runner = None
            return
        moe_runner_backend = get_moe_runner_backend()
        if _use_aiter and get_moe_a2a_backend().supports_aiter():
            moe_runner_backend = MoeRunnerBackend.AITER

        if moe_runner_backend.is_aiter():
            # MXFP4 hard-codes Swiglu in the AITER kernel path.
            self.runner = MoeRunner(
                moe_runner_backend, replace(moe_runner_config, activation="swiglu")
            )
        else:
            raise NotImplementedError(
                "QuarkW4A8MXFp4MoE is currently only supported with AITER."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if (
            _use_tokenspeed_moe
            and getattr(layer, "_tokenspeed_moe_ready", False)
            and _ts_gluon_fused_moe is not None
        ):
            return self._apply_weights_tokenspeed(layer, dispatch_output)

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        if hasattr(torch, "float4_e2m1fn_x2"):
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
        else:
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

        if hasattr(layer.w13_weight, "is_shuffled"):
            w13_weight.is_shuffled = True
            w2_weight.is_shuffled = True

        x_padded = torch.nn.functional.pad(
            dispatch_output.hidden_states,
            (0, self.hidden_pad),
            mode="constant",
            value=0.0,
        )
        quant_info = AiterMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            b13=layer.w13_weight_bias,
            b2=layer.w2_weight_bias,
            expert_mask=layer.dispatcher.expert_mask_gpu,
            doweight_stage1=self.moe_runner_config.apply_router_weight_on_input,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            # gpt-oss populates `gemm1_clamp_limit` (renamed in
            # `models/gpt_oss.py` from `config.swiglu_limit`); DSv4 populates
            # `swiglu_limit` directly. Accept either so the AITER `gate_mode`
            # + `swiglu_limit` dispatch block in `moe_runner/aiter.py` (gated
            # on `quant_info.swiglu_limit > 0`) is actually entered for both
            # families. Mirrors the same fix PR #27201 applied to the native
            # `Mxfp4MoEMethod.apply` path.
            swiglu_limit=(
                self.moe_runner_config.gemm1_clamp_limit
                or self.moe_runner_config.swiglu_limit
                or 0.0
            ),
        )
        return self.runner.run(
            dispatch_output._replace(hidden_states=x_padded), quant_info
        )

    def _apply_weights_tokenspeed(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        # FP8-routed fused MoE does its own softmax-topk routing from the raw
        # router logits (validated to match SGLang's routing, cos=1.0) and can
        # take the small-M warp-decode fast path. Fall back to precomputed topk
        # only if router_logits is unavailable.
        router_logits = getattr(topk_output, "router_logits", None)

        # SGLang pads hidden_size (2880 -> 3072) in create_weights, so the
        # tokenspeed-preprocessed weights expect padded activations. Pad the
        # input K to match, then slice the output back to the true hidden size.
        true_hidden = hidden_states.shape[-1]
        if self.hidden_pad:
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, self.hidden_pad), mode="constant", value=0.0
            )

        top_k = int(topk_output.topk_ids.shape[-1])
        out = _ts_gluon_fused_moe(
            hidden_states,
            router_logits,
            layer.ts_w13_weight,
            layer.ts_w2_weight,
            w13_mx_scale=layer.ts_w13_mx_scale,
            w2_mx_scale=layer.ts_w2_mx_scale,
            w13_act_scale=layer.ts_w13_act_scale,
            w2_act_scale=layer.ts_w2_act_scale,
            top_k=top_k,
            w13_bias=layer.ts_w13_bias,
            w2_bias=layer.ts_w2_bias,
            out_dtype=hidden_states.dtype,
            swiglu_limit=layer.ts_swiglu_limit or 7.0,
        )
        if out.shape[-1] != true_hidden:
            out = out[..., :true_hidden].contiguous()
        return StandardCombineInput(hidden_states=out)
