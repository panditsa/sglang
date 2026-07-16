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


class _TokenspeedMoEModule(torch.nn.Module):
    """Carrier module exposing the attribute names tokenspeed's
    moe_process_weights / registry gluon apply expect."""

    pass


def _build_tokenspeed_moe_module(layer: torch.nn.Module, top_k: int, swiglu_limit: float):
    """Wrap the raw (pre-shuffle) Quark MXFP4 MoE weights in a module the
    tokenspeed registry can preprocess + run, and attach gpt-oss routing
    config (top-4 softmax, renormalized, no expert groups)."""
    w = _TokenspeedMoEModule()
    w.w13_weight = torch.nn.Parameter(
        layer.w13_weight.data.contiguous(), requires_grad=False
    )
    w.w13_weight_scale = torch.nn.Parameter(
        layer.w13_weight_scale.data.contiguous(), requires_grad=False
    )
    w.w2_weight = torch.nn.Parameter(
        layer.w2_weight.data.contiguous(), requires_grad=False
    )
    w.w2_weight_scale = torch.nn.Parameter(
        layer.w2_weight_scale.data.contiguous(), requires_grad=False
    )
    w.w13_input_scale = torch.nn.Parameter(
        layer.w13_input_scale.data.to(torch.float32), requires_grad=False
    )
    w.w2_input_scale = torch.nn.Parameter(
        layer.w2_input_scale.data.to(torch.float32), requires_grad=False
    )
    if getattr(layer, "w13_weight_bias", None) is not None:
        w.w13_weight_bias = torch.nn.Parameter(
            layer.w13_weight_bias.data.to(torch.float32), requires_grad=False
        )
    if getattr(layer, "w2_weight_bias", None) is not None:
        w.w2_weight_bias = torch.nn.Parameter(
            layer.w2_weight_bias.data.to(torch.float32), requires_grad=False
        )
    # GPT-OSS gate_up is stored concatenated [gate; up].
    w.w13_input_layout = "concatenated"
    # Routing config read by the registry gluon apply kernel.
    w.top_k = int(top_k)
    w._normalize_topk_weights = True
    w._routing_method_type = 0
    w._n_group = 0
    w._topk_group = 0
    w._routed_scaling_factor = 1.0
    import types

    w.swiglu_arg = types.SimpleNamespace(alpha=1.702, limit=float(swiglu_limit))
    w.swiglu_beta = 1.0
    return w


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
        if getattr(self, "_use_tokenspeed_runner", False):
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
        """Weight prep for the tokenspeed MoE runner backend: build the plan and
        run the registry weight preprocessor on the raw Quark weights, then stash
        the processed carrier module + plan on the layer for apply_weights."""
        from sglang.srt.layers.moe.moe_runner.tokenspeed import (
            load_tokenspeed_registry,
        )

        if layer.w13_input_scale is None or layer.w2_input_scale is None:
            raise ValueError("W4A8 MXFP4-FP8 MoE requires static input scales.")

        tk = load_tokenspeed_registry()
        swiglu_limit = float(
            getattr(self.moe_runner_config, "gemm1_clamp_limit", 0.0)
            or getattr(self.moe_runner_config, "swiglu_limit", 0.0)
            or 7.0
        )
        top_k = int(getattr(self, "top_k", 0) or getattr(layer, "top_k", 4))
        w = _build_tokenspeed_moe_module(layer, top_k, swiglu_limit)

        # Request internal_activation_dtype="fp8" so the registry selects the
        # FP8-routed warp-decode kernel (gluon_mxfp4_moe_apply -> the a8w4
        # gluon_mxfp_fused_moe path, cos=1.0 vs bf16 ref) rather than the
        # dynamic MXFP4-activation kernel (a4w4, ~1.2% error).
        plan = tk.moe_plan(
            weight_dtype="mxfp4",
            input_dtype=torch.bfloat16,
            activation="swiglu",
            with_bias=True,
            internal_activation_dtype="fp8",
        )
        tk.moe_process_weights(plan, w)

        layer.tokenspeed_moe_module = w
        layer.tokenspeed_moe_plan = plan
        layer.tokenspeed_swiglu_limit = swiglu_limit
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
        moe_runner_backend = get_moe_runner_backend()
        self._use_tokenspeed_runner = moe_runner_backend.is_tokenspeed()
        if self._use_tokenspeed_runner:
            # tokenspeed registry MoE: swiglu activation; weight prep + apply go
            # through moe_plan/moe_process_weights/moe_apply.
            self.runner = MoeRunner(
                MoeRunnerBackend.TOKENSPEED,
                replace(moe_runner_config, activation="swiglu"),
            )
            return
        if _use_aiter and get_moe_a2a_backend().supports_aiter():
            moe_runner_backend = MoeRunnerBackend.AITER

        if moe_runner_backend.is_aiter():
            # MXFP4 hard-codes Swiglu in the AITER kernel path.
            self.runner = MoeRunner(
                moe_runner_backend, replace(moe_runner_config, activation="swiglu")
            )
        else:
            raise NotImplementedError(
                "QuarkW4A8MXFp4MoE is currently only supported with AITER "
                "or tokenspeed (--moe-runner-backend tokenspeed)."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if getattr(self, "_use_tokenspeed_runner", False) and getattr(
            layer, "_tokenspeed_moe_ready", False
        ):
            from sglang.srt.layers.moe.moe_runner.tokenspeed import (
                TokenspeedMoeQuantInfo,
            )

            quant_info = TokenspeedMoeQuantInfo(
                w=layer.tokenspeed_moe_module,
                plan=layer.tokenspeed_moe_plan,
                hidden_pad=self.hidden_pad,
                swiglu_limit=layer.tokenspeed_swiglu_limit,
            )
            return self.runner.run(dispatch_output, quant_info)

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
