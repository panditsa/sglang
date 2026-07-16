from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


# ---------------------------------------------------------------------------
# Lazy registry import
#
# tokenspeed_kernel pulls in tokenspeed_triton, whose native modules perturb
# the (fragile, circular) triton-custom init if imported before SGLang has
# fully booted `triton`. So the registry is imported lazily, from inside run()
# / weight prep, which only execute after full engine startup.
# ---------------------------------------------------------------------------
_tk = None


def load_tokenspeed_registry():
    """Import and cache the tokenspeed_kernel registry module. Returns it."""
    global _tk
    if _tk is None:
        import tokenspeed_kernel as tk

        _tk = tk
    return _tk


@dataclass
class TokenspeedMoeQuantInfo(MoeQuantInfo):
    """Carrier for the tokenspeed MoE path.

    The heavy state (processed weights, precision configs, act scales, routing
    config) lives on the layer module ``w`` after ``moe_process_weights``; the
    runner only needs a handle to that module plus the execution ``plan``.
    """

    w: torch.nn.Module
    plan: dict
    hidden_pad: int = 0
    swiglu_limit: float = 0.0


@dataclass
class TokenspeedRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor  # int32
    topk_weights: torch.Tensor  # bf16
    router_logits: Optional[torch.Tensor]

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TOKENSPEED


@dataclass
class TokenspeedRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TOKENSPEED


class TokenspeedRunnerCore(MoeRunnerCore):
    """MoE runner that dispatches to the tokenspeed_kernel registry via
    ``moe_apply``. Weight preprocessing is done at load time through
    ``moe_plan`` + ``moe_process_weights`` (see the quant scheme)."""

    def run(
        self,
        runner_input: TokenspeedRunnerInput,
        quant_info: TokenspeedMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TokenspeedRunnerOutput:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        tk = load_tokenspeed_registry()

        hidden_states = runner_input.hidden_states
        true_hidden = hidden_states.shape[-1]

        # SGLang pads hidden_size (e.g. 2880 -> 3072) in create_weights, so the
        # tokenspeed-preprocessed weights expect padded activations.
        if quant_info.hidden_pad:
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, quant_info.hidden_pad), mode="constant", value=0.0
            )

        plan = quant_info.plan
        if plan.get("support_routing"):
            out = tk.moe_apply(
                plan,
                hidden_states,
                quant_info.w,
                runner_input.router_logits,
            )
        else:
            out = tk.moe_apply(
                plan,
                hidden_states,
                quant_info.w,
                runner_input.router_logits,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
            )

        if out.shape[-1] != true_hidden:
            out = out[..., :true_hidden].contiguous()
        return TokenspeedRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TOKENSPEED


# ---------------------------------------------------------------------------
# Pre-permute: StandardDispatchOutput -> TokenspeedRunnerInput
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "tokenspeed")
def pre_permute_standard_to_tokenspeed(
    dispatch_output: StandardDispatchOutput,
    quant_info: TokenspeedMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TokenspeedRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    router_logits = getattr(topk_output, "router_logits", None)

    return TokenspeedRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.bfloat16),
        router_logits=router_logits,
    )


# ---------------------------------------------------------------------------
# Post-permute: TokenspeedRunnerOutput -> StandardCombineInput
# ---------------------------------------------------------------------------


@register_post_permute("tokenspeed", "standard")
def post_permute_tokenspeed_to_standard(
    runner_output: TokenspeedRunnerOutput,
    quant_info: TokenspeedMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
