import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    """
    Configuration arguments for the Mixture of Experts (MoE) layer.

    Attributes:
        num_experts (int): Total number of expert networks available in the MoE layer.
        num_experts_per_tok (int): Number of experts to select (route to) per input token.
    """
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    """
    A Mixture of Experts (MoE) layer that routes each input to a subset of expert networks.

    This implementation selects top-k experts for each input token based on gate scores,
    and combines the expert outputs using a weighted sum of the top-k expert outputs.

    Args:
        experts (List[nn.Module]): A list of expert modules (e.g., feed-forward networks).
        gate (nn.Module): A gating network that outputs logits for selecting experts.
        moe_args (MoeArgs): Configuration parameters for the MoE layer.
    """

    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0, "There must be at least one expert."

        # Store experts as a ModuleList so they are properly registered with the model.
        self.experts = nn.ModuleList(experts)

        # The gating network that produces logits indicating which experts to use.
        self.gate = gate

        # MoE configuration arguments (number of experts, top-k selection, etc.)
        self.args = moe_args

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Output tensor after routing inputs through top-k experts.
        """

        # Get gate logits for all experts. Shape: (batch_size, num_experts)
        gate_logits = self.gate(inputs)

        # Select the top-k experts and corresponding weights per input (token).
        # selected_experts: indices of top-k experts for each token
        # weights: unnormalized gate values for top-k experts
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok)

        # Normalize weights with softmax to get probabilities for combining expert outputs.
        # Shape remains (batch_size, num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

        # Initialize a tensor to store the aggregated outputs.
        results = torch.zeros_like(inputs)

        # Iterate over all experts to compute and accumulate their contributions.
        for i, expert in enumerate(self.experts):
            # Find input indices where this expert was selected.
            # batch_idx: positions in batch, nth_expert: position within top-k for that token
            batch_idx, nth_expert = torch.where(selected_experts == i)

            # For each input that chose this expert, apply the expert and scale output
            # by its corresponding gate weight, then add to final results.
            results[batch_idx] += weights[batch_idx,
                                          nth_expert, None] * expert(inputs[batch_idx])

        return results
