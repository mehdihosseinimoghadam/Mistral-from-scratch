from typing import Tuple
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """
    Precomputes the complex sinusoidal frequencies used in Rotary Positional Encoding (RoPE).

    Args:
        dim (int): The embedding dimension (must be even).
        end (int): The sequence length (number of positions).
        theta (float): Base frequency factor (typically 10,000.0).

    Returns:
        torch.Tensor: A complex tensor of shape (end, dim // 2) representing rotation angles.
    """
    # Calculate inverse frequency factors: shape (dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))

    # Sequence positions: shape (end,)
    t = torch.arange(end, device=freqs.device)

    # Compute outer product of positions and frequencies to get rotation angles: shape (end, dim // 2)
    freqs = torch.outer(t, freqs).float()

    # Convert angles to complex numbers on the unit circle: cos(angle) + i*sin(angle)
    # shape: (end, dim // 2), dtype: complex64
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional encoding to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor of shape (seq_len, batch_size, dim).
        xk (torch.Tensor): Key tensor of shape (seq_len, batch_size, dim).
        freqs_cis (torch.Tensor): Precomputed complex frequencies of shape (seq_len, dim // 2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of encoded (xq, xk), both with shape (seq_len, batch_size, dim).
    """
    # Convert xq to complex by reshaping last dim into pairs (real, imag): shape -> (seq_len, batch, dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

    # Same transformation for xk
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Expand freqs_cis to match query/key batch shape: (seq_len, 1, dim // 2)
    freqs_cis = freqs_cis[:, None, :]

    # Apply rotation (complex multiplication) for queries: shape -> (seq_len, batch, dim // 2, 2) → flatten → (seq_len, batch, dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)

    # Same for keys
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    # Convert back to original dtype (e.g., float16 or float32)
    return xq_out.type_as(xq), xk_out.type_as(xk)
