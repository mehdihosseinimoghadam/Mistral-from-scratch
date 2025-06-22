"""
Mistral Cache Utilities

This module contains helper classes and functions to manage the key/value cache
for transformer attention layers, supporting both causal and local attention windows.
The cache supports a "paged" design that is allocated for each layer separately,
and metadata is tracked to understand which positions in the sequence to cache.

Key components:
- get_cache_sizes: Determines per-layer cache size
- CacheInputMetadata: Holds per-layer cache position info
- BufferCache: Manages allocation of cache tensors per layer
- CacheView: Gives a view into a particular layer's cache and manages updates
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


def get_cache_sizes(
    n_layers: int,
    max_seq_len: int,
    sliding_window: Optional[int] | Optional[List[int]],
) -> List[int]:
    """
    Compute the per-layer cache capacity.
    - If no sliding_window: cache full sequence
    - If single integer: all layers share the same sliding window size
    - If list: repeat this pattern across all layers
    """
    if sliding_window is None:
        # Cache up to max_seq_len per layer
        return n_layers * [max_seq_len]
    elif isinstance(sliding_window, int):
        # Cache up to sliding_window tokens per layer
        return n_layers * [sliding_window]
    else:
        # Ensure list length evenly divides number of layers
        assert isinstance(
            sliding_window, list), f"Expected list, got {type(sliding_window)}"
        assert (
            n_layers % len(sliding_window) == 0
        ), f"Expected n_layers % len(sliding_window) == 0, got {n_layers} % {len(sliding_window)}"
        num_repeats = n_layers // len(sliding_window)
        return num_repeats * [w if w is not None else max_seq_len for w in sliding_window]


@dataclass
class CacheInputMetadata:
    """
    Packaged metadata that CacheView and BufferCache use to index into caches:
      - positions: global absolute token indices for RoPE
      - to_cache_mask: which token slots are retained
      - cached_elements: number of elements per sequence retained in cache
      - cache_positions: flattened index into cache tensors
      - prefill: True if it's a prefill pass; False if decoding incrementally
      - mask: attention mask object
      - seqlens: list of sequence lengths in the batch
    """
    positions: torch.Tensor
    to_cache_mask: torch.Tensor
    cached_elements: torch.Tensor
    cache_positions: torch.Tensor
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Interleaves two lists of tensors: [a,b,...], [x,y,...] -> [a,x,b,y,...]
    """
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    """
    Given a rotating buffer `cache`, return the last `seqlen` entries in order.
    This 'unrotates' a circular buffer view so that it's contiguous.
    """
    assert cache.ndim == 3  # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    """
    Gives a view into a particular layer's key and value cache tensors.
    Handles updates and returning correctly-interleaved keys and values.
    """

    def __init__(self, cache_k, cache_v, metadata: CacheInputMetadata, kv_seqlens):
        self.cache_k = cache_k  # shape: (B, cache_size, H, D)
        self.cache_v = cache_v
        self.metadata = metadata
        self.kv_seqlens = kv_seqlens

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        Write new key and value tensors into the cache using cache_positions.
        to_cache_mask is applied so we only store the last part of the sequence.
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        # Index-copy into flat cache buffer
        flat_cache_k.index_copy_(
            0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(
            0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge cached KV tensors with new KV tensors for decoding.
        Unrotates the cached buffer and interleaves with new tensors.
        """
        assert xk.ndim == xv.ndim == 3  # shape: (B*T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # Nothing is cached
            return xk, xv

        # Split new tensors by sequence length so we can interleave per-sequence
        xk_split: Tuple[torch.Tensor] = torch.split(
            xk, self.metadata.seqlens)  # type: ignore
        xv_split: Tuple[torch.Tensor] = torch.split(
            xv, self.metadata.seqlens)  # type: ignore
        assert len(xk_split) == len(self.kv_seqlens)

        # Unrotate per-batch caches to their original order
        cache_k = [unrotate(t, s)
                   for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s)
                   for t, s in zip(self.cache_v, self.kv_seqlens)]

        # Interleave old and new keys and values
        interleaved_k = interleave_list(cache_k, list(xk_split))
        interleaved_v = interleave_list(cache_v, list(xv_split))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        # Return cache up to current batch
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    Maintains a rectangular buffer for each layer's KV cache. 
    Provides utilities for incrementing sequence lengths, generating input metadata, and updating buffers.
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] | Optional[List[int]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        self.cache_sizes = get_cache_sizes(
            n_layers, max_seq_len, sliding_window)
        assert len(self.cache_sizes) == n_layers

        self.cache_k = {
            i: torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim)) for i, cache_size in enumerate(self.cache_sizes)
        }
        self.cache_v = {
            i: torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim)) for i, cache_size in enumerate(self.cache_sizes)
        }
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.cache_k[0].device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        for i in range(self.n_layers):
            self.cache_k[i] = self.cache_k[i].to(device=device, dtype=dtype)
            self.cache_v[i] = self.cache_v[i].to(device=device, dtype=dtype)
        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        # Increase cached sequence length per batch index
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens,
                                        device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> List[CacheInputMetadata]:
        """
        Compute metadata for each layer's cache based on current sequence lengths and cache capacity.
        The mask types depend on whether it's a prefill (first time) or decoding.
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert self.kv_seqlens is not None
        assert len(seqlens) == len(
            self.kv_seqlens), "seqlens length mismatch; did you forget to reset cache?"

        seqpos = self.kv_seqlens.tolist()
        return [self._get_input_metadata_layer(size, seqlens, seqpos) for size in self.cache_sizes]

    def _get_input_metadata_layer(self, cache_size: int, seqlens: List[int], seqpos: List[int]) -> CacheInputMetadata:
        # Compute boolean mask for tokens to cache
        masks = [
            [x >= seqlen - cache_size for x in range(seqlen)] for seqlen in seqlens]
        to_cache_mask = torch.tensor(
            sum(masks, []), device=self.device, dtype=torch.bool)
        cached_elements = torch.tensor(
            [sum(mask) for mask in masks], device=self.device, dtype=torch.long)
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(
            device=self.device, dtype=torch.long
        )
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions % cache_size + batch_idx * cache_size

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)

        # Choose correct mask based on prefill or decode pass
        if first_prefill:
            mask = BlockDiagonalCausalMask.from_seqlens(
                seqlens).make_local_attention(cache_size)
        elif subsequent_prefill:
            # Allow appending to already cached sequences
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=cache_size).item() for s, cached_s in zip(seqlens, self.kv_seqlens)
                ],
            ).make_local_attention_from_bottomright(cache_size)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=cache_size,
                kv_seqlen=(self.kv_seqlens +
                           cached_elements).clamp(max=cache_size).tolist(),
            )

        return CacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
