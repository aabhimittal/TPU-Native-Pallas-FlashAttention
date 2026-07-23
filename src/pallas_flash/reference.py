"""A plain, readable attention implementation in pure ``jax.numpy``.

This serves two purposes:

* **Correctness oracle** — the Pallas kernel in :mod:`pallas_flash.flash_attention`
  is validated against this implementation in the test-suite.
* **XLA baseline** — this is exactly the kind of attention that the XLA compiler
  fuses by default; the benchmark compares the custom Pallas kernel against it.

It materializes the full ``[seq, seq]`` score matrix, which is what
FlashAttention is designed to avoid.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> jax.Array:
    """Standard scaled dot-product attention.

    Supports grouped-query / multi-query attention: if ``k``/``v`` have fewer
    heads than ``q`` (``num_heads`` a multiple of ``num_kv_heads``), the KV heads
    are repeated to match, mirroring the kernel's head grouping.

    Args:
        q: array of shape ``[batch, num_heads, seq_len_q, head_dim]``.
        k, v: arrays of shape ``[batch, num_kv_heads, seq_len_k, head_dim]``.
        causal: apply a causal (lower-triangular) mask if ``True``.
        sm_scale: softmax scale; defaults to ``1 / sqrt(head_dim)``.

    Returns:
        Array of shape ``[batch, num_heads, seq_len_q, head_dim]``.
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    if num_kv_heads != num_heads:
        q_per_kv = num_heads // num_kv_heads
        k = jnp.repeat(k, q_per_kv, axis=1)
        v = jnp.repeat(v, q_per_kv, axis=1)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * sm_scale

    if causal:
        seq_q = q.shape[2]
        seq_k = k.shape[2]
        q_pos = jax.lax.broadcasted_iota(jnp.int32, (seq_q, seq_k), 0)
        k_pos = jax.lax.broadcasted_iota(jnp.int32, (seq_q, seq_k), 1)
        scores = jnp.where(q_pos >= k_pos, scores, -1e30)

    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v.astype(jnp.float32))
    return out.astype(q.dtype)
