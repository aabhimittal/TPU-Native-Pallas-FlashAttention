"""Tests for RoPE fused into the attention kernel.

The fused kernel must be numerically identical to applying RoPE with separate
XLA ops and then calling the ordinary kernel — it just avoids the extra HBM
round-trips of the rotated Q and K.
"""

import jax
import jax.numpy as jnp
import pytest

from pallas_flash import flash_attention, flash_attention_rope
from pallas_flash.model import apply_rope, precompute_rope

THETA = 10000.0


def _inputs(n_heads, n_kv_heads, seq, dim=128, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (1, n_heads, seq, dim), jnp.float32)
    k = jax.random.normal(keys[1], (1, n_kv_heads, seq, dim), jnp.float32)
    v = jax.random.normal(keys[2], (1, n_kv_heads, seq, dim), jnp.float32)
    return q, k, v


@pytest.mark.parametrize("n_heads,n_kv_heads", [(2, 2), (4, 2), (4, 1)])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_matches_unfused(n_heads, n_kv_heads, causal):
    seq = 256
    q, k, v = _inputs(n_heads, n_kv_heads, seq)
    cos, sin = precompute_rope(seq, 128, THETA)

    fused = flash_attention_rope(q, k, v, cos, sin, causal=causal, interpret=True)
    unfused = flash_attention(
        apply_rope(q, cos, sin), apply_rope(k, cos, sin), v,
        causal=causal, interpret=True,
    )
    assert fused.shape == q.shape
    assert jnp.max(jnp.abs(fused - unfused)) < 2e-3


@pytest.mark.parametrize("seq", [130, 200])
def test_fused_ragged_lengths(seq):
    q, k, v = _inputs(2, 2, seq, seed=1)
    cos, sin = precompute_rope(seq, 128, THETA)
    fused = flash_attention_rope(q, k, v, cos, sin, causal=True, interpret=True)
    unfused = flash_attention(
        apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, causal=True, interpret=True
    )
    assert jnp.max(jnp.abs(fused - unfused)) < 2e-3


def test_fused_decode_shape_separate_tables():
    """A single query at absolute position `pos` against a longer key range."""
    pos, cache_size = 37, 256
    keys = jax.random.split(jax.random.PRNGKey(2), 3)
    q = jax.random.normal(keys[0], (1, 2, 1, 128), jnp.float32)
    k = jax.random.normal(keys[1], (1, 2, cache_size, 128), jnp.float32)
    v = jax.random.normal(keys[2], (1, 2, cache_size, 128), jnp.float32)
    cos, sin = precompute_rope(cache_size, 128, THETA)

    fused = flash_attention_rope(
        q, k, v, cos[pos:pos + 1], sin[pos:pos + 1], cos, sin,
        causal=False, block_q=1, kv_len=pos + 1, interpret=True,
    )
    unfused = flash_attention(
        apply_rope(q, cos[pos:pos + 1], sin[pos:pos + 1]), apply_rope(k, cos, sin), v,
        causal=False, block_q=1, kv_len=pos + 1, interpret=True,
    )
    assert jnp.max(jnp.abs(fused - unfused)) < 2e-3


def test_fused_bfloat16():
    seq = 128
    q, k, v = _inputs(2, 2, seq, seed=3)
    cos, sin = precompute_rope(seq, 128, THETA)
    out = flash_attention_rope(
        *(x.astype(jnp.bfloat16) for x in (q, k, v)), cos, sin,
        causal=True, interpret=True,
    )
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))


def test_fused_rejects_bad_table_shape():
    q, k, v = _inputs(2, 2, 128, seed=4)
    cos, sin = precompute_rope(64, 128, THETA)   # wrong length
    with pytest.raises(ValueError):
        flash_attention_rope(q, k, v, cos, sin, interpret=True)


def test_fused_requires_k_tables_when_lengths_differ():
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    q = jax.random.normal(keys[0], (1, 2, 1, 128), jnp.float32)
    k = jax.random.normal(keys[1], (1, 2, 128, 128), jnp.float32)
    v = jax.random.normal(keys[2], (1, 2, 128, 128), jnp.float32)
    cos, sin = precompute_rope(1, 128, THETA)
    with pytest.raises(ValueError):
        flash_attention_rope(q, k, v, cos, sin, block_q=1, interpret=True)
