"""Correctness tests for the Pallas FlashAttention kernel.

These run in Pallas *interpret* mode so they execute on CPU and need no TPU.
The kernel output is compared against the naive reference attention.
"""

import jax
import jax.numpy as jnp
import pytest

from pallas_flash.flash_attention import flash_attention
from pallas_flash.reference import reference_attention


def _make_qkv(batch, heads, seq, dim, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (batch, heads, seq, dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, heads, seq, dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, heads, seq, dim), jnp.float32)
    return q, k, v


# (batch, heads, seq, head_dim) -- head_dim is 128 to match TPU tiling.
SHAPES = [
    (1, 1, 128, 128),
    (1, 2, 256, 128),
    (2, 3, 384, 128),
    (1, 2, 200, 128),   # seq not divisible by block size
    (1, 2, 130, 128),   # only just over one block
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_matches_reference(shape, causal):
    q, k, v = _make_qkv(*shape)
    out = flash_attention(q, k, v, causal=causal, interpret=True)
    ref = reference_attention(q, k, v, causal=causal)
    assert out.shape == ref.shape
    assert jnp.all(jnp.isfinite(out))
    assert jnp.max(jnp.abs(out - ref)) < 2e-3


@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 2), (8, 1), (4, 4)])
@pytest.mark.parametrize("causal", [False, True])
def test_grouped_query_attention(n_heads, n_kv_heads, causal):
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    q = jax.random.normal(keys[0], (1, n_heads, 256, 128), jnp.float32)
    k = jax.random.normal(keys[1], (1, n_kv_heads, 256, 128), jnp.float32)
    v = jax.random.normal(keys[2], (1, n_kv_heads, 256, 128), jnp.float32)
    out = flash_attention(q, k, v, causal=causal, interpret=True)
    ref = reference_attention(q, k, v, causal=causal)
    assert out.shape == (1, n_heads, 256, 128)
    assert jnp.max(jnp.abs(out - ref)) < 2e-3


def test_gqa_requires_divisible_heads():
    q = jnp.zeros((1, 6, 128, 128))
    k = v = jnp.zeros((1, 4, 128, 128))  # 6 not divisible by 4
    with pytest.raises(ValueError):
        flash_attention(q, k, v, interpret=True)


@pytest.mark.parametrize("block", [(64, 64), (128, 256), (256, 128)])
def test_block_sizes(block):
    bq, bk = block
    q, k, v = _make_qkv(1, 2, 512, 128, seed=1)
    out = flash_attention(q, k, v, causal=True, block_q=bq, block_k=bk, interpret=True)
    ref = reference_attention(q, k, v, causal=True)
    assert jnp.max(jnp.abs(out - ref)) < 2e-3


def test_custom_scale():
    q, k, v = _make_qkv(1, 2, 256, 128, seed=2)
    scale = 0.05
    out = flash_attention(q, k, v, causal=False, sm_scale=scale, interpret=True)
    ref = reference_attention(q, k, v, causal=False, sm_scale=scale)
    assert jnp.max(jnp.abs(out - ref)) < 2e-3
