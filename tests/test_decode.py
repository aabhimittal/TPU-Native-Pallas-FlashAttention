"""Tests for the single-query decode kernel and the fixed-size KV cache."""

import jax
import jax.numpy as jnp
import pytest

from pallas_flash import (
    ModelConfig,
    flash_attention_decode,
    init_kv_cache,
    init_params,
    prefill,
    decode_step,
)
from pallas_flash.reference import reference_attention


def _cache(batch, n_kv_heads, cache_size, dim, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (batch, 4, 1, dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, n_kv_heads, cache_size, dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, n_kv_heads, cache_size, dim), jnp.float32)
    return q, k, v


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("cache_len", [1, 5, 130, 256])
def test_decode_matches_reference(n_kv_heads, cache_len):
    """Decoding against a partially-filled cache == attention over its prefix."""
    cache_size = 256
    q, k, v = _cache(1, n_kv_heads, cache_size, 128, seed=1)
    out = flash_attention_decode(q, k, v, cache_len, interpret=True)
    ref = reference_attention(q, k[:, :, :cache_len], v[:, :, :cache_len], causal=False)
    assert out.shape == (1, 4, 1, 128)
    assert jnp.max(jnp.abs(out - ref)) < 2e-3


def test_decode_ignores_unwritten_slots():
    """Garbage past cache_len must not leak into the result."""
    filled = 8
    k = jnp.concatenate(
        [jnp.ones((1, 2, filled, 128)), jnp.full((1, 2, 120, 128), jnp.nan)], axis=2
    )
    v = k
    q = jnp.ones((1, 2, 1, 128))
    out = flash_attention_decode(q, k, v, filled, interpret=True)
    assert jnp.all(jnp.isfinite(out))
    # Attending uniformly over identical all-ones values returns ones.
    assert jnp.allclose(out, 1.0, atol=1e-5)


def test_decode_rejects_multi_row_query():
    q = jnp.zeros((1, 2, 4, 128))
    k = v = jnp.zeros((1, 2, 128, 128))
    with pytest.raises(ValueError):
        flash_attention_decode(q, k, v, 128, interpret=True)


def _tiny_cfg(n_heads=2, n_kv_heads=0):
    return ModelConfig(
        vocab_size=64, dim=n_heads * 128, n_layers=2, n_heads=n_heads,
        n_kv_heads=n_kv_heads, head_dim=128, ffn_hidden=512, max_seq_len=128,
    )


def test_init_kv_cache_shapes():
    cfg = _tiny_cfg(n_heads=4, n_kv_heads=2)
    cache = init_kv_cache(cfg, batch=2, cache_size=64)
    assert len(cache) == cfg.n_layers
    for layer in cache:
        assert layer["k"].shape == (2, cfg.n_kv_heads, 64, cfg.head_dim)
        assert layer["v"].shape == layer["k"].shape


def test_cache_capacity_is_fixed_across_steps():
    """The cache is written in place, so its shape never grows."""
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(0), cfg)
    prompt = jnp.array([[1, 2, 3]])
    _, cache = prefill(params, prompt, cfg, interpret=True, cache_size=16)
    assert cache[0]["k"].shape[2] == 16

    token = jnp.array([[4]])
    for pos in (3, 4, 5):
        _, cache = decode_step(params, token, cache, pos, cfg, interpret=True)
        assert cache[0]["k"].shape[2] == 16


def test_prefill_rejects_too_small_cache():
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(0), cfg)
    with pytest.raises(ValueError):
        prefill(params, jnp.arange(8).reshape(1, 8), cfg, interpret=True, cache_size=4)


def test_decode_step_rejects_overflow():
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(0), cfg)
    _, cache = prefill(params, jnp.array([[1, 2]]), cfg, interpret=True, cache_size=4)
    with pytest.raises(ValueError):
        decode_step(params, jnp.array([[3]]), cache, 4, cfg, interpret=True)
