"""Tests for the ring-buffer cache (sliding-window / unbounded streaming)."""

import jax
import jax.numpy as jnp
import pytest

from pallas_flash import (
    ModelConfig,
    flash_attention_decode,
    generate_cached,
    generate_streaming,
    init_params,
    init_ring_cache,
    prefill_ring,
    reference_attention,
    ring_decode_step,
    ring_kv_len,
)


def _cfg(n_heads=2, n_kv_heads=0):
    return ModelConfig(
        vocab_size=64, dim=n_heads * 128, n_layers=2, n_heads=n_heads,
        n_kv_heads=n_kv_heads, head_dim=128, ffn_hidden=512, max_seq_len=256,
    )


def test_ring_kv_len():
    assert ring_kv_len(3, 8) == 3      # not yet wrapped
    assert ring_kv_len(8, 8) == 8      # exactly full
    assert ring_kv_len(50, 8) == 8     # wrapped: window stays saturated
    assert ring_kv_len(1, 8) == 1


def test_ring_cache_shape_is_constant():
    cfg = _cfg()
    cache = init_ring_cache(cfg, batch=1, capacity=16)
    assert cache[0]["k"].shape == (1, cfg.n_kv_heads, 16, cfg.head_dim)


@pytest.mark.parametrize("n_heads,n_kv_heads", [(2, 2), (4, 2)])
def test_streaming_matches_cached_when_nothing_evicted(n_heads, n_kv_heads):
    """A window larger than the whole sequence must reproduce generate_cached."""
    cfg = _cfg(n_heads, n_kv_heads)
    params = init_params(jax.random.PRNGKey(0), cfg)
    prompt = jnp.array([[5, 9, 2, 7, 1]])
    n = 10
    streamed = generate_streaming(params, prompt, n, cfg, window=64, interpret=True)
    cached = generate_cached(params, prompt, n, cfg, interpret=True)
    assert jnp.array_equal(streamed, cached)


def test_streaming_runs_in_constant_memory_when_window_wraps():
    """With a small window the cache never grows, and generation continues."""
    cfg = _cfg()
    params = init_params(jax.random.PRNGKey(1), cfg)
    prompt = jnp.array([[5, 9, 2, 7, 1]])
    window = 8
    out = generate_streaming(params, prompt, 20, cfg, window=window, interpret=True)
    assert out.shape == (1, prompt.shape[1] + 20)

    # Walk the cache directly and confirm capacity is fixed throughout.
    _, cache = prefill_ring(params, prompt, cfg, window, interpret=True)
    token = jnp.array([[3]])
    for pos in range(prompt.shape[1], prompt.shape[1] + 15):
        _, cache = ring_decode_step(params, token, cache, pos, cfg, interpret=True)
        assert cache[0]["k"].shape[2] == window


def test_prefill_ring_keeps_the_most_recent_window():
    """A prompt longer than the window retains only its tail."""
    cfg = _cfg()
    params = init_params(jax.random.PRNGKey(2), cfg)
    window = 8
    prompt = jnp.arange(20).reshape(1, 20) % cfg.vocab_size
    _, cache = prefill_ring(params, prompt, cfg, window, interpret=True)
    assert cache[0]["k"].shape[2] == window
    # Every slot holds a real (non-zero) entry once the prompt exceeds the window.
    assert jnp.all(jnp.any(cache[0]["k"] != 0, axis=-1))


def test_ring_layout_is_permutation_invariant():
    """Attention over a wrapped ring == attention over the same tokens in order.

    RoPE is baked into the cached K, and attention does not care about key order,
    so a rotated ring must give the same answer as the contiguous window.
    """
    capacity, dim = 8, 128
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    q = jax.random.normal(keys[0], (1, 2, 1, dim), jnp.float32)
    k = jax.random.normal(keys[1], (1, 2, capacity, dim), jnp.float32)
    v = jax.random.normal(keys[2], (1, 2, capacity, dim), jnp.float32)

    in_order = flash_attention_decode(q, k, v, capacity, block_k=capacity, interpret=True)
    shift = 3
    rolled = flash_attention_decode(
        q, jnp.roll(k, shift, axis=2), jnp.roll(v, shift, axis=2),
        capacity, block_k=capacity, interpret=True,
    )
    assert jnp.max(jnp.abs(in_order - rolled)) < 2e-3

    ref = reference_attention(q, k, v, causal=False)
    assert jnp.max(jnp.abs(in_order - ref)) < 2e-3
