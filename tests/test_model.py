"""Smoke tests for the tiny LLaMA model (interpret mode, CPU)."""

import jax
import jax.numpy as jnp

from pallas_flash import ModelConfig, init_params, forward, generate
from pallas_flash.model import decoder_block, precompute_rope


def _tiny_cfg():
    return ModelConfig(
        vocab_size=64,
        dim=256,
        n_layers=2,
        n_heads=2,
        head_dim=128,
        ffn_hidden=512,
        max_seq_len=256,
    )


def test_forward_shape_and_finite():
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(0), cfg)
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = forward(params, tokens, cfg, interpret=True)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_generate_length():
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(1), cfg)
    prompt = jnp.array([[3, 1, 4, 1, 5]])
    out = generate(params, prompt, max_new_tokens=6, cfg=cfg, interpret=True)
    assert out.shape == (1, prompt.shape[1] + 6)
    # prompt prefix is preserved
    assert jnp.array_equal(out[:, : prompt.shape[1]], prompt)


def test_generate_deterministic():
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(2), cfg)
    prompt = jnp.array([[2, 7, 1]])
    a = generate(params, prompt, max_new_tokens=4, cfg=cfg, interpret=True)
    b = generate(params, prompt, max_new_tokens=4, cfg=cfg, interpret=True)
    assert jnp.array_equal(a, b)  # greedy decoding is deterministic


def test_single_block_finite():
    cfg = _tiny_cfg()
    params = init_params(jax.random.PRNGKey(3), cfg)
    x = jax.random.normal(jax.random.PRNGKey(4), (1, 16, cfg.dim), jnp.float32)
    cos, sin = precompute_rope(16, cfg.head_dim, cfg.rope_theta)
    y = decoder_block(params["layers"][0], x, cos, sin, cfg, interpret=True)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))
