"""bfloat16 tests.

The kernel up-casts inputs to float32 inside VMEM, so accumulation (the online
softmax and the ``p @ v`` matmul) stays in float32 while HBM traffic halves.
Outputs come back in the input dtype. Tolerances here are bf16-scale, not f32.
"""

import jax
import jax.numpy as jnp
import pytest

from pallas_flash import ModelConfig, init_params, forward, generate_cached
from pallas_flash.flash_attention import flash_attention
from pallas_flash.reference import reference_attention

# bfloat16 carries ~8 mantissa bits; ~2e-2 absolute is the realistic band for
# these input magnitudes.
BF16_TOL = 3e-2


def _qkv(n_heads, n_kv_heads, seq, dim=128, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(keys[0], (1, n_heads, seq, dim), jnp.float32)
    k = jax.random.normal(keys[1], (1, n_kv_heads, seq, dim), jnp.float32)
    v = jax.random.normal(keys[2], (1, n_kv_heads, seq, dim), jnp.float32)
    return q, k, v


@pytest.mark.parametrize("n_heads,n_kv_heads", [(2, 2), (4, 2)])
@pytest.mark.parametrize("causal", [False, True])
def test_bf16_forward_matches_f32_reference(n_heads, n_kv_heads, causal):
    q, k, v = _qkv(n_heads, n_kv_heads, 256)
    out = flash_attention(
        *(x.astype(jnp.bfloat16) for x in (q, k, v)), causal=causal, interpret=True
    )
    ref = reference_attention(q, k, v, causal=causal)   # float32 oracle
    assert out.dtype == jnp.bfloat16
    assert jnp.max(jnp.abs(out.astype(jnp.float32) - ref)) < BF16_TOL


def test_bf16_output_dtype_preserved():
    q, k, v = _qkv(2, 2, 128, seed=1)
    out = flash_attention(
        *(x.astype(jnp.bfloat16) for x in (q, k, v)), causal=True, interpret=True
    )
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out.astype(jnp.float32)))


def test_bf16_model_forward_and_generate():
    """A bf16 tiny LLaMA runs forward and decodes with the cache."""
    cfg = ModelConfig(
        vocab_size=64, dim=256, n_layers=2, n_heads=2, head_dim=128,
        ffn_hidden=512, max_seq_len=128,
    )
    params = init_params(jax.random.PRNGKey(0), cfg, dtype=jnp.bfloat16)
    assert params["layers"][0]["wq"].dtype == jnp.bfloat16

    logits = forward(params, jnp.array([[1, 2, 3, 4]]), cfg, interpret=True)
    assert jnp.all(jnp.isfinite(logits.astype(jnp.float32)))

    out = generate_cached(
        params, jnp.array([[1, 2, 3]]), max_new_tokens=4, cfg=cfg, interpret=True
    )
    assert out.shape == (1, 7)
