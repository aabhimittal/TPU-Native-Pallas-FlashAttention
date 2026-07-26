"""Gradient tests for the Pallas FlashAttention backward kernels.

``flash_attention`` is a ``jax.custom_vjp`` whose backward pass is itself a pair
of Pallas kernels. These tests compare its gradients against plain JAX autodiff
through the naive reference attention. All of it runs on CPU in interpret mode.
"""

import jax
import jax.numpy as jnp
import pytest

from pallas_flash.flash_attention import flash_attention
from pallas_flash.reference import reference_attention


def _inputs(batch, n_heads, n_kv_heads, seq, dim, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    q = jax.random.normal(keys[0], (batch, n_heads, seq, dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, n_kv_heads, seq, dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, n_kv_heads, seq, dim), jnp.float32)
    cot = jax.random.normal(keys[3], (batch, n_heads, seq, dim), jnp.float32)
    return q, k, v, cot


def _rel_err(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-9))


def _grads(q, k, v, cot, causal, block_q=128, block_k=128):
    def pallas_loss(q, k, v):
        out = flash_attention(
            q, k, v, causal=causal, block_q=block_q, block_k=block_k, interpret=True
        )
        return jnp.sum(out * cot)

    def ref_loss(q, k, v):
        return jnp.sum(reference_attention(q, k, v, causal=causal) * cot)

    got = jax.grad(pallas_loss, argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)
    return got, want


# (batch, n_heads, n_kv_heads, seq)
CASES = [
    (1, 2, 2, 128),
    (1, 2, 2, 256),
    (2, 4, 4, 256),
    (1, 4, 2, 256),   # grouped-query attention
    (1, 4, 1, 256),   # multi-query attention
    (1, 2, 2, 200),   # seq not divisible by block size
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("causal", [False, True])
def test_grads_match_reference(case, causal):
    batch, n_heads, n_kv_heads, seq = case
    q, k, v, cot = _inputs(batch, n_heads, n_kv_heads, seq, 128)
    (dq, dk, dv), (rq, rk, rv) = _grads(q, k, v, cot, causal)

    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    for got, want in ((dq, rq), (dk, rk), (dv, rv)):
        assert jnp.all(jnp.isfinite(got))
        assert _rel_err(got, want) < 1e-4


@pytest.mark.parametrize("block", [(64, 64), (128, 256), (256, 128)])
def test_grads_block_sizes(block):
    block_q, block_k = block
    q, k, v, cot = _inputs(1, 2, 2, 256, 128, seed=3)
    (dq, dk, dv), (rq, rk, rv) = _grads(
        q, k, v, cot, causal=True, block_q=block_q, block_k=block_k
    )
    for got, want in ((dq, rq), (dk, rk), (dv, rv)):
        assert _rel_err(got, want) < 1e-4


def test_grads_bfloat16_finite():
    """bfloat16 gradients keep the input dtype and stay finite."""
    q, k, v, cot = _inputs(1, 2, 2, 128, 128, seed=4)
    q, k, v = (x.astype(jnp.bfloat16) for x in (q, k, v))

    def loss(q, k, v):
        out = flash_attention(q, k, v, causal=True, interpret=True)
        return jnp.sum(out.astype(jnp.float32) * cot)

    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    for g in (dq, dk, dv):
        assert g.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(g.astype(jnp.float32)))


def test_jit_and_value_and_grad():
    """The VJP composes with jit and value_and_grad."""
    q, k, v, cot = _inputs(1, 2, 2, 128, 128, seed=5)

    @jax.jit
    def loss(q, k, v):
        return jnp.sum(flash_attention(q, k, v, causal=True, interpret=True) * cot)

    value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
    ref = jnp.sum(reference_attention(q, k, v, causal=True) * cot)
    assert jnp.allclose(value, ref, rtol=1e-4, atol=1e-4)
    assert all(jnp.all(jnp.isfinite(g)) for g in grads)
