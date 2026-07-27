# TPU-Native "Pallas" FlashAttention

A from-scratch **FlashAttention** kernel written in **JAX Pallas** for Google's
**TPU v3-8** (free on Kaggle), wired into a tiny **LLaMA-style decoder** for
inference. This is the JAX/TPU analogue of "writing CUDA": instead of relying on
the XLA compiler to fuse a naive attention, we hand-write a kernel that tiles
Q/K/V, manages the **HBM → VMEM** data movement explicitly, and runs the
online-softmax algorithm so the full attention matrix is never materialized.

Forward **and** backward are hand-written Pallas kernels, with grouped-query
attention, a single-query decode kernel over a fixed-size KV cache, bfloat16,
**RoPE fused into the kernel**, **paged attention** for batched serving, and a
**ring-buffer cache** for unbounded streaming.

## Why bother?

The default `softmax(Q Kᵀ) V` that XLA compiles materializes the entire
`[seq, seq]` score matrix in HBM — memory and bandwidth grow **quadratically**
with sequence length. FlashAttention instead streams blocks of K/V through the
chip's small, fast VMEM and keeps only running softmax statistics, so HBM traffic
and peak memory grow roughly **linearly**. The speedup widens as `seq_len` grows.

## What's in here

| Path | What it is |
|------|------------|
| `src/pallas_flash/flash_attention.py` | The Pallas kernels: forward, backward (`custom_vjp`), and `flash_attention_decode` |
| `src/pallas_flash/fused_rope.py` | Attention with RoPE applied to Q/K *inside* the kernel |
| `src/pallas_flash/paged.py` | Paged attention: block-table KV cache via scalar prefetch |
| `src/pallas_flash/reference.py` | Naive `jnp` attention — correctness oracle **and** XLA baseline |
| `src/pallas_flash/model.py` | Tiny LLaMA: RMSNorm, RoPE, SwiGLU, decoder block, `generate()`, fixed-cache `generate_cached()` |
| `src/pallas_flash/config.py` | `ModelConfig` dataclass (incl. `n_kv_heads` for GQA) |
| `tests/` | `interpret=True` correctness + model smoke tests (run on CPU) |
| `benchmarks/benchmark_attention.py` | Pallas vs XLA attention timing table |
| `benchmarks/benchmark_decode.py` | KV-cache vs full-recompute decoding timing table |
| `benchmarks/benchmark_backward.py` | Forward vs forward+backward timing, Pallas vs XLA |
| `scripts/run_inference.py` | End-to-end tiny-LLaMA greedy generation demo |
| `notebooks/kaggle_tpu_flash_attention.ipynb` | Self-contained Kaggle TPU v3-8 notebook |

## How the kernel works

Grid layout: `(batch, num_heads, num_q_blocks, num_kv_blocks)`, executed
sequentially in lexicographic order.

1. **Manual HBM → VMEM mapping.** Each `pl.BlockSpec` `index_map` says exactly
   which block of the big HBM array to stream into VMEM for a given grid point.
   This is the lever a high-level Keras/PyTorch program never touches.
2. **Online softmax.** For each query block we loop over key/value blocks and
   keep running statistics in VMEM scratch: row max `m`, denominator `l`, and the
   weighted value accumulator `acc`. Each new block rescales the old accumulator
   by `exp(m_old - m_new)` and adds its contribution — mathematically identical to
   a full softmax, but the `[seq, seq]` matrix never exists.
3. **Causal block skipping.** With `causal=True`, key blocks entirely in the
   future of a query block are skipped — no load, no matmul.
4. **Grouped-query attention (GQA/MQA).** `k`/`v` may have fewer heads than `q`
   (`num_heads` a multiple of `num_kv_heads`). Query head `h` reads KV head
   `h // (num_heads // num_kv_heads)` — expressed purely in the K/V `BlockSpec`
   index map, so the KV cache is never physically replicated.
5. **Fixed-size KV cache (`kv_len`).** Only the first `kv_len` positions are
   attended, and kv blocks starting past it are skipped outright — so cost tracks
   the *filled* length of a preallocated cache, not its capacity.
6. **bfloat16.** Inputs are up-cast to float32 inside VMEM, so the online softmax
   and `p @ v` accumulate in float32 while HBM traffic halves. Outputs come back
   in the input dtype.
7. **Fused RoPE.** `flash_attention_rope` rotates the Q and K blocks while they
   are already in VMEM, so the rotated copies never touch HBM.
8. **Paged attention.** `paged_flash_attention` gathers KV blocks through a
   runtime block table held in scalar-prefetch memory.

The *same* source runs on CPU via `interpret=True` (used by the tests) and
compiles to TPU with `interpret=False`.

## Backward pass

`flash_attention` is a `jax.custom_vjp`, so it works with `jax.grad` /
`jax.value_and_grad` like any JAX function — and its backward pass is **two more
Pallas kernels**, not an XLA fallback. The forward kernel saves the per-row
log-sum-exp (a `[batch, heads, seq, 1]` residual), which lets the backward
recompute `p = exp(s - lse)` tile-by-tile instead of storing the score matrix:

```
delta_i = Σ_d o_id · do_id           (cheap elementwise "preprocess")
dv_j    = Σ_i p_ij · do_i
ds_ij   = p_ij (do_i·v_j − delta_i)
dq_i    = scale · Σ_j ds_ij · k_j    kernel 1: grid (b, h, q_block, kv_block)
dk_j    = scale · Σ_i ds_ij · q_i    kernel 2: grid (b, h, kv_block, q_block)
```

dQ needs an outer loop over query blocks and dK/dV over kv blocks, so they are
two kernels with transposed grids. For GQA, dK/dV are produced per *query* head
and the group is summed afterwards, keeping cross-head reductions out of the
kernel. Gradients match JAX autodiff through the reference attention to ~1e-6
relative (see `tests/test_backward.py`).

```python
loss = lambda q, k, v: jnp.sum(flash_attention(q, k, v, causal=True))
dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
```

## Quick start

### Local / CPU (correctness, no TPU needed)

```bash
pip install -e .            # or: pip install -r requirements.txt
pytest -q                   # runs the kernel + model tests in interpret mode
python scripts/run_inference.py
```

```python
import jax, jax.numpy as jnp
from pallas_flash import flash_attention

q = k = v = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 512, 128))
out = flash_attention(q, k, v, causal=True, interpret=True)  # interpret=False on TPU
```

### Kaggle TPU v3-8

1. New Kaggle notebook → **Accelerator → TPU VM v3-8**.
2. Upload / open `notebooks/kaggle_tpu_flash_attention.ipynb` and **Run all**.

It checks `jax.devices()`, defines the kernel, verifies correctness against the
reference, prints a Pallas-vs-XLA benchmark table, and runs the tiny LLaMA.

## Benchmark

`python benchmarks/benchmark_attention.py` prints a table like:

```
 seq_len |   XLA (ms) |  Pallas (ms) |  speedup
 ...
```

Run it **on a TPU** for meaningful numbers — on CPU (interpret mode) the kernel
is intentionally un-optimized and only confirms the code path executes. On a
v3-8 the Pallas kernel pulls ahead of the XLA baseline as `seq_len` increases.

## Fused RoPE

Applying RoPE the ordinary way costs two extra full round-trips of Q and K
through HBM — read, rotate, write, read back — for no extra math. Since the
kernel already streams Q and K blocks into VMEM, `flash_attention_rope` rotates
them *there*:

```python
from pallas_flash import flash_attention_rope
out = flash_attention_rope(q, k, v, cos, sin, causal=True)   # q, k are un-rotated
```

The cos/sin tables are ordinary inputs with their own `BlockSpec`s, so each grid
step pulls in only the rows it needs. Query and key tables are separate, which
also covers decoding (one query at position `pos`, keys spanning `[0, pos]`).
Numerically identical to `flash_attention(apply_rope(q), apply_rope(k), v)`.

This is a forward-only kernel; for training, use `flash_attention` with RoPE
applied outside.

## Paged attention

A contiguous cache forces you to preallocate `max_seq_len` for every sequence in
a batch, so a batch of mostly-short sequences wastes most of what it reserves.
Paged attention keeps one global pool of fixed-size **pages** plus a per-sequence
**block table** mapping logical → physical pages, so memory tracks actual length
and pages can be shared (a common prompt prefix) or recycled.

The interesting part on TPU is *how the kernel finds a page*. A Pallas
`index_map` must be a pure function of the grid indices — it cannot read a normal
input array. The block table therefore lives in **scalar-prefetch** memory
(`pltpu.PrefetchScalarGridSpec`), which index maps *can* read:

```python
def k_index_map(b, h, p, block_tables_ref, context_lens_ref):
    return (block_tables_ref[b, p], h // q_per_kv, 0, 0)
```

The DMA that streams a page from HBM into VMEM is addressed by a runtime table
lookup — fully dynamic, data-dependent gathering expressed as *data movement*
rather than compute.

```python
from pallas_flash import paged_flash_attention
out = paged_flash_attention(q, k_pages, v_pages, block_tables, context_lens)
```

Layout: `k_pages/v_pages [num_pages, n_kv_heads, page_size, head_dim]`,
`block_tables [batch, pages_per_seq]`, `context_lens [batch]`. Pages entirely
past a sequence's context are skipped, so a short sequence in the batch does no
work for a long one's pages.

## Streaming with a ring-buffer cache

The fixed-size cache still needs capacity == total length, so it cannot stream
forever. A ring buffer of capacity `C` writes the token at position `pos` into
slot `pos % C`, evicting the token from `C` steps ago — generation then runs
indefinitely in **constant memory** (sliding-window attention):

```python
from pallas_flash import generate_streaming
tokens = generate_streaming(params, prompt_ids, max_new_tokens=1000, cfg=cfg, window=256)
```

Two facts make this work with no extra masking: RoPE is applied *before* caching,
so each entry carries its absolute position; and attention is permutation-invariant
over keys, so the ring's out-of-order layout is irrelevant. Once wrapped, every
slot is live and `kv_len` is simply `C`. With `window >= prompt + new_tokens`
nothing is evicted and the output matches `generate_cached` exactly.

## KV-cache decoding

`generate()` re-runs the whole prefix every step (O(T²) work to emit T tokens).
`generate_cached()` allocates a **fixed-capacity** per-layer K/V cache once,
writes each new token's K/V in place, and calls the dedicated single-query
`flash_attention_decode` kernel — so there is no per-step reallocation and no
growing concatenate:

```python
from pallas_flash import ModelConfig, init_params, generate_cached
cfg = ModelConfig(n_heads=4, n_kv_heads=2, dim=512)   # grouped-query attention
params = init_params(jax.random.PRNGKey(0), cfg)
tokens = generate_cached(params, prompt_ids, max_new_tokens=32, cfg=cfg)
```

`generate_cached` produces the **exact same tokens** as `generate` (verified in
the test-suite) while doing far less work per step. Benchmark it with
`python benchmarks/benchmark_decode.py`.

## Notes & non-goals

- **Untrained model** — parameters are random; the LLaMA exists to exercise the
  kernel on a realistic inference path, not to produce meaningful text.
- **KV cache** — three flavours: growing-free fixed capacity
  (`generate_cached`), sliding-window ring buffer (`generate_streaming`), and
  paged (`paged_flash_attention`).
- **Sliding-window quality** — evicting old tokens changes model outputs by
  design; no attention-sink/StreamingLLM trick is implemented.
- **Training** — gradients are implemented and tested, but there is no optimizer
  or training loop here; the model stays randomly initialized.
- **Fully manual DMA** — `BlockSpec` already pipelines HBM↔VMEM copies; for
  hand-rolled control you can keep K/V in `pltpu.ANY` and drive
  `pltpu.make_async_copy` yourself (sketched in the notebook's notes).

## Requirements

JAX (CPU build for local testing, `jax[tpu]` on Kaggle/Cloud TPU) and NumPy. See
`requirements.txt` / `pyproject.toml`.
