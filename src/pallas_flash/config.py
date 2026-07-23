"""Configuration for the tiny LLaMA model and the FlashAttention kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the tiny LLaMA decoder.

    The defaults describe a deliberately small, randomly-initialized model that
    fits comfortably on a single TPU v3 core and is meant for demonstrating the
    custom Pallas attention kernel on the inference path (no training).
    """

    vocab_size: int = 256
    dim: int = 256          # model / embedding dimension
    n_layers: int = 4
    n_heads: int = 2        # number of query heads
    # Number of key/value heads. If < n_heads (and dividing it) the model uses
    # grouped-query attention (GQA); n_kv_heads == 1 is multi-query attention
    # (MQA). Defaults to n_heads (plain multi-head attention) when left as 0.
    n_kv_heads: int = 0
    # head_dim is fixed at 128 so the attention block tiling lands on the TPU's
    # native (8, 128) vector-register shape; dim must equal n_heads * head_dim.
    head_dim: int = 128
    ffn_hidden: int = 688   # SwiGLU hidden size (~ 8/3 * dim, rounded to mult. of 16)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # FlashAttention tiling. Block sizes should be multiples of 8 (rows) and the
    # head_dim should be a multiple of 128 for the TPU vector units.
    block_q: int = 128
    block_k: int = 128

    def __post_init__(self) -> None:
        if self.n_kv_heads == 0:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        if self.dim != self.n_heads * self.head_dim:
            raise ValueError(
                f"dim ({self.dim}) must equal n_heads * head_dim "
                f"({self.n_heads} * {self.head_dim} = {self.n_heads * self.head_dim})"
            )
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads "
                f"({self.n_kv_heads})"
            )

    @property
    def kv_dim(self) -> int:
        """Total width of the key/value projections."""
        return self.n_kv_heads * self.head_dim
