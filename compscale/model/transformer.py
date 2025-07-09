# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer Decoder-only model."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from compscale import config_classes
from compscale.model import fsdp
from compscale.model import rotary


class Transformer(nn.Module):
  """Transformer decoder-only."""

  cfg: config_classes.ModelConfig

  def setup(self) -> None:
    cfg = self.cfg
    self.embed = nn.Embed(
      num_embeddings=cfg.vocab_size,
      features=cfg.d_model,
      embedding_init=fsdp.init('embedding', use_fsdp=cfg.fsdp),
    )
    block = nn.remat(TBlock) if cfg.remat else TBlock
    self.blocks = [block(cfg) for _ in range(cfg.n_layers)]
    self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)

  def __call__(self, y_BxL: jax.Array) -> jax.Array:
    # For training on concatenated examples.
    if jnp.issubdtype(y_BxL.dtype, jnp.integer):
      y_BxLxD = self.embed(y_BxL)  # [B, L, d_model]
    else:
      if y_BxL.ndim != 3 or y_BxL.shape[-1] != self.cfg.d_model:
        raise ValueError(
          'Expected either int32 tokens [B, L] or embeddings [B, L, d_model]; '
          f'got {y_BxL.shape}'
        )
      y_BxLxD = y_BxL  # already embedded
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD)
    y_BxLxD = self.out_ln(y_BxLxD)
    # logits_BxLxV = self.embed.attend(y_BxLxD.astype(jnp.float32))
    # return logits_BxLxV
    return y_BxLxD


class Mlp(nn.Module):
  """Multilayer perceptron."""

  cfg: config_classes.ModelConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array) -> jax.Array:
    cfg = self.cfg
    linear = partial(
      nn.Dense,
      kernel_init=fsdp.init('mlp_kernel', use_fsdp=cfg.fsdp),
      use_bias=False,
      dtype=cfg.dtype,
    )
    x_BxLxF = linear(cfg.d_ffw)(x_BxLxD)
    x_BxLxF = jax.nn.gelu(x_BxLxF)
    x_BxLxD = linear(cfg.d_model)(x_BxLxF)
    return x_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""

  cfg: config_classes.ModelConfig

  @nn.compact
  def __call__(self, in_BxLxD: jax.Array) -> jax.Array:
    cfg = self.cfg

    # "pre-layernorm"
    x_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(in_BxLxD)
    x_BxLxD = CausalAttn(cfg)(x_BxLxD)
    x_BxLxD += in_BxLxD

    z_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(x_BxLxD)
    z_BxLxD = Mlp(cfg)(z_BxLxD)

    return x_BxLxD + z_BxLxD


class CausalAttn(nn.Module):
  """Causal attention layer."""

  cfg: config_classes.ModelConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array) -> jax.Array:
    cfg = self.cfg

    if not cfg.d_model % cfg.n_heads == 0:
      raise AssertionError(f'd_model{cfg.d_model} not divisible by n_heads {cfg.n_heads}')

    Dh = cfg.d_model // cfg.n_heads

    # Maps d_model -> (n_heads, Dh)
    multilinear = partial(
      nn.DenseGeneral,
      axis=-1,
      features=(cfg.n_heads, Dh),
      kernel_init=fsdp.init('attn_in_proj', use_fsdp=cfg.fsdp),
      use_bias=False,
      dtype=cfg.dtype,
    )

    q_BxLxHxDh, k_BxLxHxDh, v_BxLxHxDh = (
      multilinear(name='query')(x_BxLxD),
      multilinear(name='key')(x_BxLxD),
      multilinear(name='value')(x_BxLxD),
    )

    q_BxLxHxDh = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(q_BxLxHxDh)
    k_BxLxHxDh = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(k_BxLxHxDh)

    context_size = max(q_BxLxHxDh.shape[1], k_BxLxHxDh.shape[1])
    sin, cos = rotary.generate_fixed_pos_embedding(Dh, context_size)
    sin = sin.astype(cfg.dtype)
    cos = cos.astype(cfg.dtype)
    q_BxLxHxDh, k_BxLxHxDh = rotary.apply_rotary_embedding(
      q_BxLxHxDh, k_BxLxHxDh, cos, sin
    )

    q_BxLxHxDh /= Dh**0.5
    att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)

    # causal attention mask
    context_size = x_BxLxD.shape[1]
    _NEG_INF = jnp.finfo(cfg.dtype).min
    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, context_size, context_size), dtype=jnp.bool_))
    att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)

    # softmax
    att_BxHxLxL = att_BxHxLxL.astype(jnp.float32)  # cast to fp32 for softmax
    att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
    att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)

    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)

    # Output projection followed by contraction back to original dims
    out_BxLxD = nn.DenseGeneral(
      features=cfg.d_model,
      name='attn_out_proj',
      axis=(-2, -1),
      kernel_init=fsdp.init('attn_out_proj', use_fsdp=cfg.fsdp),
      use_bias=False,
      dtype=cfg.dtype,
    )(out_BxLxHxDh)
    return out_BxLxD
