"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn

from compscale import config_classes
from compscale.data.base import Dataset
from compscale.model import hypernetwork
from compscale.model import transformer as transformer_lib


class HypernetworkPolicy(nn.Module):
  cfg: config_classes.ModelConfig

  @nn.compact
  def __call__(self, batch: Dataset) -> jax.Array:
    cfg = self.cfg
    target_network = hypernetwork.VarianceScaledMLP(
      output_dim=cfg.d_out,
      hidden_dim=cfg.d_ffw,
      num_hidden=1,
    )

    def weight_generator(features: int) -> nn.Module:
      return nn.Sequential((nn.LayerNorm(), nn.Dense(features=features)))

    hnet = hypernetwork.Hypernetwork(
      input_shape=batch.x.shape,
      target_network=target_network,
      weight_generator=weight_generator,
    )
    return hnet(batch.z, batch.x)


class MultilayerPerceptronPolicy(nn.Module):
  cfg: config_classes.ModelConfig

  def setup(self) -> None:
    d_mlp = self.cfg.d_mlp
    d_out = self.cfg.d_out
    num_layers = self.cfg.num_hidden_layers

    layers = []
    for _ in range(num_layers):
      layers.append(nn.Dense(d_mlp))
      layers.append(nn.relu)
    layers.append(nn.Dense(d_out))

    self.policy = nn.Sequential(tuple(layers))

  def __call__(self, batch: Dataset) -> jax.Array:
    embed = einops.repeat(batch.z, 'b z -> b t z', t=batch.x.shape[1])
    policy_input = jnp.concatenate((embed, batch.x), axis=2)
    return self.policy(policy_input)


class TransformerPolicy(nn.Module):
  """
  Decoder-only transformer that concatenates the latent and each input sample
  into tokens and returns one prediction per sample.
  """

  cfg: config_classes.ModelConfig

  @nn.compact
  def __call__(self, batch: Dataset) -> jax.Array:
    z_emb = nn.Dense(self.cfg.d_model, use_bias=False)(batch.z[:, None, :])
    x_emb = nn.Dense(self.cfg.d_model)(batch.x)
    seq = jnp.concatenate([z_emb, x_emb], axis=1)  # [B, T_lat + T_x, D]
    h = transformer_lib.Transformer(self.cfg)(seq)
    h_x = h[:, z_emb.shape[1] :, :]  # [B, T_x, D]
    return nn.Dense(features=self.cfg.d_out)(h_x)  # [B, T_x, d_out]
