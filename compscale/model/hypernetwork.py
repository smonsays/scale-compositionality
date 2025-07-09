"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the “Software”), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from functools import partial
from typing import Any
from typing import Callable
from typing import Type

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact
from jax import lax
from jax.flatten_util import ravel_pytree


class Hypernetwork(nn.Module):
  """
  Hypernetwork nn.Module for arbitrary target networks and weight generators.
  """

  input_shape: tuple[int]
  target_network: nn.Module
  weight_generator: Type[nn.Module]
  rng_collection_names: tuple[str, ...] = ()

  def get_target_apply(
    self, module: nn.Module, input_shape: tuple[int]
  ) -> tuple[Callable, int]:
    init_rngs = {key: jax.random.key(0) for key in ('params', *self.rng_collection_names)}
    variables = module.init(init_rngs, jnp.empty(input_shape))

    if len(variables) != 1:
      raise NotImplementedError(
        'Not implemented for stateful models (e.g. using batch norm)'
      )

    params_flat, unflatten_fn = ravel_pytree(variables['params'])
    num_params = len(params_flat)

    @partial(jnp.vectorize, signature='(n),(s,d),()->(s,r)')
    def apply_fn(
      params_flat: jax.Array, input: jax.Array, rng: chex.PRNGKey
    ) -> jax.Array:
      variables = {'params': unflatten_fn(params_flat)}
      rngs = dict(
        zip(
          self.rng_collection_names, jax.random.split(rng, len(self.rng_collection_names))
        )
      )

      return module.apply(variables, input, rngs=rngs)

    return apply_fn, num_params

  def setup(self) -> None:
    self.apply_fn, self.num_params = self.get_target_apply(
      module=self.target_network,
      input_shape=self.input_shape,
    )
    # NOTE: Assuming weight generator output_dim defined as attribute `features`
    self.weight_generator_ = self.weight_generator(features=self.num_params)

  def __call__(self, latent: jax.Array, input: jax.Array) -> jax.Array:
    params_flat = self.weight_generator_(latent)
    rng = self.make_rng('target')  # give target network separate rng stream

    return self.apply_fn(params_flat, input, rng)


class VarianceScaledKernel(nn.Module):
  """Kernel with variance scaling during forward instead of at initialization."""

  input_dim: int
  output_dim: int
  distribution: str = 'truncated_normal'
  scale: float = 1.0
  mode: str = 'fan_in'
  param_dtype: Any = jnp.float32

  def setup(self) -> None:
    # Compute normalizer to be applied during forward pass instead of using it for init
    match self.mode:
      case 'fan_in':
        denominator = self.input_dim
      case 'fan_out':
        denominator = self.output_dim
      case 'fan_avg':
        denominator = (self.input_dim + self.output_dim) / 2
      case _:
        raise ValueError(f'invalid mode for variance scaling initializer: {self.mode}')
    variance = jnp.array(self.scale / denominator, dtype=self.param_dtype)

    # Initialize unnormalized kernel
    assert jnp.issubdtype(
      self.param_dtype, jnp.floating
    ), 'No support for complex numbers.'

    if self.distribution == 'truncated_normal':

      def init_fn(key: chex.PRNGKey, shape: tuple[int], dtype: jnp.dtype) -> jax.Array:
        return jax.random.truncated_normal(key, -2, 2, shape, dtype)

      # constant is stddev of standard normal truncated to (-2, 2)
      self.stddev = jnp.sqrt(variance) / jnp.array(0.87962566103423978, self.param_dtype)
    elif self.distribution == 'normal':
      init_fn = jax.random.normal
      self.stddev = jnp.sqrt(variance)
    elif self.distribution == 'uniform':
      init_fn = partial(jax.random.uniform, minval=-1)
      self.stddev = jnp.sqrt(3 * variance)
    else:
      raise ValueError(f'invalid mode for variance scaling initializer: {self.mode}')

    self.kernel = self.param(
      'unnormalized_kernel',
      init_fn,
      (self.input_dim, self.output_dim),
      self.param_dtype,
    )

  def __call__(self) -> jax.Array:
    # Normalize kernel
    return self.kernel * self.stddev


class VarianceScaledDense(nn.Dense):
  @compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    kernel = VarianceScaledKernel(
      input_dim=inputs.shape[-1], output_dim=self.features, param_dtype=self.param_dtype
    )()
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    y = lax.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class VarianceScaledMLP(nn.Module):
  output_dim: int
  hidden_dim: int
  num_hidden: int
  nonlinearity: Callable = nn.relu

  @compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    x = inputs
    for i in range(self.num_hidden):
      x = VarianceScaledDense(self.hidden_dim, name=f'norm_dense{i}')(x)
      x = self.nonlinearity(x)

    x = VarianceScaledDense(self.output_dim, name='norm_dense_out')(x)

    return x
