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

import math
from functools import partial
from typing import List

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import linen as nn

import compscale.data.base as data_base
import compscale.data.utils as data_utils


class HyperTeacher(data_base.DatasetGenerator):
  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int],
    use_bias: bool,
    num_modules: int,
    num_hot: int,
    task_distribution: str,
    frac_ood: float,
    task_support: str,
    scale: float,
    seed: int,
  ) -> None:
    super().__init__(input_shape=(input_dim,), output_dim=output_dim)

    self.num_modules = num_modules
    self.num_hot = num_hot
    self.task_distribution = task_distribution
    self.frac_ood = frac_ood
    self.task_support = task_support
    self.seed = seed
    self.hidden_dims = hidden_dims
    self.num_layers = len(hidden_dims)

    # Split the RNG key for different parameter initializations
    keys = jax.random.split(jax.random.key(seed), 2 + self.num_layers)
    self.rng = keys[0]
    rng_readout = keys[1]
    rng_modules = keys[2:]

    # Initialize parameters for each layer
    self.modules = []
    self.module_biases = []

    # First layer: input_dim to first hidden_dim
    init_fn_first = nn.initializers.variance_scaling(
      scale=scale, mode='fan_in', distribution='truncated_normal', batch_axis=(0,)
    )
    self.modules.append(
      init_fn_first(rng_modules[0], shape=(num_modules, input_dim, hidden_dims[0]))
    )

    if use_bias:
      self.module_biases.append(
        jax.random.uniform(
          rng_modules[0], shape=(num_modules, hidden_dims[0]), maxval=0.5
        )
      )
    else:
      self.module_biases.append(jnp.zeros(shape=(num_modules, hidden_dims[0])))

    # Hidden layers
    for i in range(1, self.num_layers):
      init_fn_layer = nn.initializers.variance_scaling(
        scale=scale, mode='fan_in', distribution='truncated_normal', batch_axis=(0,)
      )
      self.modules.append(
        init_fn_layer(
          rng_modules[i], shape=(num_modules, hidden_dims[i - 1], hidden_dims[i])
        )
      )

      if use_bias:
        self.module_biases.append(
          jax.random.uniform(
            rng_modules[i], shape=(num_modules, hidden_dims[i]), maxval=0.5
          )
        )
      else:
        self.module_biases.append(jnp.zeros(shape=(num_modules, hidden_dims[i])))

    # Fixed readout shared by all tasks
    init_fn_readout = nn.initializers.variance_scaling(
      scale=scale,
      mode='fan_in',
      distribution='truncated_normal',
    )
    self.readout = init_fn_readout(rng_readout, shape=(hidden_dims[-1], self.output_dim))

    # Generate all possible module combinations
    module_combinations = data_utils.make_latents(self.num_modules, self.num_hot)

    # Split into in-distribution and out-of-distribution sets
    self.combinations_in_dist, self.combinations_out_dist = (
      data_utils.split_module_combinations(
        combinations_all=module_combinations,
        task_support=self.task_support,
        num_modules=self.num_modules,
        num_hot=self.num_hot,
        frac_ood=self.frac_ood,
        rng=self.rng,
      )
    )

  def sample_latents(self, rng: chex.PRNGKey, num_tasks: int, mode: str) -> jax.Array:
    rng_choice, rng_weights = jax.random.split(rng)

    match mode:
      case 'train' | 'test':
        combinations_task = self.combinations_in_dist
        latents = jax.random.choice(rng_choice, combinations_task, shape=(num_tasks,))
      case 'ood':
        combinations_task = self.combinations_out_dist
        latents = jax.random.choice(rng_choice, combinations_task, shape=(num_tasks,))
      case str() if mode.startswith('ood_'):
        hotness = int(mode.split('_')[1])
        if hotness <= self.num_hot:
          # Filter the existing combinations_out_dist for the given hotness
          mask = jnp.all(jnp.sum(self.combinations_out_dist, axis=-1) == hotness, axis=-1)
          latents = jax.random.choice(
            key=rng_choice,
            a=self.combinations_out_dist,
            p=1.0 * mask,
            shape=(num_tasks,),
          )
        elif hotness <= self.num_modules:
          # Randomly sample modules - everything is ood here
          @jax.vmap
          def sample_k_modules(rng):
            return jax.random.choice(
              rng, self.num_modules, replace=False, shape=(hotness,)
            )

          latents_indeces = sample_k_modules(jax.random.split(rng, num_tasks))
          latents = data_utils.k_hot(latents_indeces, self.num_modules)
        else:
          raise ValueError(f'Invalid hotness {hotness}')
      case _:
        raise ValueError(f'Unknown mode: {mode}.')

    match self.task_distribution:
      case 'khot':
        pass
      case 'continuous':
        # Sample weights uniformly from simplex (see Willms, 2021)
        weights = jax.random.exponential(rng_weights, shape=latents.shape)
        weights = weights * latents
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1)

        # Shift nonzero embeddings to the range [0.5, 1.0] to prevent further sparsity
        latents = (0.5 * weights + 0.5) * latents
      case 'discrete':
        weights = jax.random.choice(
          rng_weights,
          jnp.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
          shape=latents.shape,
        )
        latents = weights * latents
      case _:
        raise ValueError(f'Unknown task_distribution: {self.task_distribution}.')

    return latents

  def forward_network(self, inputs: jax.Array, latent: jax.Array) -> jax.Array:
    """Forward pass through the multi-layer network with a given latent."""
    # Scale the latent to keep invariance to num_hot
    latent = latent / math.sqrt(self.num_hot)

    # First layer
    w = jnp.einsum('mih,m->ih', self.modules[0], latent)
    b = jnp.einsum('mh,m->h', self.module_biases[0], latent)
    activations = jax.nn.relu(jnp.einsum('ih,bi->bh', w, inputs) + jnp.expand_dims(b, axis=0))

    # Hidden layers
    for i in range(1, self.num_layers):
      w = jnp.einsum('mih,m->ih', self.modules[i], latent)
      b = jnp.einsum('mh,m->h', self.module_biases[i], latent)
      activations = jax.nn.relu(jnp.einsum('ih,bi->bh', w, activations) + jnp.expand_dims(b, axis=0))

    # Output layer
    outputs = jnp.einsum('ho,bh->bo', self.readout, activations)

    return outputs

  @partial(jax.jit, static_argnames=('self', 'num_tasks', 'num_samples', 'mode'))
  def sample(
    self, rng: chex.PRNGKey, num_tasks: int, num_samples: int, mode: str
  ) -> tuple[
    jt.Float[jt.Array, 'num_tasks num_samples input_dim'],
    jt.Float[jt.Array, 'num_tasks num_samples output_dim'],
    jt.Float[jt.Array, 'num_tasks num_modules'],
  ]:
    rng_tasks, rng_samples = jax.random.split(rng)
    rngs_samples = jax.random.split(rng_samples, num_tasks)
    latents = self.sample_latents(rng_tasks, num_tasks, mode)

    @partial(jax.vmap, in_axes=(0, 0, None))
    def forward(rng: chex.PRNGKey, latent: jt.Float[jt.Array, ' k'], num_samples: int):
      # x-support in [-1, 1]
      inputs = jax.random.uniform(rng, (num_samples, *self.input_shape), minval=-1)

      # Forward pass through the multi-layer network
      targets = self.forward_network(inputs, latent)

      return inputs, targets

    inputs, targets = forward(rngs_samples, latents, num_samples)

    return inputs, targets, latents

  @partial(jax.jit, static_argnames=('self', 'num_samples'))
  def sample_from_latent(
    self, rng: chex.PRNGKey, latent: jax.Array, num_samples: int
  ) -> tuple[
    jax.Array,
    jax.Array,
  ]:
    """
    Sample input-output examples using a specific latent vector.

    Args:
        rng: Random key for generating inputs
        latent: A specific latent vector to use (shape: num_modules)
        num_samples: Number of examples to generate

    Returns:
        inputs: Input examples (shape: num_samples, input_dim)
        targets: Target outputs (shape: num_samples, output_dim)
    """
    # x-support in [-1, 1]
    inputs = jax.random.uniform(rng, (num_samples, *self.input_shape), minval=-1)

    # Forward pass through the multi-layer network
    targets = self.forward_network(inputs, latent)

    return inputs, targets


def create_hyperteacher_dataloader(
  batch_size: int,
  num_train: int,
  num_test: int,
  num_ood: int,
  *,
  latent_encoding: str,
  num_samples: int,
  input_dim: int,
  output_dim: int,
  hidden_dims: tuple[int],
  use_bias: bool,
  num_modules: int,
  num_hot: int,
  task_distribution: str,
  frac_ood: float,
  task_support: str,
  scale: float,
  seed: int,
) -> tuple[data_base.Dataloader, dict[str, data_base.Dataloader]]:

  rng = jax.random.key(seed)
  data_generator = HyperTeacher(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dims=hidden_dims,
    use_bias=use_bias,
    num_modules=num_modules,
    num_hot=num_hot,
    frac_ood=frac_ood,
    scale=scale,
    task_distribution=task_distribution,
    task_support=task_support,
    seed=seed,
  )

  # Get the base encoding function
  encode_latent_base = data_utils.make_encode_latent(
    rng, latent_encoding, num_modules, data_generator
  )

  @partial(jax.jit, static_argnames=['mode'])
  def sample_fn(rng: chex.PRNGKey, mode: str) -> data_base.Dataset:
    # Split the RNG for main sampling and few-shot examples
    rng_main, rng_encode = jax.random.split(rng)

    # Generate main batch of data
    inputs, targets, latents = data_generator.sample(
      rng_main, num_tasks=batch_size, num_samples=num_samples, mode=mode
    )

    # Encode latents
    rngs_encode = jax.random.split(rng_encode, batch_size)
    encodings = jax.vmap(encode_latent_base)(latents, rngs_encode)

    # MSE for predicting the mean for each tasks output unit (to compute r2 )
    base_mse = (targets - jnp.mean(targets, axis=1, keepdims=True)) ** 2
    # Mean over samples per task, per output unit
    base_mse = jnp.mean(base_mse, axis=1, keepdims=True)

    return data_base.Dataset(
      x=inputs,
      y=targets,
      z=encodings,
      mask=jnp.ones_like(targets),
      info=dict(latents=latents, base_mse=base_mse),
    )

  # Create main train/test/ood loaders
  trainloader = data_base.SyntheticDataloader(
    num_tasks=num_train,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='train'),
    output_dim=output_dim,
    seed=seed,
  )

  testloader = data_base.SyntheticDataloader(
    num_tasks=num_test,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='test'),
    output_dim=output_dim,
    seed=seed,
  )

  oodloader = data_base.SyntheticDataloader(
    num_tasks=num_ood,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='ood'),
    output_dim=output_dim,
    seed=seed,
  )

  return (
    trainloader,
    dict(
      test=testloader,
      ood=oodloader,
    ),
  )
