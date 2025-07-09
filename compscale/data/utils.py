"""
Copyright (c) Florian Redhardt, Simon Schug
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

import itertools
import logging
import math
from functools import partial
from typing import Any
from typing import Callable
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


@partial(jnp.vectorize, signature='(n)->(k)', excluded=(1,))
def k_hot(ind: jt.Int[jt.Array, ' n'], length: int) -> jt.Float[jt.Array, ' k']:
  """
  Convert a vector of indeces to a k-hot vector.
  Repeating an index does not change the result.
  """
  return (jnp.sum(jax.nn.one_hot(ind, length), axis=0) > 0) * 1.0


def check_compositional(experts: jt.Array) -> jt.Array:
  """Check if every expert exists at least once in the task combinations."""
  return jnp.all(jnp.sum(experts, axis=0) > 0)


def check_connected(experts):
  """Check whether the support is connected."""
  num_experts = experts.shape[1]
  binary_experts = (experts > 0).astype(jnp.int32)
  task_expert_matrix = binary_experts
  expert_task_matrix = task_expert_matrix.T
  adjacency_matrix = jnp.matmul(expert_task_matrix, task_expert_matrix)
  adjacency_matrix = jnp.where(adjacency_matrix > 0, 1, 0)

  for _ in range(math.ceil(math.log2(num_experts))):
    adjacency_matrix = jnp.matmul(adjacency_matrix, adjacency_matrix)
    adjacency_matrix = jnp.where(adjacency_matrix > 0, 1, 0)

  return jnp.all(adjacency_matrix > 0)


def create_coupling_layer(
  dim: int, hidden_dims: tuple[int] = (64, 64), scale: float = 0.1
) -> Tuple[Callable, list]:
  """
  Creates a coupling layer for the invertible neural network.

  Args:
      dim: Input dimension
      hidden_dims: Dimensions of hidden layers
      scale: Scale factor for initialization

  Returns:
      Tuple of (forward function, parameters)
  """
  # Split dimension for coupling
  split_dim = dim // 2

  def init_params(key: chex.PRNGKey) -> list:
    """Initialize network parameters."""
    keys = jax.random.split(key, len(hidden_dims) + 1)
    dims = [split_dim, *hidden_dims, split_dim * 2]  # *2 for scale and shift
    params = []
    for in_dim, out_dim, k in zip(dims[:-1], dims[1:], keys, strict=False):
      w = jax.random.normal(k, (in_dim, out_dim)) * scale
      b = jnp.zeros(out_dim)
      params.append((w, b))
    return params

  def nn_forward(x: jnp.ndarray, params: list) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Neural network that outputs scale and translation."""
    h = x
    for w, b in params[:-1]:
      h = jnp.dot(h, w) + b
      h = jax.nn.sigmoid(h)
    w, b = params[-1]
    h = jnp.dot(h, w) + b

    # Output scale and translation
    log_scale, t = jnp.split(h, 2)
    # Constrain scale to prevent explosion/vanishing
    s = jnp.tanh(log_scale) * 0.5 + 1.0
    return s, t

  def forward(x: jnp.ndarray, params: list) -> jnp.ndarray:
    """Forward pass through the coupling layer."""
    x1, x2 = jnp.split(x, 2)
    s, t = nn_forward(x1, params)
    y2 = x2 * s + t
    return jnp.concatenate([x1, y2])

  return forward, init_params


def create_invertible_mlp(
  dim: int, num_layers: int = 3, hidden_factor: float = 2.0, scale: float = 0.1
) -> Tuple[Callable, list]:
  """
  Creates an invertible MLP using coupling layers.

  Args:
      dim: Input dimension
      num_layers: Number of coupling layers
      hidden_factor: Multiplier for hidden layer size
      scale: Scale for weight initialization

  Returns:
      Tuple of (forward function, parameters)
  """
  hidden_dims = (int(dim * hidden_factor),) * 2

  def init_params(key: chex.PRNGKey) -> list:
    """Initialize parameters for all layers."""
    keys = jax.random.split(key, num_layers)
    all_params = []
    for k in keys:
      forward, init_fn = create_coupling_layer(dim, hidden_dims, scale)
      layer_params = init_fn(k)
      all_params.append(layer_params)
    return all_params

  def forward(x: jnp.ndarray, params: list) -> jnp.ndarray:
    """Forward pass through the entire network."""
    y = x
    for i, layer_params in enumerate(params):
      # Alternate between splitting first/second half
      if i % 2 == 1:
        y = y[::-1]  # Reverse elements
      fwd, _ = create_coupling_layer(dim, hidden_dims, scale)
      y = fwd(y, layer_params)
      if i % 2 == 1:
        y = y[::-1]  # Reverse back
    return y

  return forward, init_params


def make_encode_latent(
  rng: chex.PRNGKey,
  encoding_type: str,
  num_modules: int,
  data_generator: Any | None = None,
) -> Callable[[jt.Array, jt.Array], jt.Array]:
  match encoding_type:
    case 'identity':

      def encode_goal(latent: jax.Array, rng: chex.PRNGKey) -> jax.Array:
        return latent

    case 'orthogonal':
      # jax.nn.initializers.orthogonal creates an orthonormal matrix Q
      # Multiplying Q @ latent can produce values in [-1,1] if latent is in
      # [0,1]^d, since it's a true rotation/reflection.
      Q = jax.nn.initializers.orthogonal(1.0)(rng, (num_modules, num_modules))

      def encode_goal(latent: jax.Array, rng: chex.PRNGKey) -> jax.Array:
        return jnp.dot(Q, latent)

    case 'tokens':

      def encode_goal(latent: jax.Array, rng: chex.PRNGKey) -> jax.Array:
        """
        Simplified tokenization that encodes a latent as a language-like sequence
        of integers.

        Preference weights are encoded as num_modules..(num_modules+10).
        Preference directions are 1..num_modules.
        e.g. [0.5, 0.8, 0.0] -> 9,1,12,2,4,3
        """
        preference_directions = jnp.array(range(num_modules), dtype=jnp.int32) + 1
        preference_weights = (10 * latent).astype(jnp.int32) + num_modules + 1
        return jnp.stack([preference_directions, preference_weights]).T.reshape(-1)

    case 'invertible_nn':
      # Create the invertible neural network
      forward_fn, init_fn = create_invertible_mlp(
        dim=num_modules, num_layers=3, hidden_factor=2.0, scale=0.1
      )

      params = init_fn(rng)

      def encode_goal(latent: jax.Array, rng: chex.PRNGKey) -> jax.Array:
        """Applies invertible neural network transformation."""
        return forward_fn(latent, params)

    case 'fewshot':
      if data_generator is None:
        raise ValueError(
          'Need a data_generator with sample_from_latent method for fewshot encoding'
        )
      num_fewshot = 50  # NOTE: Hardcoded fewshot samples

      def encode_goal(latent: jax.Array, rng: chex.PRNGKey) -> jax.Array:
        """
        Encode a latent vector using freshly generated few-shot examples.
        """
        inputs, targets = data_generator.sample_from_latent(rng, latent, num_fewshot)
        return jnp.concatenate([inputs.reshape(-1), targets.reshape(-1)])

    case 'interval_shuffle':
      num_intervals = 10

      indices = jnp.arange(num_intervals)
      interval_map = jax.random.permutation(rng, indices)

      def encode_goal(latent: jax.Array, rng: chex.PRNGKey) -> jax.Array:
        """
        Maps each value in latent to a new interval based on the permutation,
        preserving the fractional part within each interval.
        Values exactly equal to 1.0 are not reshuffled.
        """
        interval_size = 1.0 / num_intervals
        interval_indices = (latent / interval_size).astype(jnp.int32)
        interval_indices = jnp.minimum(interval_indices, num_intervals - 1)
        fractional_parts = latent - interval_indices * interval_size
        new_interval_indices = interval_map[interval_indices]
        new_values = new_interval_indices * interval_size + fractional_parts
        new_values = jnp.where(latent == 1.0, 1.0, new_values)

        return new_values

    case _:
      raise ValueError(f'Unknown latent_encoding={encoding_type}')

  return encode_goal


def make_latents(num_modules: int, num_hot: int) -> jax.Array:
  """Generate all possible combinations of latent vectors.

  Args:
      num_modules: Number of total modules
      num_hot: Maximum number of active modules allowed

  Returns:
      jax.Array: Array containing all possible combinations
  """
  combinations_all = []
  for h in range(1, num_hot + 1):
    perms = itertools.combinations(range(num_modules), h)
    module_indices = np.array(list(perms)).reshape(-1, h)
    combinations_all_k_hot = k_hot(module_indices, num_modules)
    combinations_all.append(combinations_all_k_hot)

  return jnp.concatenate(combinations_all)


def split_module_combinations(
  combinations_all: jax.Array,
  task_support: str,
  num_modules: int,
  num_hot: int,
  frac_ood: float,
  rng: chex.PRNGKey,
) -> tuple[jax.Array, jax.Array]:
  """Split module combinations into in-distribution and out-of-distribution sets.

  Args:
      combinations_all: All possible module combinations
      task_support: Type of task support ('random', 'connected', etc.)
      num_modules: Number of total modules
      num_hot: Maximum number of active modules
      frac_ood: Fraction of combinations to be out-of-distribution
      rng: Random number generator key

  Returns:
      tuple[jax.Array, jax.Array]: In-distribution and out-of-distribution combinations
  """
  skip_connectivity_check = [
    'disconnected',
    'pseudo_random_disconnected',
    'non_compositional',
  ]
  skip_frac_check = [
    'disconnected',
    'non_compositional',
    'random_constant',
    'random_linear',
    'equal_linear',
    'random_quadratic',
    'random_quadratic_modules',
    'random_quadratic_modules_hot',
    'random_linear_pure',
    'random_linear_modules',
    'random_linear_modules_hot',
  ]

  # Extract task split based on task_support
  match task_support:
    case 'connected' | 'disconnected':
      assert num_hot == 2
      assert num_modules > 4 and num_modules % 2 == 0

      # Generate one-hot combinations
      combinations_connected = [k_hot(np.arange(num_modules)[:, None])]

      # Generate two-hot adjacent combinations (01 12 23 34 etc)
      combinations_connected.append(
        k_hot(
          np.stack(
            (
              np.arange(num_modules),
              (np.arange(num_modules) + 1) % num_modules,
            )
          ).T
        )
      )

      # Generate two-hot skip-one combinations (02 13 24 35 etc)
      combinations_connected.append(
        k_hot(
          np.stack(
            (
              np.arange(num_modules),
              (np.arange(num_modules) + 2) % num_modules,
            )
          ).T
        )
      )

      combinations_connected = np.concatenate(combinations_connected)

      @partial(np.vectorize, signature='(n),(m,n)->()')
      def elem_in_array(elem: jax.Array, array: jax.Array) -> np.bool:
        return np.any(np.all(elem == array, axis=1))

      mask_connected = elem_in_array(combinations_all, combinations_connected)

      # Define disconnected combinations mask
      mask_1_hot = jnp.sum(combinations_all, axis=-1) == 1
      mask_2_hot = jnp.sum(combinations_all, axis=-1) == 2
      mask_modules_1 = jnp.all(combinations_all[:, : num_modules // 2] == 0, axis=1)
      mask_modules_2 = jnp.all(combinations_all[:, num_modules // 2 :] == 0, axis=1)

      mask_disconnected = (
        (mask_1_hot & mask_modules_1)
        | (mask_1_hot & mask_modules_2)
        | (mask_2_hot & mask_modules_1)
        | (mask_2_hot & mask_modules_2)
      )

      if task_support == 'connected':
        mask_in_dist = mask_connected
      else:  # disconnected
        mask_in_dist = mask_disconnected

      mask_out_dist = ~(mask_connected | mask_disconnected)

      combinations_in_dist = jnp.array(combinations_all[mask_in_dist])
      combinations_out_dist = jnp.array(combinations_all[mask_out_dist])

    case 'pseudo_random_disconnected':
      assert num_modules % 2 == 0

      def calculate_combinations(n, k_hot):
        return sum(math.comb(n, k) for k in range(1, k_hot + 1))

      # Find best split size
      total_combinations = calculate_combinations(num_modules, num_hot)
      split_options = []

      for split_size in range(1, (num_modules // 2) + 1):
        combinations_first = calculate_combinations(split_size, num_hot)
        combinations_second = calculate_combinations(num_modules - split_size, num_hot)
        in_dist = combinations_first + combinations_second
        fraction = (total_combinations - in_dist) / total_combinations
        split_options.append((split_size, fraction))

      best_split = None
      best_fraction = 0

      for split_size, fraction in split_options:
        if fraction <= frac_ood and fraction > best_fraction:
          best_split = split_size
          best_fraction = fraction

      if best_split is None:
        best_split = 1
        combinations_first = calculate_combinations(best_split, num_hot)
        combinations_second = calculate_combinations(num_modules - best_split, num_hot)
        in_dist = combinations_first + combinations_second
        best_fraction = (total_combinations - in_dist) / total_combinations
        print(f'No valid split found, defaulting to 1,{num_modules - 1} split')
        print(f'Initial split sizes: {best_split} and {num_modules - best_split}')
        print(f'Initial fraction: {best_fraction:.3f} (target: {frac_ood:.3f})')

      def check_alignment(combination):
        first_part = combination[:best_split]
        second_part = combination[best_split:]
        all_in_first = jnp.all(second_part == 0)
        all_in_second = jnp.all(first_part == 0)
        return all_in_first | all_in_second

      valid_mask = jnp.array([check_alignment(c) for c in combinations_all])
      valid_combinations = combinations_all[valid_mask]
      invalid_combinations = combinations_all[~valid_mask]

      current_ood_fraction = len(invalid_combinations) / total_combinations

      if current_ood_fraction < frac_ood:
        additional_needed = int((frac_ood - current_ood_fraction) * total_combinations)

        if additional_needed >= len(valid_combinations):
          error_message = (
            f'Cannot raise OOD fraction to {frac_ood:.3f} without emptying in-dist!'
          )
          raise ValueError(error_message)

        rng_split, rng = jax.random.split(rng)
        perm = jax.random.permutation(rng_split, valid_combinations)

        combinations_in_dist = perm[additional_needed:]
        combinations_out_dist = jnp.concatenate(
          [invalid_combinations, perm[:additional_needed]]
        )
      else:
        combinations_in_dist = valid_combinations
        combinations_out_dist = invalid_combinations

      final_ood_fraction = len(combinations_out_dist) / total_combinations
      print(f'Final OOD fraction: {final_ood_fraction:.3f}')
      print(f'In-distribution combinations: {len(combinations_in_dist)}')
      print(f'Out-of-distribution combinations: {len(combinations_out_dist)}')

    case 'non_compositional':
      mask_last_module = combinations_all[:, -1] == 1
      combinations_in_dist = jnp.array(combinations_all[~mask_last_module])
      combinations_out_dist = jnp.array(combinations_all[mask_last_module])

    case 'random' | 'equal':
      num_ood = int(len(combinations_all) * frac_ood)
      if task_support == 'equal':
        combinations_in_dist, combinations_out_dist = (
          find_balanced_connected_compositional(
            combinations_all, len(combinations_all) - num_ood, rng
          )
        )

      else:
        num_attempts = 1000

        for attempt in range(num_attempts):
          logging.info(
            f'Attempt {attempt} at connected and compositional task_support.'
          )
          r, rng = jax.random.split(rng)
          combinations_permuted = jax.random.permutation(r, combinations_all)
          combinations_in_dist = jnp.array(combinations_permuted[:-num_ood])
          combinations_out_dist = jnp.array(combinations_permuted[-num_ood:])

          if check_compositional(combinations_in_dist) and check_connected(
            combinations_in_dist
          ):
            break
        else:
          error_message = (
            'Could not find a random split with connected and compositional support'
          )
          raise ValueError(error_message)

    case (
      'random_constant'
      | 'random_linear'
      | 'equal_linear'
      | 'random_quadratic'
      | 'random_quadratic_modules'
      | 'random_quadratic_modules_hot'
      | 'random_linear_pure'
      | 'random_linear_modules'
      | 'random_linear_modules_hot'
    ):

      def calc_total_combinations(n_pref, k_hot):
        return sum(math.comb(n_pref, k) for k in range(1, k_hot + 1))

      REF_MODULES = 16
      REFERENCE_FRAC_IN_DIST = 0.75

      total_ref_combinations = calc_total_combinations(REF_MODULES, num_hot)
      base_reference_in_dist = int(total_ref_combinations * REFERENCE_FRAC_IN_DIST)

      if task_support == 'random_linear':
        reference_in_dist = int(base_reference_in_dist * (num_modules / REF_MODULES))
      elif task_support == 'random_linear_pure':
        reference_in_dist = num_modules
      elif task_support == 'random_linear_modules':
        reference_in_dist = num_modules * 8
      elif task_support == 'random_linear_modules_hot':
        reference_in_dist = num_modules * num_hot * 8
      elif task_support == 'random_quadratic':
        reference_in_dist = int(
          base_reference_in_dist * (num_modules / REF_MODULES) ** 2
        )
      elif task_support == 'random_quadratic_modules':
        reference_in_dist = num_modules * num_modules
      elif task_support == 'random_quadratic_modules_hot':
        reference_in_dist = num_modules * num_modules * num_hot
      else:
        reference_in_dist = base_reference_in_dist

      total_combinations = len(combinations_all)

      print('\nReference case (8 preferences):')
      print(f'- Base in-distribution size: {base_reference_in_dist}')
      if task_support == 'random_linear':
        print(f'- Scaling factor: {num_modules / REF_MODULES:.2f}x')
      print(f'- Target in-distribution size: {reference_in_dist}')
      print(f'\nActual case ({num_modules} preferences):')
      print(f'- Total combinations: {total_combinations}')

      if total_combinations < reference_in_dist:
        error_message = (
          f'Not enough combinations ({total_combinations}) to maintain '
          f'reference in-distribution size ({reference_in_dist})'
        )
        raise ValueError(error_message)

      success = False
      max_attempts = 1000
      attempt = 0

      while not success and attempt < max_attempts:
        attempt += 1
        rng_split, rng = jax.random.split(rng)
        perm = jax.random.permutation(rng_split, combinations_all)

        candidate_in_dist = perm[:reference_in_dist]
        candidate_out_dist = perm[reference_in_dist:]

        if check_compositional(candidate_in_dist) and check_connected(
          candidate_in_dist
        ):
          combinations_in_dist = candidate_in_dist
          combinations_out_dist = candidate_out_dist
          success = True
          print(f'- In-distribution combinations: {len(combinations_in_dist)}')
          print(f'- Out-of-distribution combinations: {len(combinations_out_dist)}')
          print(
            f'- Actual OOD fraction: {len(combinations_out_dist) / total_combinations:.3f}'
          )
          break

        if not success:
          logging.info('repeat: %d', attempt)

      if not success:
        error_message = f'Could not find a random {task_support} split with connected and compositional support'
        raise ValueError(error_message)

    case 'popular_modules':
      total_combinations = len(combinations_all)
      target_in_dist = int(total_combinations * (1 - frac_ood))
      print(
        f'Target in-distribution combinations: {target_in_dist} out of {total_combinations}'
      )

      best_n = 0
      best_in_dist = 0

      for n_special in range(num_modules + 1):
        if n_special == 0:
          in_dist_count = 0
        else:
          mask_in_dist = np.array(
            [any(c[i] > 0 for i in range(n_special)) for c in combinations_all]
          )
          in_dist_count = np.sum(mask_in_dist)

        print(f'n={n_special}: in_dist={in_dist_count}, target={target_in_dist}')

        # Take the largest n where in_dist doesn't exceed target
        if in_dist_count > best_in_dist:
          if in_dist_count <= target_in_dist:
            best_in_dist = in_dist_count
            best_n = n_special
          else:
            break

      n_special = best_n
      print(
        f'Selected n_special={n_special} with {best_in_dist} in-distribution combinations (target: {target_in_dist})'
      )

      # Get combinations with at least one special module (in-distribution)
      mask_in_dist = np.array(
        [any(c[i] > 0 for i in range(n_special)) for c in combinations_all]
      )
      combinations_in_dist = combinations_all[mask_in_dist]
      cur_in = len(combinations_in_dist)

      # Check if we need to add more combinations to get closer to the target OOD fraction
      additional_needed = max(0, target_in_dist - cur_in)

      if additional_needed > 0 and n_special < num_modules:
        print(f'Current in-dist: {cur_in}, Target in-dist: {target_in_dist}')
        print(
          f'Need to add {additional_needed} more combinations to approach target OOD fraction'
        )

        # Next module becomes our pseudo-special one
        next_module = n_special

        # Find all combinations with the pseudo-special module that aren't already included
        pseudo_special_candidates = []
        for i, comb in enumerate(combinations_all):
          if not mask_in_dist[i] and comb[next_module] > 0:
            pseudo_special_candidates.append(i)

        # Try multiple attempts with different random orderings
        max_attempts = 1000
        attempt = 0
        success = False

        while not success and attempt < max_attempts:
          attempt += 1

          # Randomize the order of pseudo-special combinations
          rng_split, rng = jax.random.split(rng)
          perm_candidates = jax.random.permutation(
            rng_split, np.array(pseudo_special_candidates)
          )

          current_mask = mask_in_dist.copy()

          # Add up to additional_needed combinations
          to_add = min(additional_needed, len(perm_candidates))
          for i in range(to_add):
            current_mask[perm_candidates[i]] = True

          # Check if the resulting set is compositional and connected
          test_in_dist = combinations_all[current_mask]
          is_compositional = check_compositional(test_in_dist)
          is_connected = check_connected(test_in_dist)

          if is_compositional and is_connected:
            mask_in_dist = current_mask
            combinations_in_dist = test_in_dist
            success = True
            print(
              f'Successfully added {to_add} combinations with the pseudo-special module {next_module}'
            )
            break

          print(
            f'Attempt {attempt}: Adding {to_add} combinations failed, compositional={is_compositional}, connected={is_connected}'
          )

        if not success:
          print(
            'Could not add combinations with pseudo-special module while maintaining compositionality and connectivity'
          )

      combinations_out_dist = combinations_all[~mask_in_dist]

      actual_ood_frac = len(combinations_out_dist) / total_combinations

      print(
        f'Module concentration: {n_special} modules, OOD fraction: {actual_ood_frac:.4f} (target: {frac_ood:.4f})'
      )
      print(f'- In-distribution combinations: {len(combinations_in_dist)}')
      print(f'- Out-of-distribution combinations: {len(combinations_out_dist)}')
      print(f'- Is compositional: {check_compositional(combinations_in_dist)}')
      print(f'- Is connected: {check_connected(combinations_in_dist)}')

    case 'minimal_first':  # no mass on the last ones
      # Find optimal number of modules to fully connect
      total_combinations = len(combinations_all)

      def calculate_full_combinations(n, k_hot):
        """Calculate all possible combinations up to k_hot for n modules"""
        return sum(math.comb(n, k) for k in range(1, k_hot + 1))

      target_in_dist = int(total_combinations * (1 - frac_ood))
      print(
        f'Target in-distribution combinations: {target_in_dist} out of {total_combinations}'
      )

      # Find the largest n where core combos + minimal connections is <= target_in_dist
      selected_n = 0
      for n in range(num_modules - 1, 0, -1):
        core_combinations = calculate_full_combinations(n, num_hot)
        minimal_connections = num_modules - n

        total_in_dist = core_combinations + minimal_connections
        print(
          f'n={n}: core={core_combinations}, minimal={minimal_connections}, total={total_in_dist}, target={target_in_dist}'
        )

        if total_in_dist <= target_in_dist:
          selected_n = n
          break

      core_modules = list(range(selected_n))

      # The modules that will have minimal connections
      minimal_modules = list(range(selected_n, num_modules))

      print(
        f'Using {selected_n} fully connected core modules and {num_modules - selected_n} minimally connected modules'
      )

      # Initialize our in-distribution mask
      mask_in_dist = np.zeros(len(combinations_all), dtype=bool)

      # Include combinations that only involve core modules
      for i, comb in enumerate(combinations_all):
        if all(comb[j] == 0 for j in minimal_modules):
          mask_in_dist[i] = True

      # Connect all minimal modules to the first core module
      first_core = core_modules[0]

      for minimal_module in minimal_modules:
        connected = False
        for i, comb in enumerate(combinations_all):
          if not mask_in_dist[i] and comb[minimal_module] > 0 and comb[first_core] > 0:
            if all(comb[j] == 0 for j in minimal_modules if j != minimal_module):
              mask_in_dist[i] = True
              connected = True
              break

        if not connected:
          print(
            f'Warning: Could not find a simple connection for minimal module {minimal_module}'
          )
          # Try to find any connection to any core module
          for i, comb in enumerate(combinations_all):
            if (
              not mask_in_dist[i]
              and comb[minimal_module] > 0
              and any(comb[j] > 0 for j in core_modules)
            ):
              if all(comb[j] == 0 for j in minimal_modules if j != minimal_module):
                mask_in_dist[i] = True
                connected = True
                break

      current_in_dist = np.sum(mask_in_dist)

      connections_to_add = max(0, target_in_dist - current_in_dist)

      if connections_to_add > 0:
        print(
          f'Adding up to {connections_to_add} more connections for core+1 to approach target OOD fraction'
        )

        first_minimal = minimal_modules[0] if minimal_modules else None

        if first_minimal is not None:
          # Find combinations that include first_minimal and any core modules,
          # but no other minimal modules
          core_plus_one_candidates = []
          for i, comb in enumerate(combinations_all):
            if not mask_in_dist[i] and comb[first_minimal] > 0:
              has_core = any(comb[j] > 0 for j in core_modules)
              has_other_minimal = any(
                comb[j] > 0 for j in minimal_modules if j != first_minimal
              )

              if has_core and not has_other_minimal:
                core_plus_one_candidates.append(i)

          to_add = min(connections_to_add, len(core_plus_one_candidates))
          print(f'Found {len(core_plus_one_candidates)} candidates, adding {to_add}')

          for i in range(to_add):
            mask_in_dist[core_plus_one_candidates[i]] = True

      combinations_in_dist = combinations_all[mask_in_dist]
      combinations_out_dist = combinations_all[~mask_in_dist]

      actual_ood_frac = len(combinations_out_dist) / total_combinations

      print(
        f'Minimal first: {selected_n} core modules, OOD fraction: {actual_ood_frac:.4f} (target: {frac_ood:.4f})'
      )
      print(f'- In-distribution combinations: {len(combinations_in_dist)}')
      print(f'- Out-of-distribution combinations: {len(combinations_out_dist)}')

      # By construction, our solution is guaranteed to be compositional and connected
      print('- Is compositional: True (by construction)')
      print('- Is connected: True (by construction)')

    case 'unpopular_modules':
      # Find optimal number of modules to fully connect
      total_combinations = len(combinations_all)

      def calculate_full_combinations(n, k_hot):
        """Calculate all possible combinations up to k_hot for n modules"""
        return sum(math.comb(n, k) for k in range(1, k_hot + 1))

      target_in_dist = int(total_combinations * (1 - frac_ood))
      print(
        f'Target in-distribution combinations: {target_in_dist} out of {total_combinations}'
      )

      # Find the largest n where core combos + minimal connections is <= target_in_dist
      selected_n = 0
      for n in range(num_modules - 1, 1, -1):
        core_combinations = calculate_full_combinations(n, num_hot)
        minimal_connections = num_modules - n

        total_in_dist = core_combinations + minimal_connections
        print(
          f'n={n}: core={core_combinations}, minimal={minimal_connections}, total={total_in_dist}, target={target_in_dist}'
        )

        if total_in_dist <= target_in_dist:
          selected_n = n
          break

      if selected_n == 0:
        selected_n = 2
        print(f'All options exceed target, using minimum n={selected_n}')

      core_modules = list(range(selected_n))

      minimal_modules = list(range(selected_n, num_modules))

      print(
        f'Using {selected_n} fully connected core modules and {num_modules - selected_n} minimally connected modules'
      )

      mask_in_dist = np.zeros(len(combinations_all), dtype=bool)

      # Include combinations that only involve core modules
      for i, comb in enumerate(combinations_all):
        if all(comb[j] == 0 for j in minimal_modules):
          mask_in_dist[i] = True

      # Connect each minimal module to any of the core modules
      for minimal_module in minimal_modules:
        candidates = []
        for i, comb in enumerate(combinations_all):
          if not mask_in_dist[i] and comb[minimal_module] > 0:
            if any(comb[j] > 0 for j in core_modules):
              if all(comb[j] == 0 for j in minimal_modules if j != minimal_module):
                candidates.append(i)

        if candidates:
          selected_idx = candidates[0]
          mask_in_dist[selected_idx] = True

          active_core = [
            j for j in core_modules if combinations_all[selected_idx][j] > 0
          ]
          print(
            f'Connected minimal module {minimal_module} to core modules: {active_core}'
          )
        else:
          print(
            f'Warning: Could not find a connection for minimal module {minimal_module}'
          )

      current_in_dist = np.sum(mask_in_dist)

      # Check if we need to add more connections to approach target OOD fraction
      connections_to_add = max(0, target_in_dist - current_in_dist)

      if connections_to_add > 0:
        print(
          f'Adding up to {connections_to_add} more connections for core+1 to approach target OOD fraction'
        )

        first_minimal = minimal_modules[0] if minimal_modules else None

        if first_minimal is not None:
          core_plus_one_candidates = []
          for i, comb in enumerate(combinations_all):
            if not mask_in_dist[i] and comb[first_minimal] > 0:
              has_core = any(comb[j] > 0 for j in core_modules)
              has_other_minimal = any(
                comb[j] > 0 for j in minimal_modules if j != first_minimal
              )

              if has_core and not has_other_minimal:
                core_plus_one_candidates.append(i)

          to_add = min(connections_to_add, len(core_plus_one_candidates))
          print(f'Found {len(core_plus_one_candidates)} candidates, adding {to_add}')

          for i in range(to_add):
            mask_in_dist[core_plus_one_candidates[i]] = True

      combinations_in_dist = combinations_all[mask_in_dist]
      combinations_out_dist = combinations_all[~mask_in_dist]

      actual_ood_frac = len(combinations_out_dist) / total_combinations

      print(
        f'Minimal any core: {selected_n} core modules, OOD fraction: {actual_ood_frac:.4f} (target: {frac_ood:.4f})'
      )
      print(f'- In-distribution combinations: {len(combinations_in_dist)}')
      print(f'- Out-of-distribution combinations: {len(combinations_out_dist)}')

      is_compositional = check_compositional(combinations_in_dist)
      is_connected = check_connected(combinations_in_dist)
      print(f'- Is compositional: {is_compositional}')
      print(f'- Is connected: {is_connected}')

  # Check that in-distribution and out-distribution sets have no overlap
  in_dist_np = np.array(combinations_in_dist)
  out_dist_np = np.array(combinations_out_dist)
  all_combinations_np = np.array(combinations_all)

  in_dist_np = np.round(in_dist_np).astype(np.int32)
  out_dist_np = np.round(out_dist_np).astype(np.int32)
  all_combinations_np = np.round(all_combinations_np).astype(np.int32)

  in_dist_set = set(tuple(row) for row in in_dist_np)
  out_dist_set = set(tuple(row) for row in out_dist_np)

  overlap = in_dist_set.intersection(out_dist_set)
  if overlap:
    error_message = f'Found {len(overlap)} overlapping combinations between in-distribution and out-distribution sets'
    raise ValueError(error_message)

  # Check that all combinations in both sets appear exactly once
  all_combinations_set = in_dist_set.union(out_dist_set)
  all_combinations_from_original = set(tuple(row) for row in all_combinations_np)

  if len(all_combinations_set) != len(combinations_all):
    error_message = f'Some combinations are missing or duplicated. Expected {len(combinations_all)}, got {len(all_combinations_set)}'
    raise ValueError(error_message)

  if all_combinations_set != all_combinations_from_original:
    error_message = 'The union of in-distribution and out-distribution sets does not match the original combinations'
    raise ValueError(error_message)

  # Check that the OOD fraction is within tolerance (±1 combination)
  total_combinations = len(combinations_all)
  actual_ood = len(combinations_out_dist)
  expected_ood = int(total_combinations * frac_ood)
  if task_support not in skip_frac_check:
    if abs(actual_ood - expected_ood) > 1:
      error_message = f'OOD fraction is off by more than 1 combination. Expected {expected_ood}, got {actual_ood}'
      raise ValueError(error_message)

  # Only check connectedness and compositionality for specific task support types
  if task_support not in skip_connectivity_check:
    if not check_connected(combinations_in_dist):
      error_message = f'In-distribution set for {task_support} is not connected'
      raise ValueError(error_message)

    if not check_compositional(combinations_in_dist):
      error_message = f'In-distribution set for {task_support} is not compositional'
      raise ValueError(error_message)

  return combinations_in_dist, combinations_out_dist


def find_balanced_connected_compositional(
  all_tasks: jax.Array,
  desired_num_tasks: int,
  rng: jax.random.PRNGKey,
  max_attempts: int = 1000,
) -> tuple[jax.Array, jax.Array]:
  """
  1) Randomly shuffle ALL tasks.
  2) Select the first `desired_num_tasks` as a candidate set.
  3) Run local search to balance usage counts by:
     - Computing ideal mean usage
     - Systematically considering all possible swaps
     - Optimizing distance to mean vector
     - Requiring max_usage - min_usage <= 1
  4) Check compositionality & connectedness. If it fails, repeat up to `max_attempts`.

  Args:
      all_tasks: jax.Array (N, num_modules). All possible tasks.
      desired_num_tasks: int. The exact number of tasks you want to pick.
      rng: jax.random.PRNGKey, for randomness.
      max_attempts: number of times to repeat if checks fail.

  Returns:
      tuple[jax.Array, jax.Array]: (in_distribution_tasks, out_of_distribution_tasks)

  Raises:
      ValueError if no suitable subset is found after `max_attempts`.
  """
  N = all_tasks.shape[0]
  num_modules = all_tasks.shape[1]

  if desired_num_tasks > N:
    raise ValueError(f'Requested {desired_num_tasks} tasks, but only have {N} available.')

  def imbalance(usage_counts):
    """Calculate imbalance as difference between max and min usage."""
    return usage_counts.max() - usage_counts.min()

  def distance_to_mean(usage_counts, target_mean):
    """Calculate L2 distance to ideal balanced usage."""
    return np.sum((usage_counts - target_mean) ** 2)

  for attempt in range(max_attempts):
    rng, rng_perm = jax.random.split(rng)
    perm_indices = jax.random.permutation(rng_perm, N)
    tasks_shuffled = np.array(all_tasks[perm_indices])

    candidate_np = tasks_shuffled[:desired_num_tasks].copy()
    outside_np = tasks_shuffled[desired_num_tasks:].copy()

    usage_count = candidate_np.sum(axis=0)
    current_imbalance = imbalance(usage_count)

    # Local search with systematic swaps
    improved = True
    last_in_idx = 0
    last_out_idx = 0

    while improved and current_imbalance > 1:
      improved = False

      total_activations = np.sum(usage_count)
      target_mean = total_activations / num_modules
      current_distance = distance_to_mean(usage_count, target_mean)

      for idx_in in range(last_in_idx, desired_num_tasks):
        task_in = candidate_np[idx_in]

        start_out_idx = last_out_idx if idx_in == last_in_idx else 0

        for idx_out in range(start_out_idx, len(outside_np)):
          task_out = outside_np[idx_out]

          new_usage = usage_count - task_in + task_out
          new_imbalance = imbalance(new_usage)
          new_distance = distance_to_mean(new_usage, target_mean)

          if new_distance < current_distance and new_imbalance <= current_imbalance:
            temp = candidate_np[idx_in].copy()
            candidate_np[idx_in] = outside_np[idx_out].copy()
            outside_np[idx_out] = temp

            usage_count = new_usage
            current_imbalance = new_imbalance
            current_distance = new_distance
            improved = True
            last_in_idx = idx_in
            last_out_idx = idx_out + 1
            break

        if improved:
          break

      if not improved:
        last_in_idx = 0
        last_out_idx = 0

      if current_imbalance <= 1:
        final_in_dist = jnp.array(candidate_np)
        final_out_dist = jnp.array(outside_np)
        if check_compositional(final_in_dist) and check_connected(final_in_dist):
          print(f'Found balanced subset in attempt #{attempt + 1}')
          print(f'Final imbalance: {current_imbalance}')
          print(f'Distance from mean: {current_distance:.2f}')
          print(f'Module usages: {usage_count}')
          return final_in_dist, final_out_dist
        print(
          f'Attempt {attempt + 1} balanced but failed compositionality/connectivity check.'
        )

    if current_imbalance > 1:
      print(f'Attempt {attempt + 1} failed to achieve balance.')

  raise ValueError(
    f'Could not find a balanced, connected, and compositional subset '
    f'after {max_attempts} attempts.'
  )
