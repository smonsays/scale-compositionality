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

import itertools
from functools import partial

import chex
import jax
import jax.numpy as jnp
import sentencepiece

import compscale.data.utils as data_utils
from compscale.data import base
from compscale.data.envs import grid
from compscale.data.envs import preference


def goal_to_sentence(goal: grid.CompositionalGridGoal) -> str:
  direction_str = {
    0: 'top-left quadrant',
    1: 'top-right quadrant',
    2: 'bottom-left quadrant',
    3: 'bottom-right quadrant',
  }[goal.direction]

  interaction_str = {
    0: 'sell',
    1: 'throw',
    2: 'grab',
    3: 'clean',
    4: 'examine',
  }[goal.interaction]

  maze_str = {
    0: 'maze 1',
    1: 'maze 2',
    2: 'maze 3',
    3: 'maze 4',
    4: 'maze 5',
  }[goal.maze]

  object_str = {
    0: 'chair',
    1: 'phone',
    2: 'shoes',
    3: 'pen',
    4: 'clock',
  }[goal.object]

  return (
    f'In the {direction_str}, find and {interaction_str} the {object_str} in {maze_str}.'
  )


def create_compgrid_dataloader(
  batch_size: int,
  num_train: int,
  num_test: int,
  num_ood: int,
  *,
  tokenizer_path: str,
  grid_size: int,
  num_interactions: int,
  num_mazes: int,
  num_objects: int,
  num_distractors: int,
  frac_ood: float,
  seed: int,
) -> tuple[base.Dataloader, dict[str, base.Dataloader]]:
  # Load env and tokenizer
  tokenizer = sentencepiece.SentencePieceProcessor()
  tokenizer.Load(tokenizer_path)
  env = grid.CompositionalGrid(
    grid_size=grid_size,
    num_interactions=num_interactions,
    num_mazes=num_mazes,
    num_objects=num_objects,
    num_distractors=num_distractors,
    frac_ood=frac_ood,
    task_support='random',
    seed=seed,
  )

  # Pregenerate all goal tokenizations
  all_goals = itertools.product(
    range(4),  # directions
    range(env.num_interactions),
    range(env.num_mazes),
    range(env.num_objects),
  )
  # NOTE: 17 comes from tokenization of sentence structure defined in goal_to_sentence()
  goal_seq_len = 17
  goal_to_tokens = jnp.zeros(
    (4, num_interactions, num_mazes, num_objects, goal_seq_len), dtype=jnp.int32
  )
  for goal_combination in all_goals:
    goal = grid.CompositionalGridGoal(*goal_combination)
    goal_text = goal_to_sentence(goal)
    goal_tokenized = jnp.array(tokenizer.EncodeAsIds(goal_text), dtype=jnp.int32)
    goal_to_tokens = goal_to_tokens.at[
      goal_combination[0],
      goal_combination[1],
      goal_combination[2],
      goal_combination[3],
    ].set(goal_tokenized)

  rng = jax.random.key(seed)
  goal_to_orthogonal_embed = jax.nn.initializers.orthogonal(1.0)(
    key=rng, shape=(4, num_interactions, num_mazes, num_objects, 256)
  )

  # Define a simple normalization function instead of using nn.RMSNorm
  def normalize(x):
    # RMS Normalization implementation
    mean_squared = jnp.mean(x**2, axis=-1, keepdims=True)
    inv_rms = jax.lax.rsqrt(mean_squared + 1e-6)
    return x * inv_rms

  @partial(jax.jit, static_argnames=['mode'])
  def sample_fn(rng: chex.PRNGKey, mode: str) -> base.Dataset:
    @jax.vmap
    def _sample_fn(rng: chex.PRNGKey) -> base.Dataset:
      rng_goal, rng_reset, rng_demo = jax.random.split(rng, 3)
      goal, info = env.reset_goal(rng_goal, mode=mode)
      env_state, _ = env.reset(rng_reset, goal=goal)
      trajectory, actions = env.demonstrate(rng_demo, env_state)

      # Get observations and normalize them
      observations = trajectory.observation.reshape(grid_size**2 - 1, -1)
      normalized_observations = normalize(observations)

      return base.Dataset(
        x=normalized_observations,  # Use normalized observations
        y=actions,
        z=goal_to_tokens[goal.direction, goal.interaction, goal.maze, goal.object],
        mask=~trajectory.done,
        info=dict(
          goal=goal_to_orthogonal_embed[
            goal.direction, goal.interaction, goal.maze, goal.object
          ]
        ),
      )

    rngs = jax.random.split(rng, batch_size)
    return _sample_fn(rngs)

  trainloader = base.SyntheticDataloader(
    num_tasks=num_train,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='train'),
    output_dim=env.num_actions,
    seed=seed,
  )

  testloader = base.SyntheticDataloader(
    num_tasks=num_test,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='test'),
    output_dim=env.num_actions,
    seed=seed,
  )

  oodloader = base.SyntheticDataloader(
    num_tasks=num_ood,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='ood'),
    output_dim=env.num_actions,
    seed=seed,
  )

  return (trainloader, dict(test=testloader, ood=oodloader))


def create_prefgrid_dataloader(
  batch_size: int,
  num_train: int,
  num_test: int,
  num_ood: int,
  *,
  latent_encoding: str,
  num_preferences: int,
  num_features: int,
  num_objects: int,
  num_hot: int,
  task_distribution: str,
  frac_ood: float,
  discount: float,
  timelimit: int,
  task_support: str,
  seed: int,
) -> tuple[base.Dataloader, dict[str, base.Dataloader]]:
  """Creates dataloaders for preference grid tasks."""

  rng = jax.random.key(seed)
  encode_goal = data_utils.make_encode_latent(rng, latent_encoding, num_preferences)

  def normalize(x):
    mean_squared = jnp.mean(x**2, axis=-1, keepdims=True)
    inv_rms = jax.lax.rsqrt(mean_squared + 1e-6)
    return x * inv_rms

  env = preference.CompositionalPreference(
    num_preferences=num_preferences,
    num_features=num_features,
    num_objects=num_objects,
    num_hot=num_hot,
    task_distribution=task_distribution,
    discount=discount,
    timelimit=timelimit,
    frac_ood=frac_ood,
    task_support=task_support,
    seed=seed,
  )

  @partial(jax.jit, static_argnames=['mode'])
  def sample_fn(rng: chex.PRNGKey, mode: str) -> base.Dataset:
    @jax.vmap
    def _sample_fn(rng: chex.PRNGKey) -> base.Dataset:
      rng_goal, rng_reset, rng_demo, rng_encode = jax.random.split(rng, 4)
      goal, info = env.reset_goal(rng_goal, mode=mode)
      env_state, _ = env.reset(rng_reset, goal=goal)
      trajectory, actions = env.demonstrate(rng_demo, env_state)

      # Get observations and normalize them
      observations = trajectory.observation.reshape(timelimit, -1)
      normalized_observations = normalize(observations)

      return base.Dataset(
        x=normalized_observations,  # Use normalized observations
        y=actions,
        z=encode_goal(goal, rng_encode),
        mask=~trajectory.done,
        info=dict(goal=goal, latents=goal),
      )

    rngs = jax.random.split(rng, batch_size)
    return _sample_fn(rngs)


  # Create main train/test/ood loaders
  trainloader = base.SyntheticDataloader(
    num_tasks=num_train,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='train'),
    output_dim=len(preference.ACTIONS),
    seed=seed,
  )

  testloader = base.SyntheticDataloader(
    num_tasks=num_test,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='test'),
    output_dim=len(preference.ACTIONS),
    seed=seed,
  )

  oodloader = base.SyntheticDataloader(
    num_tasks=num_ood,
    batch_size=batch_size,
    sample_fn=partial(sample_fn, mode='ood'),
    output_dim=len(preference.ACTIONS),
    seed=seed,
  )

  # Return combined loaders
  return (
    trainloader,
    dict(
      test=testloader,
      ood=oodloader,
    ),
  )
