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

import abc
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

EnvironmentState = Any


@struct.dataclass
class EnvironmentInteraction:
  done: jax.Array
  observation: jax.Array
  reward: jax.Array
  timestep: int
  info: dict = struct.field(pytree_node=False, default_factory=dict)


class Environment(abc.ABC):
  @property
  @abc.abstractmethod
  def num_actions(self) -> int:
    """Number of possible actions."""

  @property
  @abc.abstractmethod
  def observation_shape(self) -> tuple[int, ...]:
    """The shape of the observation."""

  @abc.abstractmethod
  def observe(self, env_state: EnvironmentState) -> jax.Array:
    """Returns the observation from the environment state."""

  @abc.abstractmethod
  def reset(
    self, rng: chex.PRNGKey, goal: jax.Array | None = None
  ) -> tuple[EnvironmentState, EnvironmentInteraction]:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def reset_goal(self, rng: chex.PRNGKey, mode: str) -> jax.Array:
    """Resets the environment goal."""

  def step(
    self, rng: chex.PRNGKey, env_state: EnvironmentState, action: jax.Array
  ) -> tuple[EnvironmentState, EnvironmentInteraction]:
    """Run one timestep of the environment's dynamics.

    Returns the Transition and the Environment state.
    """

    # return self._step(rng, env_state, action)
    def empty_step(
      rng: chex.Array, state: EnvironmentState, action: jax.Array
    ) -> tuple[EnvironmentState, EnvironmentInteraction]:
      """
      Only update time and give no reward.
      """
      new_timestep = state.timestep + 1
      new_state = state.replace(timestep=new_timestep)
      new_emission = EnvironmentInteraction(
        observation=self.observe(state),
        reward=jnp.array(0.0),
        done=state.done,
        timestep=new_timestep,
      )
      return new_state, new_emission

    # Only run env step if not already done
    return jax.lax.cond(
      env_state.done,
      empty_step,
      self._step,
      rng,
      env_state,
      action,
    )

  @abc.abstractmethod
  def _step(
    self, rng: chex.PRNGKey, env_state: EnvironmentState, action: jax.Array
  ) -> tuple[EnvironmentState, EnvironmentInteraction]:
    """Run one timestep of the environment's dynamics.
    Returns the Transition and the Environment state.
    """
