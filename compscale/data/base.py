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
from typing import Callable
from typing import ClassVar
from typing import Iterator
from typing import NamedTuple

import chex
import jax


class Dataset(NamedTuple):
  x: jax.Array  # sequence of inputs
  y: jax.Array  # sequence of targets
  z: jax.Array  # goal, latent, conditioner
  mask: jax.Array
  info: dict = ClassVar[dict()]


class Dataloader(abc.ABC):
  @property
  @abc.abstractmethod
  def output_dim(self) -> int:
    pass

  @abc.abstractmethod
  def __len__(self) -> int:
    pass

  @abc.abstractmethod
  def __iter__(self) -> Iterator[Dataset]:
    pass


class DatasetGenerator(abc.ABC):
  def __init__(self, input_shape: tuple[int], output_dim: int) -> None:
    self.input_shape = input_shape
    self.output_dim = output_dim

  @abc.abstractmethod
  def sample(
    self, rng: chex.PRNGKey, num_tasks: int, num_samples: int, mode: str
  ) -> Dataset:
    """
    Generate a batch of tasks.

    Args:
        rng (jax.random.key): The random number generator to use.
        num_tasks (int): The number of tasks to generate.
        num_samples (int): The number of samples per task.
        mode (str): The mode of the generated data (e.g. 'train', 'test', 'ood').

    Returns:
        A tuple (x, y) containing the input and output data for the generated tasks.
        x has shape (num_tasks, num_samples) + input_shape.
        y has shape (num_tasks, num_samples, output_dim).
    """
    pass


class SyntheticDataloader(Dataloader):
  def __init__(
    self, num_tasks: int, batch_size: int, sample_fn: Callable, output_dim: int, seed: int
  ) -> None:
    if not num_tasks % batch_size == 0:
      raise AssertionError('Number of tasks must be divisible by the batch_size.')
    self.batch_size = batch_size
    self.sample_fn = sample_fn
    self.steps = num_tasks // batch_size
    self.fixed_rng = jax.random.key(seed)
    self._output_dim = output_dim

  @property
  def output_dim(self) -> int:
    return self._output_dim

  def __len__(self) -> int:
    return self.steps

  def __iter__(self) -> Iterator[Dataset]:
    for i in range(self.steps):
      rng = jax.random.fold_in(self.fixed_rng, i)
      yield self.sample_fn(rng)
