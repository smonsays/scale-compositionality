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

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

from compscale.data import imitation


class ImitationTestCase(chex.TestCase):
  rng = jax.random.key(0)

  # @parameterized.parameters(dict(frac_ood=0.25), dict(frac_ood=0.5))
  def test_create_compgrid_dataloader(self, frac_ood: float = 0.25) -> None:
    batch_size = 256
    grid_size = 7
    num_objects = 2

    with jax.disable_jit(False):
      train_loader, eval_loaders = imitation.create_compgrid_dataloader(
        batch_size=batch_size,
        num_train=25600,
        num_test=25600,
        num_ood=25600,
        tokenizer_path='configs/sentencepiece_cc_all.32000.100extra-sentencepiece.model',
        grid_size=grid_size,
        num_interactions=2,
        num_mazes=2,
        num_objects=num_objects,
        num_distractors=2,
        frac_ood=frac_ood,
        seed=0,
      )
      for ds in [train_loader, eval_loaders['test'], eval_loaders['ood']]:
        batch = next(iter(ds))
        chex.assert_shape(
          batch.x,
          (batch_size, grid_size**2 - 1, grid_size * grid_size * (num_objects + 2)),
        )
        chex.assert_shape(batch.y, (batch_size, grid_size**2 - 1))
        chex.assert_shape(batch.z, (batch_size, 17))
        chex.assert_type(batch.z, jnp.int32)

  @parameterized.parameters(
    dict(
      num_preferences=5,
      latent_encoding='tokens',
      task_distribution='continuous',
      task_support='random',
      latent_shape=(2 * 5,),
      latent_dtype='int32',
    )
  )
  def test_create_prefgrid_dataloader(
    self,
    num_preferences: int,
    latent_encoding: str,
    task_distribution: str,
    task_support: str,
    latent_shape: tuple[int, ...],
    latent_dtype: str,
  ) -> None:
    batch_size = 256
    num_objects = 2
    num_features = 3
    timelimit = 17

    with jax.disable_jit(False):
      train_loader, eval_loaders = imitation.create_prefgrid_dataloader(
        batch_size=batch_size,
        num_train=25600,
        num_test=25600,
        num_ood=25600,
        latent_encoding=latent_encoding,
        num_preferences=num_preferences,
        num_features=num_features,
        num_objects=num_objects,
        num_hot=2,
        task_distribution=task_distribution,
        discount=0.9,
        timelimit=timelimit,
        task_support=task_support,
        frac_ood=0.25,
        seed=0,
      )
      for ds in [train_loader, eval_loaders['test'], eval_loaders['ood']]:
        batch = next(iter(ds))
        # preference grid has a fixed shape of (7, 7)
        chex.assert_shape(batch.x, (batch_size, timelimit, 7 * 7 * (num_features + 2)))
        chex.assert_shape(batch.y, (batch_size, timelimit))
        chex.assert_shape(batch.z, (batch_size, *latent_shape))
        chex.assert_type(batch.z, jnp.dtype(latent_dtype))

    # import tensorflow_datasets as tfds
    # tfds.benchmark(train_loader)


if __name__ == '__main__':
  absltest.main()
