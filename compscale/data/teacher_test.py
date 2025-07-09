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
import pandas as pd
import plotly.express as px
from absl.testing import absltest
from absl.testing import parameterized

from compscale.data import teacher


class HyperTeacherTestCase(parameterized.TestCase):
  @parameterized.parameters(
    dict(task_distribution='discrete', seed=0),
    dict(task_distribution='khot', seed=1),
    dict(task_distribution='continuous', seed=2),
  )
  def test_regression_visual(self, task_distribution: str, seed: int):
    data_generator = teacher.HyperTeacher(
      input_dim=1,
      output_dim=1,
      hidden_dims=(32,),
      use_bias=True,
      num_modules=8,
      num_hot=6,
      frac_ood=0.25,
      scale=1.0,
      task_distribution=task_distribution,
      task_support='random',
      seed=seed,
    )

    df_list = []
    for task in range(25):
      rng = jax.random.PRNGKey(task)
      inputs, targets, latents = data_generator.sample(
        rng, num_tasks=5, num_samples=1000, mode='train'
      )
      df_list.append(
        pd.DataFrame(
          dict(
            x=inputs[0, :, 0],
            y=targets[0, :, 0],
            task=[str(latents[0])] * len(inputs[0, :, 0]),
          ),
        )
      )

    df = pd.concat(df_list).sort_values(by=['task'])
    fig = px.scatter(df, x='x', y='y', facet_col='task', facet_col_wrap=5)
    fig.show()

  @parameterized.parameters(
    dict(latent_encoding='identity', seed=0),
    dict(latent_encoding='orthogonal', seed=1),
  )
  def test_create_hyperteacher_dataset(self, latent_encoding: str, seed: int):
    input_dim = 1
    output_dim = 1
    num_modules = 8
    num_samples = 64

    with jax.disable_jit(False):
      ds_train, ds_eval = teacher.create_hyperteacher_dataloader(
        batch_size := 128,
        num_train=128000,
        num_test=128,
        num_ood=128,
        latent_encoding=latent_encoding,
        num_samples=num_samples,
        frac_ood=0.25,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=(32,),
        use_bias=True,
        num_modules=num_modules,
        num_hot=6,
        scale=1.0,
        task_distribution='discrete',
        task_support='random',
        seed=seed,
      )

      for ds in [ds_train, *ds_eval.values()]:
        batch = next(iter(ds))
        chex.assert_shape(batch.x, (batch_size, num_samples, input_dim))
        chex.assert_shape(batch.y, (batch_size, num_samples, output_dim))
        chex.assert_shape(batch.z, (batch_size, num_modules))
        chex.assert_shape(batch.info['latents'], (batch_size, num_modules))
        assert jnp.all(batch.x <= 1.0)
        assert jnp.all(batch.x >= -1.0)

    # import tensorflow_datasets as tfds
    # tfds.benchmark(ds_train)


if __name__ == '__main__':
  absltest.main()
