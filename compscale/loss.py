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
import optax
from flax import struct

from compscale.data.base import Dataset
from compscale.experiment import ExperimentLoss


@struct.dataclass
class MeanSquaredError(ExperimentLoss):
  def __call__(
    self, params: dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool
  ) -> tuple[jax.Array, dict]:
    logits = self.apply_fn(params, batch, rngs={'dropout': rng, 'target': rng})

    loss = optax.squared_error(predictions=logits, targets=batch.y)
    loss_aggr = jnp.mean(loss)
    r2 = 1 - (loss / batch.info['base_mse'])

    return loss_aggr, dict(r2=jnp.mean(r2), loss=loss_aggr)


@struct.dataclass
class CrossEntropyMaskedLoss(ExperimentLoss):
  def __call__(
    self, params: dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool
  ) -> tuple[jax.Array, dict]:
    logits = self.apply_fn(params, batch, rngs={'dropout': rng, 'target': rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.y)
    loss = jnp.sum(batch.mask * loss) / jnp.sum(batch.mask)

    correct = jnp.argmax(logits, axis=-1) == batch.y
    acc = jnp.sum(batch.mask * correct) / jnp.sum(batch.mask)
    correct_all = jnp.sum(batch.mask * correct, axis=1) == jnp.sum(batch.mask, axis=1)
    acc_all = jnp.mean(correct_all)

    return loss, dict(acc=acc, acc_all=acc_all)
