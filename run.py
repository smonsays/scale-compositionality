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

import dataclasses
import logging
import os
import pprint
import random
import socket
import time
import uuid

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from absl import app
from absl import flags
from flax import traverse_util as flax_traverse
from ml_collections import config_flags

from compscale import config_classes
from compscale import loss
from compscale.callbacks import ExtendedActivationProbing
from compscale.data import imitation
from compscale.data import teacher
from compscale.experiment import CallbackEvent
from compscale.experiment import Experiment
from compscale.logging import StandardLogger
from compscale.logging import WandbLogger
from compscale.model import policy as policy_lib

FLAGS = flags.FLAGS

# jax.config.update('jax_enable_x64', True)
# Import ml_collections flags, expose jax flags and add additional abseil flags
config_flags.DEFINE_config_file(
  name='config',
  default='configs/teacher.py',  # teacher, preference, goal
  help_string='Training configuration.',
)
jax.config.parse_flags_with_absl()


def setup_experiment(
  config: config_classes.compscaleConfig,
  logger_list: tuple = (),
  callbacks: tuple = (),
) -> Experiment:
  # Instantiate data
  logging.info('Loading dataset...')
  match config.dataset.env:
    case config_classes.GoalGridConfig():
      train_loader, eval_loaders = imitation.create_compgrid_dataloader(
        batch_size=config.batch_size,
        seed=config.seed,
        num_train=config.dataset.num_train,
        num_test=config.dataset.num_test,
        num_ood=config.dataset.num_ood,
        frac_ood=config.dataset.frac_ood,
        tokenizer_path=config.dataset.env.tokenizer_path,
        grid_size=config.dataset.env.grid_size,
        num_interactions=config.dataset.env.num_interactions,
        num_mazes=config.dataset.env.num_mazes,
        num_objects=config.dataset.env.num_objects,
        num_distractors=config.dataset.env.num_distractors,
      )
      loss_fn = loss.CrossEntropyMaskedLoss
    case config_classes.PreferenceGridConfig():
      if config.dataset.env.num_preferences > config.dataset.env.num_features:
        raise ValueError(
          'Number of preferences must be less or equal to number of features'
        )
      train_loader, eval_loaders = imitation.create_prefgrid_dataloader(
        batch_size=config.batch_size,
        seed=config.seed,
        num_train=config.dataset.num_train,
        num_test=config.dataset.num_test,
        num_ood=config.dataset.num_ood,
        frac_ood=config.dataset.frac_ood,
        latent_encoding=config.dataset.env.latent_encoding,
        task_support=config.dataset.task_support,
        num_hot=config.dataset.env.num_hot,
        num_preferences=config.dataset.env.num_preferences,
        num_features=config.dataset.env.num_features,
        num_objects=config.dataset.env.num_objects,
        task_distribution=config.dataset.env.task_distribution,
        discount=config.dataset.env.discount,
        timelimit=config.dataset.env.timelimit,
      )
      loss_fn = loss.CrossEntropyMaskedLoss
    case config_classes.HyperTeacherConfig():
      train_loader, eval_loaders = teacher.create_hyperteacher_dataloader(
        batch_size=config.batch_size,
        seed=config.seed,
        num_train=config.dataset.num_train,
        num_test=config.dataset.num_test,
        num_ood=config.dataset.num_ood,
        frac_ood=config.dataset.frac_ood,
        latent_encoding=config.dataset.env.latent_encoding,
        num_samples=config.dataset.env.num_samples,
        input_dim=config.dataset.env.input_dim,
        output_dim=config.dataset.env.output_dim,
        hidden_dims=config.dataset.env.hidden_dims,
        use_bias=config.dataset.env.use_bias,
        num_modules=config.dataset.env.num_modules,
        num_hot=config.dataset.env.num_hot,
        scale=config.dataset.env.scale,
        task_distribution=config.dataset.env.task_distribution,
        task_support=config.dataset.task_support,
      )
      loss_fn = loss.MeanSquaredError
    case _:
      raise ValueError(f'Unknown dataset env {config.dataset.env}')

  logging.info('...done')

  config.model = dataclasses.replace(config.model, d_out=train_loader.output_dim)

  # Instantiate model
  match config.model.policy_type:
    case 'hypernetwork':
      sequence_model = policy_lib.HypernetworkPolicy(config.model)
    case 'mlp':
      sequence_model = policy_lib.MultilayerPerceptronPolicy(config.model)
    case 'transformer':
      sequence_model = policy_lib.TransformerPolicy(config.model)
    case _:
      raise ValueError(f'Unsupported policy type: {config.model.policy_type}')

  # Instantiate optimizer
  optimizer_ops = []

  if config.optimizer.clip_by_global_norm is not None:
    optimizer_ops.append(optax.clip_by_global_norm(config.optimizer.clip_by_global_norm))

  # Instantiate learning rate scheduler
  match config.optimizer.schedule:
    case 'cosine':
      schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer.learning_rate,
        warmup_steps=config.optimizer.warmup_steps,
        decay_steps=len(train_loader) - config.optimizer.warmup_steps,
        end_value=config.optimizer.learning_rate * 0.1,
      )
    case 'constant':
      schedule = optax.constant_schedule(config.optimizer.learning_rate)
    case _:
      raise ValueError(f'Unknown learning rate schedule: {config.optimizer.schedule}')

  optimizer_ops.append(
    getattr(optax, config.optimizer.optimizer)(
      learning_rate=schedule,
      weight_decay=config.optimizer.weight_decay,
      mask=(
        lambda p: jax.tree_util.tree_map(  # mask weight decay for biases and layernorms
          lambda x: x.ndim != 1, p
        )
      )
      if config.optimizer.mask_weight_decay
      else None,
    )
  )
  optimizer = optax.chain(*optimizer_ops)

  callback_list = list(callbacks)
  if config.callbacks.mlp_probing and config.model.policy_type in ['mlp', 'transformer']:
    mlp_probing = ExtendedActivationProbing(
      log_level=config.log_level,
      onevent=CallbackEvent.STEP,
      save_to_disk=config.callbacks.save_to_disk,
      l2_reg=config.callbacks.l2_reg,
    )
    callback_list.append(mlp_probing)

  # Instantiate experiment runner
  return Experiment(
    config=config,
    model=sequence_model,
    loss=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    eval_loaders=eval_loaders,
    logger_list=logger_list,
    callbacks=tuple(
      callback_list
    ),  # Use the callback list with potential probing callback
    log_every=config.log_every,
    log_level=config.log_level,
  )


def main(argv: list[str]) -> None:
  del argv

  logging.info('Running on {}'.format(jax.default_backend()))
  config: config_classes.compscaleConfig = flags.FLAGS.config

  if config.seed is None:
    config.seed = random.randint(0, 99999)

  # Setup workdir and overwrite the workdir path in the config
  unique_id = str(uuid.uuid4())[-4:]
  hostname = socket.gethostname()
  datetime = time.strftime('%Y%m%d_%H%M%S_')
  id = datetime + hostname + '_' + unique_id + '_{}'.format(config.name)
  workdir = os.path.join(os.getcwd(), config.workdir, id)
  logging.info('Logging to {}'.format(workdir))
  config = dataclasses.replace(config, workdir=workdir)

  logger_list = []
  for logger in config.logger_types:
    match logger:
      case 'standard':
        logger_list.append(StandardLogger(workdir))
      case 'wandb':
        logger_list.append(WandbLogger(config, workdir, synchronize=True))  # type: ignore
      case _:
        raise ValueError()

  logging.info('Setup experiment')
  exp = setup_experiment(config, tuple(logger_list), callbacks=())
  exp_state = exp.reset(jax.random.key(config.seed))

  # Log number of parameters
  logging.info('Running on {}'.format(jax.default_backend()))
  logging.info(jtu.tree_map(jnp.shape, exp_state.params['params']))
  n_params = jax.flatten_util.ravel_pytree(exp_state.params['params'])[0].shape[0]  # type: ignore
  exp.log(step=0, log_dict=dict(n_params=n_params))

  logging.info('Start training with parametrization')

  config_str = pprint.pformat(flax_traverse.flatten_dict(dataclasses.asdict(config)))
  logging.info(f'\n{config_str}')
  exp_state = exp.run(exp_state)

  # Save experiment state
  # logging.info('Saving experiment state to disk ...')
  # exp.save(exp_state)
  # logging.info('...done')


if __name__ == '__main__':
  # with jax.disable_jit():
  app.run(main)
