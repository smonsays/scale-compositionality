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

import math

from compscale import config_classes


def get_config() -> config_classes.compscaleConfig:
  batch_size = 128
  hyperteacher_config = config_classes.HyperTeacherConfig(
    input_dim=16,
    output_dim=16,
    hidden_dims=(16,),
    use_bias=True,
    latent_encoding='identity',
    num_hot=3,
    num_modules=16,
    num_samples=16,
    scale=math.sqrt(3.0),
    task_distribution='discrete',
    task_support='random',
  )

  dataset = config_classes.DatasetConfig(
    num_train=batch_size * 1_000_00,
    num_test=1024,
    num_ood=1024,
    task_support='random',
    frac_ood=0.25,
    env=hyperteacher_config,
  )

  optimizer = config_classes.OptimizerConfig(
    learning_rate=0.0003,
    warmup_steps=1000,
    optimizer='adamw',
    schedule='constant',
    weight_decay=0.003,
    clip_by_global_norm=None,  # 1.0 is common for many well-known LLMs.
    mask_weight_decay=True,
  )

  model = config_classes.ModelConfig(
    policy_type='mlp',
    d_mlp=1024,
    num_hidden_layers=4,
    d_out=-1,
    d_model=256,  # model/embed dim  = qkv dim
    n_heads=4,  # num attention heads
    n_layers=4,  # number of transformer block layers
    d_ffw=1024,  # FF inner dimension
    vocab_size=32000,
    dtype='float32',  # computation dtype.
    fsdp=False,  # True to shard the model.
    remat=False,  # Transformer block gradient checkpointing to save memory.
  )

  callbacks = config_classes.CallbackConfig(
    mlp_probing=True,
    save_to_disk=False,
    l2_reg=1.0,
  )

  config = config_classes.compscaleConfig(
    name='hyperteacher_default',
    batch_size=batch_size,
    seed=43,
    optimizer=optimizer,
    dataset=dataset,
    model=model,
    workdir='logs',
    log_every=10000,
    log_level=1,
    logger_types=('standard', 'wandb'),
    callbacks=callbacks,
  )

  return config
