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

import dataclasses


@dataclasses.dataclass
class ModelConfig:
  # Policy
  policy_type: str
  d_mlp: int
  d_out: int
  num_hidden_layers: int

  # Transformer
  d_model: int
  d_ffw: int
  n_heads: int
  n_layers: int
  vocab_size: int
  dtype: str = 'bfloat16'
  fsdp: bool = False
  remat: bool = False


@dataclasses.dataclass
class OptimizerConfig:
  learning_rate: float
  weight_decay: float
  optimizer: str
  schedule: str
  warmup_steps: int
  mask_weight_decay: bool = True
  clip_by_global_norm: None | float = None


@dataclasses.dataclass
class HyperTeacherConfig:
  input_dim: int
  output_dim: int
  hidden_dims: tuple[int]
  use_bias: bool
  latent_encoding: str
  num_hot: int
  num_modules: int
  num_samples: int
  scale: float
  task_distribution: str
  task_support: str


@dataclasses.dataclass
class PreferenceGridConfig:
  num_objects: int
  # Preference grid specific
  num_preferences: int
  num_features: int
  num_hot: int
  task_distribution: str
  latent_encoding: str
  timelimit: int
  discount: float


@dataclasses.dataclass
class GoalGridConfig:
  num_objects: int
  num_interactions: int
  num_mazes: int
  num_distractors: int
  grid_size: int
  tokenizer_path: str


@dataclasses.dataclass
class DatasetConfig:
  num_train: int
  num_test: int
  num_ood: int
  task_support: str
  frac_ood: float
  env: GoalGridConfig | PreferenceGridConfig | HyperTeacherConfig


@dataclasses.dataclass
class CallbackConfig:
  mlp_probing: bool
  save_to_disk: bool
  l2_reg: float


@dataclasses.dataclass
class compscaleConfig:
  name: str
  batch_size: int
  seed: int
  model: ModelConfig
  dataset: DatasetConfig
  optimizer: OptimizerConfig
  workdir: str
  log_every: int
  log_level: int
  logger_types: tuple[str, ...]
  callbacks: CallbackConfig
