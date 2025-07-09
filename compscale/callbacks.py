"""
Copyright (c) Florian Redhardt
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

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from compscale.experiment import Callback
from compscale.experiment import CallbackEvent
from compscale.experiment import Metrics
from compscale.model.regression import RidgeRegression


class ExtendedActivationProbing(Callback):
  """
  Probes activations to predict both latent codes and target network effective weights.
  Produces concise output with only the most essential metrics.
  """

  def __init__(
    self,
    log_level: int,
    onevent: CallbackEvent,
    save_to_disk: bool,
    l2_reg: float,
    test_loader_name: str = 'test',
    ood_loader_name: str = 'ood',
    teacher_ref=None,
  ) -> None:
    super().__init__(log_level, onevent)
    self.teacher_ref = teacher_ref
    self.save_to_disk = save_to_disk
    self.l2_reg = l2_reg
    self.test_loader_name = test_loader_name
    self.ood_loader_name = ood_loader_name
    self.component_layers = None  # Will be populated dynamically

    # Cache the compiled functions
    self._fit_and_score_fn = None

  def _extract_layer_structure(self, activations):
    """
    Dynamically detect the model's layer structure.
    Returns a dictionary mapping component names to their layers.
    """
    components = {}

    # Extract all top-level components (transformer, policy, etc.)
    for component_name, component_data in activations.items():
      # Skip the top-level __call__ key
      if component_name == '__call__':
        continue

      # Store the layers for each component
      component_layers = []
      if isinstance(component_data, dict):
        for layer_name in component_data.keys():
          # Skip the component's __call__ key
          if layer_name != '__call__':
            component_layers.append(layer_name)

        # Only add components that have layers
        if component_layers:
          components[component_name] = sorted(component_layers)

    return components

  def _compute_target_weights(self, latent, layer_idx):
    """
    Compute the effective target network weights for a given latent code and layer.
    """
    if self.teacher_ref is None or layer_idx >= len(self.teacher_ref.modules):
      return None

    # Get the teacher modules for this layer
    teacher_module = self.teacher_ref.modules[layer_idx]

    # Compute effective weights using the same approach as in forward_network
    # Scale the latent to keep invariance to num_hot
    scaled_latent = latent / jnp.sqrt(self.teacher_ref.num_hot)

    # Compute the effective weights for this layer
    effective_weights = jnp.einsum('mih,m->ih', teacher_module, scaled_latent)

    return effective_weights

  def __call__(self, ctx, exp_state) -> Metrics:
    @jax.jit
    def extract_activations(params, batch):
      # Create a dummy RNG key for the model
      rng = jax.random.key(0)
      rngs = {'target': rng, 'dropout': rng}

      _, variables = ctx.model.apply(
        params,
        batch,
        mutable=['intermediates'],
        capture_intermediates=True,
        rngs=rngs,
      )

      # Get all intermediates
      intermediates = variables.get('intermediates', {})

      # If batch.z exists, treat it as another "layer" in the activations
      z = getattr(batch, 'z', None)
      if z is not None:
        # Get the model's dtype directly from config
        model_dtype = ctx.config.model.dtype

        # Convert string dtype to jnp dtype
        if model_dtype == 'float32':
          dtype = jnp.float32
        elif model_dtype == 'float64':
          dtype = jnp.float64
        elif model_dtype == 'bfloat16':
          dtype = jnp.bfloat16
        else:
          raise ValueError(f'Unsupported dtype: {model_dtype}')

        # Convert to model's dtype to ensure consistent types
        z = z.astype(dtype)

        if len(z.shape) == 1:
          z = z[:, None, None]
        else:
          z = z[:, None, :]

        if 'policy' not in intermediates:
          intermediates['policy'] = {}
        intermediates['policy']['z_layer'] = {'__call__': (z, None)}

      return {
        'activations': intermediates,
        'latent': batch.info.get('latents', None),
        'y': batch.y,
      }

    # Get test data for training and evaluation
    test_train_activations = None
    test_train_latents = None
    test_eval_activations = None
    test_eval_latents = None

    # Process test data
    if self.test_loader_name in ctx.eval_loaders:
      test_loader = ctx.eval_loaders[self.test_loader_name]
      all_activations = []
      all_latents = []

      # Extract data from all batches
      for batch in test_loader:
        result = extract_activations(exp_state.params, batch)
        all_activations.append(result['activations'])
        all_latents.append(result['latent'])

      # Dynamically detect components and layers if not already done
      if self.component_layers is None and all_activations:
        self.component_layers = self._extract_layer_structure(all_activations[0])
        print(f'Detected components and layers: {self.component_layers}')

      # Combine data from all batches for each component and layer
      combined_activations = {}

      # Process each component
      for component_name, layer_names in self.component_layers.items():
        # Create a component dictionary if it doesn't exist
        if component_name not in combined_activations:
          combined_activations[component_name] = {}

        # Process each layer in the component
        for layer_name in layer_names:
          layer_activations = []

          for batch_act in all_activations:
            if component_name in batch_act and layer_name in batch_act[component_name]:
              call_data = batch_act[component_name][layer_name].get('__call__', None)

              if (
                isinstance(call_data, tuple)
                and len(call_data) > 0
                and call_data[0] is not None
              ):
                # Keep as JAX arrays
                layer_activations.append(call_data[0])

          if layer_activations:
            # Use JAX concatenate for larger arrays
            combined_activations[component_name][layer_name] = jnp.concatenate(
              layer_activations, axis=0
            )

      # Combine all latents
      combined_latents = (
        jnp.concatenate(all_latents, axis=0)
        if all_latents and all(l is not None for l in all_latents)
        else None
      )

      if combined_latents is not None:
        # Split into training and evaluation sets (50/50)
        num_samples = combined_latents.shape[0]
        split_idx = num_samples // 2

        # Split latents
        test_train_latents = combined_latents[:split_idx]
        test_eval_latents = combined_latents[split_idx:]

        # Split activations for each component and layer
        test_train_activations = {}
        test_eval_activations = {}

        for component_name, layers in combined_activations.items():
          if component_name not in test_train_activations:
            test_train_activations[component_name] = {}
            test_eval_activations[component_name] = {}

          for layer_name, activation in layers.items():
            test_train_activations[component_name][layer_name] = activation[:split_idx]
            test_eval_activations[component_name][layer_name] = activation[split_idx:]

    # Process OOD data
    ood_activations = None
    ood_latents = None

    if self.ood_loader_name in ctx.eval_loaders:
      ood_loader = ctx.eval_loaders[self.ood_loader_name]
      all_activations = []
      all_latents = []

      # Storage for network R² values for teacher vs student comparison
      r2_ood_batches = []

      # Extract data from all OOD batches
      for batch in ood_loader:
        result = extract_activations(exp_state.params, batch)
        all_activations.append(result['activations'])
        all_latents.append(result['latent'])

        if self.teacher_ref is not None:
          # Calculate network R² (teacher vs student) for this batch
          preds = ctx.model.apply(
            exp_state.params,
            batch,
            mutable=False,
            rngs={'dropout': jax.random.key(0), 'target': jax.random.key(0)},
          )
          y_true = batch.y
          y_pred = preds
          mse = jnp.mean((y_true - y_pred) ** 2, axis=1)
          var = jnp.mean((y_true - jnp.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
          r2_t = 1.0 - jnp.mean(mse, axis=1) / (jnp.mean(var, axis=1) + 1e-9)
          r2_ood_batches.append(r2_t)

      # Combine OOD data
      combined_activations = {}

      # Process each component
      for component_name, layer_names in self.component_layers.items():
        # Create a component dictionary if it doesn't exist
        if component_name not in combined_activations:
          combined_activations[component_name] = {}

        # Process each layer in the component
        for layer_name in layer_names:
          layer_activations = []

          for batch_act in all_activations:
            if component_name in batch_act and layer_name in batch_act[component_name]:
              call_data = batch_act[component_name][layer_name].get('__call__', None)

              if (
                isinstance(call_data, tuple)
                and len(call_data) > 0
                and call_data[0] is not None
              ):
                # Keep as JAX arrays
                layer_activations.append(call_data[0])

          if layer_activations:
            # Use JAX concatenate for larger arrays
            combined_activations[component_name][layer_name] = jnp.concatenate(
              layer_activations, axis=0
            )

      # Combine all OOD latents
      ood_latents = (
        jnp.concatenate(all_latents, axis=0)
        if all_latents and all(l is not None for l in all_latents)
        else None
      )
      ood_activations = combined_activations

      # Combine network R² values (teacher vs student)
      if r2_ood_batches:
        r2_ood_vec = jnp.concatenate(r2_ood_batches, axis=0)

    # Save to disk if requested
    if self.save_to_disk:
      # Create a dictionary of datasets
      datasets_dict = {
        'test_train': {
          'activations': test_train_activations,
          'latent': test_train_latents,
        },
        'test_eval': {'activations': test_eval_activations, 'latent': test_eval_latents},
      }

      if ood_latents is not None:
        datasets_dict['ood'] = {'activations': ood_activations, 'latent': ood_latents}

      # Convert to numpy for easier serialization
      datasets_numpy = {}
      for name, dataset in datasets_dict.items():
        # Convert activations component by component
        activations_numpy = {}

        for component_name, layers in dataset['activations'].items():
          activations_numpy[component_name] = {}

          for layer_name, activation in layers.items():
            activations_numpy[component_name][layer_name] = np.array(activation)

        datasets_numpy[name] = {
          'activations': activations_numpy,
          'latent': np.array(dataset['latent'])
          if dataset['latent'] is not None
          else None,
        }

      # Save to disk
      os.makedirs(ctx.config.workdir, exist_ok=True)
      pickle_path = os.path.join(
        ctx.config.workdir, f'activations_step_{exp_state.step}.pkl'
      )
      with open(pickle_path, 'wb') as f:
        pickle.dump(datasets_numpy, f)

      print(f'Saved activations to {pickle_path}')

    # Perform linear probing with our optimized approach
    metrics = {}

    # Skip if we don't have training data
    if test_train_activations is None or test_train_latents is None:
      return metrics

    # Create or reuse JIT-compiled ridge regression function for prediction
    if self._fit_and_score_fn is None:

      @jax.jit
      def fit_and_score(x_train, y_train_dim, x_test, y_test_dim, l2_reg):
        # Create ridge regression model
        reg = RidgeRegression(feature_dim=x_train.shape[1], l2_reg=l2_reg, intercept=True)

        # Initialize and fit
        params = reg.init(None, x_train)
        params = reg.fit(params, x_train, y_train_dim)

        # Score
        return reg.score(params, x_test, y_test_dim)

      self._fit_and_score_fn = fit_and_score

    # Process test data for each component and layer for latent prediction
    for component_name, layers in test_train_activations.items():
      for layer_name, activation in layers.items():
        # Process activations - use the last token in the sequence for transformers
        # or the whole activation for other components
        if activation.shape[1] == 1:
          x_train = activation[:, 0, :]
        else:
          x_train = activation[:, -1, :]  # Use last token for sequence data

        y_train = test_train_latents

        # Get test data
        test_activation = test_eval_activations[component_name][layer_name]
        if test_activation.shape[1] == 1:
          x_test = test_activation[:, 0, :]
        else:
          x_test = test_activation[:, -1, :]

        y_test = test_eval_latents

        # Train and evaluate for each latent dimension
        latent_dim = y_train.shape[1]
        test_r2_scores = []

        # Arrays are already JAX arrays - no conversion needed
        x_train_jax = x_train
        x_test_jax = x_test

        for dim in range(latent_dim):
          try:
            # Get this dimension's data - already JAX arrays
            y_train_dim = y_train[:, dim]
            y_test_dim = y_test[:, dim]

            # Use JIT-compiled function
            r2 = float(
              self._fit_and_score_fn(
                x_train_jax, y_train_dim, x_test_jax, y_test_dim, self.l2_reg
              )
            )
            test_r2_scores.append(r2)
          except Exception as e:
            print(f'Error for {component_name}/{layer_name}, dimension {dim}: {e}')

        # Compute mean R² for this layer
        if test_r2_scores:
          mean_r2 = sum(test_r2_scores) / len(test_r2_scores)
          metrics[f'r2_test_{component_name}_{layer_name}'] = float(mean_r2)

        # Process OOD data if available
        if (
          ood_activations is not None
          and ood_latents is not None
          and component_name in ood_activations
          and layer_name in ood_activations[component_name]
        ):
          ood_activation = ood_activations[component_name][layer_name]
          if ood_activation.shape[1] == 1:
            x_ood = ood_activation[:, 0, :]
          else:
            x_ood = ood_activation[:, -1, :]

          y_ood = ood_latents

          # Already JAX arrays - no conversion needed
          x_ood_jax = x_ood

          # Per-dimension R² scores (original approach)
          ood_r2_scores = []

          # For per-task R² scores (new addition for comparison with network R²)
          # We'll train models for each dimension and calculate task-level R²

          # Store fitted models for each dimension
          fitted_models = []

          for dim in range(latent_dim):
            try:
              # Get this dimension's data - already JAX arrays
              y_train_dim = y_train[:, dim]
              y_ood_dim = y_ood[:, dim]

              # Use JIT-compiled function for dimension-level R²
              r2 = float(
                self._fit_and_score_fn(
                  x_train_jax, y_train_dim, x_ood_jax, y_ood_dim, self.l2_reg
                )
              )
              ood_r2_scores.append(r2)

              if self.save_to_disk:
                # Train a model for this dimension
                reg = RidgeRegression(
                  feature_dim=x_train_jax.shape[1], l2_reg=self.l2_reg, intercept=True
                )
                params = reg.init(None, x_train_jax)
                params = reg.fit(params, x_train_jax, y_train_dim)
                fitted_models.append((reg, params))

            except Exception as e:
              print(f'Error for OOD {component_name}/{layer_name}, dimension {dim}: {e}')
              fitted_models.append(None)  # Keep placeholder for failed models

          # Now calculate task-level R² using the fitted models
          if (
            fitted_models
            and any(m is not None for m in fitted_models)
            and self.save_to_disk
          ):
            # Initialize arrays for predictions and actual values
            all_preds = jnp.zeros((y_ood.shape[0], latent_dim))
            all_true = y_ood

            # Make predictions for each dimension
            for dim, model_info in enumerate(fitted_models):
              if model_info is not None:
                reg, params = model_info
                # Use apply instead of predict
                dim_preds = reg.apply(params, x_ood_jax)
                # Update the predictions array
                all_preds = all_preds.at[:, dim].set(dim_preds)

            # Calculate task-level MSE
            task_mse = jnp.mean((all_preds - all_true) ** 2, axis=1)

            # Calculate task-level variance
            task_var = jnp.mean(
              (all_true - jnp.mean(all_true, axis=1, keepdims=True)) ** 2, axis=1
            )

            # Calculate task-level R²
            task_r2 = 1.0 - task_mse / (task_var + 1e-9)

            # Save the per-task R² values
            np.save(
              os.path.join(
                ctx.config.workdir,
                f'latent_r2_ood_task_{component_name}_{layer_name}_step_{exp_state.step}.npy',
              ),
              np.array(task_r2),
            )

          # Compute mean OOD R² for this layer (dimension-level)
          if ood_r2_scores:
            mean_ood_r2 = sum(ood_r2_scores) / len(ood_r2_scores)
            metrics[f'r2_ood_{component_name}_{layer_name}'] = float(mean_ood_r2)

            if self.save_to_disk:
              # Save the per-dimension R² scores for OOD data
              np.save(
                os.path.join(
                  ctx.config.workdir,
                  f'latent_r2_ood_{component_name}_{layer_name}_step_{exp_state.step}.npy',
                ),
                np.array(ood_r2_scores),
              )

    # Save network R² values for OOD data if available
    if 'r2_ood_vec' in locals() and self.save_to_disk:
      # Store mean network R² for OOD data
      mean_r2_ood = float(jnp.mean(r2_ood_vec))
      metrics['network_r2_ood'] = mean_r2_ood

      if self.save_to_disk:
        # Save the vector of R² values for plotting
        np.save(
          os.path.join(
            ctx.config.workdir, f'network_r2_ood_vec_step_{exp_state.step}.npy'
          ),
          np.array(r2_ood_vec),
        )

    if self.teacher_ref is not None and self.save_to_disk:
      # Now, add the target weights prediction part
      # Process each layer of the teacher network
      for layer_idx in range(len(self.teacher_ref.modules)):
        # Compute target network weights for each sample in training, testing, and OOD sets
        print(f'Computing target weights for layer {layer_idx}')

        # Pre-compute target weights for training samples
        train_target_weights = []
        for i in range(test_train_latents.shape[0]):
          latent = test_train_latents[i]
          weights = self._compute_target_weights(latent, layer_idx)
          if weights is not None:
            train_target_weights.append(weights.reshape(-1))

        # Pre-compute target weights for test samples
        test_target_weights = []
        for i in range(test_eval_latents.shape[0]):
          latent = test_eval_latents[i]
          weights = self._compute_target_weights(latent, layer_idx)
          if weights is not None:
            test_target_weights.append(weights.reshape(-1))

        # Pre-compute target weights for OOD samples
        ood_target_weights = []
        if ood_latents is not None:
          for i in range(ood_latents.shape[0]):
            latent = ood_latents[i]
            weights = self._compute_target_weights(latent, layer_idx)
            if weights is not None:
              ood_target_weights.append(weights.reshape(-1))

        # Skip if we couldn't compute weights
        if not train_target_weights or not test_target_weights:
          print(f"  Skipping layer {layer_idx} - couldn't compute target weights")
          continue

        # Convert to arrays
        train_target_weights = jnp.stack(train_target_weights)
        test_target_weights = jnp.stack(test_target_weights)

        if ood_target_weights:
          ood_target_weights = jnp.stack(ood_target_weights)
          print(f'  Processing weights: {train_target_weights.shape[1]} dimensions')
        else:
          print(f'  Processing weights: {train_target_weights.shape[1]} dimensions')

        # Process each component and layer of student activations
        for component_name, layers in test_train_activations.items():
          for layer_name, activation in layers.items():
            # Process activations - training set
            if activation.shape[1] == 1:
              x_train = activation[:, 0, :]
            else:
              x_train = activation[:, -1, :]

            # Get test set activations
            test_activation = test_eval_activations[component_name][layer_name]
            if test_activation.shape[1] == 1:
              x_test = test_activation[:, 0, :]
            else:
              x_test = test_activation[:, -1, :]

            # Get OOD activations if available
            x_ood = None
            if (
              ood_activations is not None
              and ood_target_weights is not None
              and component_name in ood_activations
              and layer_name in ood_activations[component_name]
            ):
              ood_activation = ood_activations[component_name][layer_name]
              if ood_activation.shape[1] == 1:
                x_ood = ood_activation[:, 0, :]
              else:
                x_ood = ood_activation[:, -1, :]

            print(f'  Predicting {component_name}/{layer_name}...')

            # Process all weight dimensions for test set predictions
            weight_dim = train_target_weights.shape[1]
            test_weight_r2_sum = 0.0
            test_weight_r2_count = 0

            # Process all dimensions directly (no intermediate results printed)
            for idx in range(weight_dim):
              try:
                # Extract this dimension from the target weights
                y_train_weight = train_target_weights[:, idx]
                y_test_weight = test_target_weights[:, idx]

                # Train and evaluate the predictor
                r2 = float(
                  self._fit_and_score_fn(
                    x_train, y_train_weight, x_test, y_test_weight, self.l2_reg
                  )
                )
                test_weight_r2_sum += r2
                test_weight_r2_count += 1
              except Exception as e:
                if idx == 0:  # Only print first error
                  print(f'  Error predicting test weight: {e!s}')

            # Compute mean test R² for this component/layer predicting this teacher layer
            if test_weight_r2_count > 0:
              mean_test_r2 = test_weight_r2_sum / test_weight_r2_count
              metrics[
                f'weights_r2_test_layer{layer_idx}_{component_name}_{layer_name}'
              ] = float(mean_test_r2)
              print(f'  Test R² = {mean_test_r2:.4f} ({test_weight_r2_count} dimensions)')

            # Predict all dimensions for OOD set
            if x_ood is not None and ood_target_weights is not None:
              ood_weight_r2_sum = 0.0
              ood_weight_r2_count = 0

              # Process all dimensions directly (no intermediate results printed)
              for idx in range(weight_dim):
                try:
                  # Extract this dimension from the target weights
                  y_train_weight = train_target_weights[:, idx]
                  y_ood_weight = ood_target_weights[:, idx]

                  # Train and evaluate the predictor
                  r2 = float(
                    self._fit_and_score_fn(
                      x_train, y_train_weight, x_ood, y_ood_weight, self.l2_reg
                    )
                  )
                  ood_weight_r2_sum += r2
                  ood_weight_r2_count += 1
                except Exception as e:
                  if idx == 0:  # Only print first error
                    print(f'  Error predicting OOD weight: {e!s}')

              # Compute mean OOD R² for this component/layer predicting this teacher layer
              if ood_weight_r2_count > 0:
                mean_ood_r2 = ood_weight_r2_sum / ood_weight_r2_count
                metrics[
                  f'weights_r2_ood_layer{layer_idx}_{component_name}_{layer_name}'
                ] = float(mean_ood_r2)
                print(f'  OOD R² = {mean_ood_r2:.4f} ({ood_weight_r2_count} dimensions)')

    return metrics
