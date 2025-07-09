# Scale leads to compositional generalization

Official code for the paper [Scale leads to compositional generalization](#).

Code to run the image generation experiments can be found [in this repository](https://github.com/florian-toll/compgen-vision).


## 🚧 Installation

We use `uv` for Python dependency management.
To install all necessary dependencies, run the following command from the project root:

```bash
uv sync 
```

## 🧪 Running experiments

Experiments have a corresponding sweep file in `sweeps/` and can be run using
```bash
`uv run wandb sweep /sweeps/[name].yaml`
```
Default hyperparameters for all methods and experiments can be found in `configs/`.
If you'd like to directly run a specific experiment for a single seed you can use:

```bash
uv run run.py --config 'configs/[experiment].py'
```

where `experiment` $\in$ [`teacher`, `preference`].
