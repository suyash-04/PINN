# Physics-Informed Neural Network (PINN) for Landslide Stability Analysis

A modular Python implementation of a Physics-Informed Neural Network (PINN) to estimate landslide stability (Factor of Safety) using HYDRUS-1D simulation outputs and geotechnical parameters.

This repository implements data ingestion, configurable PINN architecture, custom physics-aware losses, and a training pipeline with checkpointing and logging.

---

## Table of contents
- Project overview
- Installation & requirements
- Project structure
- Configuration
- Data ingestion & processing
- Model architecture (PINN)
- Training procedure
- Usage / Examples
- Logging & artifacts
- Saving & loading models / batches
- Contributing & license

## Overview / Description

This project provides a production-ready, modular PINN implementation for landslide stability analysis. HYDRUS‑1D output files (for example `dataset/Nod_Inf.out`) are parsed into training tensors. The PINN learns to approximate relevant fields and enforces geotechnical and physical constraints through custom loss terms. Geotechnical features such as cohesion (c'), friction angle (φ'), unit weight (γ), slope angle (β), soil water retention parameters, and hydraulic conductivity (Ks) are used to compute the Factor of Safety (FS) and to inform physics losses.

Key features
- Config-driven architecture and training (YAML files: `config/config.yaml`, `params.yaml`).
- Modular components: configuration, data ingestion, model architecture, custom loss, training pipeline.
- Uses PyTorch for model definition and training; uses pandas/numpy for data handling.
- Training supports checkpointing, tensor batch saving/loading (`torch.save` / `torch.load`).
- Structured logging for all major steps.

## Installation / Requirements

This project targets Python 3.12.3. Install dependencies listed in `requirements.txt` (PyTorch is a required runtime dependency; specify a compatible CUDA build if you plan to use GPU acceleration).

Recommended steps (example, using a Python venv or conda):

```bash
# create and activate a virtual environment (example shown for bash/fish compatible commands)
python -m venv .venv
. .venv/bin/activate  # or `source .venv/bin/activate` in bash

# install dependencies
pip install -r requirements.txt

# install PyTorch separately following instructions for your CUDA version from https://pytorch.org/
```

Minimum relevant packages (from `requirements.txt`):
- deepxde >= 1.9.3
- numpy, pandas, scipy, matplotlib, pyyaml

Note: PyTorch is required but intentionally left out of `requirements.txt` to let you choose the best binary for your platform. Install PyTorch (CPU or CUDA) per PyTorch installation instructions.

## Project Structure

Top-level layout (most important files and folders):

```
README.md
main.py                # pipeline runner that executes ingestion, loading, training
config/                # YAML configuration (config/config.yaml)
params.yaml            # model/training/geotechnical params
dataset/               # raw HYDRUS output (e.g. Nod_Inf.out)
artifacts/             # produced artifacts: datasets, batches, model checkpoints, logs
src/pinn_landslide/
  components/          # core modules: data_ingestion, data_loader, pinn_architecture, loss, trainer
  config/              # configuration manager
  utils/               # utilities (read yaml, create dirs, helpers)
  logger/              # logging wrapper
  pipeline/            # high-level pipeline stages
```

Detailed module responsibilities:
- `src/pinn_landslide/components/data_ingestion.py` — parse HYDRUS outputs to CSV or structured data.
- `src/pinn_landslide/components/data_loader.py` — build PyTorch tensors / batches from parsed data.
- `src/pinn_landslide/components/pinn_architecture.py` — configurable neural network (hidden layers, neurons, activation, in/out dims).
- `src/pinn_landslide/components/loss.py` — custom loss combining physics, boundary, anchor, and failure losses.
- `src/pinn_landslide/components/trainer.py` — training loop, checkpointing, scheduler, metrics.
- `src/pinn_landslide/config/configuration.py` — configuration loader/manager.
- `src/pinn_landslide/utils/utils.py` — helpers (create dirs, read yaml, seed, device selection, etc.).

## Configuration

Two YAML files control behavior:

- `config/config.yaml` — artifact paths and dataset/model output locations.
- `params.yaml` — model architecture, training hyperparameters, geotechnical parameters, loss weights.

Example (key fields):

```yaml
# config/config.yaml
artifacts_root: artifacts
data_ingestion:
  root: artifacts/dataset
  raw_data_file: dataset/Nod_Inf.out
  ingested_data: artifacts/dataset/final_data.csv
  batch_path: artifacts/dataset/data_batch.pt

model_training:
  root: artifacts/model
  model_path: artifacts/model/pinn_model.pt
```

```yaml
# params.yaml
architecture:
  n_hidden_layers: 7
  neurons_per_layer: 64
  activation: tanh
  input_dim: 2
  output_dim: 1

training:
  epochs: 5000
  learning_rate: 0.001
  device: cpu

geo_params:
  Ks: 1.5e-5
  theta_s: 0.45
  theta_r: 0.05
  alpha: 0.8
  n: 1.4
  l: 0.5
  c_prime: 5.0
  phi_prime: 28.0
  gamma: 20.0
  beta: 42.0
  gamma_w: 9.81

loss_weights:
  lambda_physics: 1.0
  lambda_anchor: 10.0
  lambda_initial: 5.0
  lambda_boundary: 5.0
  lambda_failure: 20.0
```

The repository includes a configuration manager implemented in `src/pinn_landslide/config/configuration.py` that reads and validates these YAMLs and exposes convenient accessors used by the pipelines.

## Data Ingestion / Processing

Raw HYDRUS-1D output files (for example `dataset/Nod_Inf.out`) are ingested and converted to a tabular CSV (`artifacts/dataset/final_data.csv`) and to serialized PyTorch batches (`artifacts/dataset/data_batch.pt`). The ingestion stage:

- Parses HYDRUS output into time/space fields and related poro-hydraulic information.
- Computes derived features used by the PINN and geotechnical calculations, including the Factor of Safety (FS).
- Saves processed CSV and a pre-batched PyTorch file with tensors for fast experiments.

Typical data ingestion usage is via the pipeline in `main.py` or by importing `DataIngestionPipeline`:

```python
from src.pinn_landslide.pipeline.stage_1_data_ingestion_pipeline import DataIngestionPipeline

ing = DataIngestionPipeline()
ing.run()  # creates artifacts/dataset/final_data.csv and batch files
```

## Model Architecture (PINN)

The neural network architecture is configurable via `params.yaml` (number of hidden layers, neurons, activation, input/output dimensions). The architecture module constructs a standard feed-forward network that the PINN uses for function approximation. The physics-informed losses are applied to network outputs and derivatives (computed with PyTorch autograd) to encode governing equations and constraints.

Example of building the model (illustrative):

```python
from src.pinn_landslide.components.pinn_architecture import PINN
from src.pinn_landslide.config.configuration import Configuration

cfg = Configuration('config/config.yaml', 'params.yaml')
params = cfg.params

model = PINN(input_dim=params['architecture']['input_dim'],
             output_dim=params['architecture']['output_dim'],
             n_hidden=params['architecture']['n_hidden_layers'],
             neurons=params['architecture']['neurons_per_layer'],
             activation=params['architecture']['activation'])
```

The PINN component uses autograd to compute spatial and temporal derivatives required by physics loss terms.

## Training Procedure

The training pipeline supports:

- Config-driven hyperparameters: epochs, learning rate, device selection (CPU/GPU).
- Composite loss: a weighted sum of physics, anchor, initial, boundary, and failure losses (weights in `params.yaml`).
- Checkpointing: model state and optimizer state are saved periodically to `artifacts/model/pinn_model.pt` (or configured path).
- Batch serialization: precomputed batches are saved/loaded using `torch.save` / `torch.load` for fast experiments.

High-level training usage (via `main.py`):

```bash
python main.py
```

Or programmatically:

```python
from src.pinn_landslide.pipeline.stage_3_model_training import ModelTrainingPipeline

trainer = ModelTrainingPipeline()
trainer.run()  # runs training loop, saves checkpoints and logs
```

Inside the trainer, the loop looks like:

```python
for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = composite_loss(outputs, targets, physics_terms, params)
    loss.backward()
    optimizer.step()

    if epoch % save_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
```

The training code automatically selects device:

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

## Usage / Example

Quick start:

1. Ensure `config/config.yaml` and `params.yaml` are configured for your environment.
2. Place HYDRUS output (e.g., `Nod_Inf.out`) into `dataset/` or change the `raw_data_file` path in `config/config.yaml`.
3. Run the pipeline:

```bash
python main.py
```

This executes the three stages: data ingestion, data loader creation, and model training. Check `logs/running_logs.log` (or configured logger) for progress and details.

If you prefer to run stages separately, import the pipeline classes directly (see code snippets in the repository). The pipelines are implemented as classes under `src/pinn_landslide/pipeline/`.

## Logging and Artifacts

- Logging is implemented in `src/pinn_landslide/logger/logger.py`. Logs are written for every major step (ingestion, batch creation, training epochs, checkpoints).
- Artifacts (CSV, batch tensors, checkpoints) are saved under `artifacts/` by default. Paths are configurable via `config/config.yaml`.

Common artifact locations (defaults):

- Dataset CSV: `artifacts/dataset/final_data.csv`
- Batch file: `artifacts/dataset/data_batch.pt`
- Model checkpoint: `artifacts/model/pinn_model.pt`
- Logs: `logs/running_logs.log`

## Contributing / License

Contributions are welcome. Please open issues or pull requests for bug fixes, improvements, or new features. Follow common best practices:

- Create a feature branch, add tests where applicable, and open a pull request.
- Keep changes modular and documented.

This repository includes a `LICENSE` file at the project root — please review it for the licensing terms.

## Notes & Next Steps

- If you plan to run training on GPU, install the corresponding CUDA-enabled PyTorch binary.
- Consider adding unit tests for data parsing and loss functions to protect physics constraints and numerical stability.
- Optionally add a simple script/notebook to visualize HYDRUS fields and predicted FS maps.

If you'd like, I can also add a short Quickstart notebook or a Makefile / CLI wrapper to simplify experiments.

---

If anything should be adjusted (tone, extra examples, or added badges), tell me what you'd like and I will update the README.
# PINN