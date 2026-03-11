# 🏔️ PINN for Landslide Stability Analysis

> Physics-Informed Neural Network for predicting subsurface pore-water pressure and slope stability under rainfall infiltration, with a production-grade React dashboard.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-see%20LICENSE-blue)](#license)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [PINN Training Pipeline](#pinn-training-pipeline)
- [Interactive Dashboard (React)](#interactive-dashboard-react)
- [Streamlit Dashboard (Legacy)](#streamlit-dashboard-legacy)
- [Configuration](#configuration)
- [Physics & Governing Equations](#physics--governing-equations)
- [Model Architecture](#model-architecture)
- [API Reference](#api-reference)
- [Artifacts & Logging](#artifacts--logging)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** that learns to solve **Richards' equation** for unsaturated water flow in hillslopes, and computes the **Factor of Safety (FS)** using the infinite-slope stability model with matric suction.

The PINN is trained on **HYDRUS-1D** simulation data (24,000 points across 24 timesteps × 1,000 depth nodes) and enforces governing PDEs, boundary conditions, and initial conditions directly in the loss function.

A **production-grade React + FastAPI dashboard** with 17 interactive pages provides real-time model inference, sensitivity analysis, uncertainty quantification, and comprehensive validation against HYDRUS-1D reference solutions.

## Key Features

### 🧠 PINN Training
- Config-driven architecture and training via YAML files
- 7 hidden layers × 64 neurons, tanh activation (~29K parameters)
- Composite loss: data MSE + PDE residual + initial conditions + boundary conditions + failure constraints
- Two-phase training: Adam optimizer → L-BFGS fine-tuning
- Automatic device selection (CPU / CUDA / Apple Silicon MPS)
- Checkpointing, batch serialization, and structured logging

### 📊 Interactive Dashboard (React)
- **17 interactive pages** organized into Core, Advanced, and Research sections
- Real-time PINN inference via FastAPI backend
- Interactive Plotly.js charts with zoom, pan, and hover
- Adjustable geotechnical parameters with live re-computation
- Monte Carlo uncertainty quantification
- Data export in CSV/JSON formats
- Responsive design with collapsible sidebar (desktop + mobile)

### 🔬 Physics
- **Richards' Equation** — 1D unsaturated vertical flow PDE
- **Van Genuchten** — soil-water retention and hydraulic conductivity
- **Mohr–Coulomb** — shear-strength failure criterion
- **Infinite-Slope Model** — factor of safety with matric suction contribution

## Screenshots

| Overview Dashboard | Factor of Safety Contour | HYDRUS vs PINN |
|:---:|:---:|:---:|
| Metric cards, ψ & FS profiles | 2D heatmap with stability stats | Overlay, scatter, error profiles |

| Model Architecture | Soil Properties | Uncertainty (Monte Carlo) |
|:---:|:---:|:---:|
| Network topology diagram | Van Genuchten WRC, K(ψ), C(ψ) | FS distribution & reliability bands |

---

## Project Structure

```
PINN/
├── main.py                          # Training pipeline runner
├── config/config.yaml               # Artifact paths & dataset config
├── params.yaml                      # Architecture, training, geotechnical params
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── dataset/                         # Raw HYDRUS-1D output (Nod_Inf.out)
├── artifacts/                       # Generated artifacts
│   ├── dataset/
│   │   ├── final_data.csv           #   Processed training data (24K points)
│   │   └── data_batch.pt            #   Pre-batched PyTorch tensors
│   └── model/
│       └── pinn_model.pt            #   Trained model checkpoint
│
├── src/pinn_landslide/              # Core PINN library
│   ├── components/
│   │   ├── data_ingestion.py        #   HYDRUS output → CSV parser
│   │   ├── data_loader.py           #   CSV → PyTorch tensor batches
│   │   ├── pinn_architecture.py     #   Configurable feed-forward network
│   │   ├── loss.py                  #   Composite physics-aware loss
│   │   └── trainer.py               #   Training loop with checkpointing
│   ├── config/configuration.py      #   YAML config loader/manager
│   ├── pipeline/                    #   3-stage pipeline (ingest → load → train)
│   ├── logger/                      #   Structured logging
│   └── utils/                       #   Helpers (read YAML, create dirs, seed)
│
├── ui-react/                        # ⭐ Production React Dashboard
│   ├── api/                         #   FastAPI backend
│   │   ├── main.py                  #     16 REST endpoints
│   │   └── schemas.py               #     Pydantic request/response models
│   └── frontend/                    #   React 19 + Vite 7 frontend
│       ├── src/
│       │   ├── App.jsx              #     Router with 17 routes
│       │   ├── api.js               #     API client
│       │   ├── context.jsx          #     Global state (geo params, norm)
│       │   ├── components/
│       │   │   ├── Layout.jsx       #     Sidebar + responsive shell
│       │   │   ├── Chart.jsx        #     Plotly wrapper
│       │   │   └── ui.jsx           #     Shared UI components
│       │   └── pages/               #     17 page components
│       │       ├── Overview.jsx
│       │       ├── PressureHead.jsx
│       │       ├── FactorOfSafety.jsx
│       │       ├── ParameterExplorer.jsx
│       │       ├── HydrusComparison.jsx
│       │       ├── SoilProperties.jsx
│       │       ├── ModelInfo.jsx
│       │       ├── Animation.jsx
│       │       ├── TrainingLoss.jsx
│       │       ├── PDEResidual.jsx
│       │       ├── ErrorAnalysis.jsx
│       │       ├── Validation.jsx
│       │       ├── Uncertainty.jsx
│       │       ├── CriticalSlip.jsx
│       │       ├── RainfallSim.jsx
│       │       ├── ScenarioComparator.jsx
│       │       └── Export.jsx
│       └── package.json
│
├── ui/                              # Streamlit dashboard (legacy)
│   ├── app.py                       #   Streamlit entry point
│   ├── model_inference.py           #   Shared inference engine
│   └── pages/                       #   17 Streamlit page modules
│
└── research/                        # Jupyter notebooks for exploration
    ├── data.ipynb
    ├── loss_model.ipynb
    └── pinn_model.ipynb
```

---

## Installation

### Prerequisites

- Python 3.12+ 
- Node.js 18+ and npm (for the React dashboard)
- Git

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/suyash-04/PINN.git
cd PINN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install PyTorch

Install PyTorch for your platform from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# CPU only
pip install torch torchvision

# macOS Apple Silicon (MPS)
pip install torch torchvision

# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dashboard Dependencies (FastAPI + React)

```bash
# FastAPI backend
pip install fastapi uvicorn

# React frontend
cd ui-react/frontend
npm install
cd ../..
```

---

## Quick Start

### Option A: Train the Model + Launch Dashboard

```bash
# Step 1 — Run the training pipeline
python main.py

# Step 2 — Start the FastAPI backend (Terminal 1)
source .venv/bin/activate
cd ui-react/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Step 3 — Start the React frontend (Terminal 2)
cd ui-react/frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

### Option B: Dashboard Only (pre-trained model)

If the model is already trained (`artifacts/model/pinn_model.pt` exists):

```bash
# Terminal 1 — Backend
source .venv/bin/activate && cd ui-react/api
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd ui-react/frontend && npm run dev
```

---

## PINN Training Pipeline

The training pipeline is executed in three stages via `main.py`:

### Stage 1: Data Ingestion
Parses HYDRUS-1D output files into a structured CSV with depth, time, and pressure head columns.

```python
from src.pinn_landslide.pipeline.stage_1_data_ingestion_pipeline import DataIngestionPipeline
DataIngestionPipeline().run()
# → artifacts/dataset/final_data.csv
```

### Stage 2: Data Loading
Converts processed CSV into PyTorch tensors, normalizes features, and creates training batches.

```python
from src.pinn_landslide.pipeline.stage_2_data_loader_pipeline import DataLoaderPipeline
DataLoaderPipeline().run()
# → artifacts/dataset/data_batch.pt
```

### Stage 3: Model Training
Trains the PINN with composite physics-aware loss and two-phase optimization.

```python
from src.pinn_landslide.pipeline.stage_3_model_training import ModelTrainingPipeline
ModelTrainingPipeline().run()
# → artifacts/model/pinn_model.pt
```

Or run all three stages at once:

```bash
python main.py
```

### Training Details

| Parameter | Value |
|---|---|
| Architecture | 7 hidden layers × 64 neurons |
| Activation | tanh with Xavier initialization |
| Total parameters | ~29,000 |
| Phase 1 optimizer | Adam with learning rate scheduling |
| Phase 2 optimizer | L-BFGS (full batch, quasi-Newton) |
| Training data | 24,000 points (24 timesteps × 1,000 nodes) |
| Loss components | Data MSE + PDE residual + IC + BC + failure |
| Normalization | t ∈ [0, 123] → [0, 1], z ∈ [0, 55] → [0, 1], ψ → [0, 1] |

---

## Interactive Dashboard (React)

The production dashboard is built with **React 19**, **Vite 7**, **TailwindCSS v4**, and **Plotly.js**, backed by a **FastAPI** server.

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 7, TailwindCSS 4, Plotly.js, React Router 7, Lucide Icons |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Inference | PyTorch, NumPy, Pandas |
| Dev proxy | Vite `/api` → `http://127.0.0.1:8000` |

### Dashboard Pages (17)

#### Core (7 pages)
| Page | Route | Description |
|---|---|---|
| **Overview** | `/` | Metric cards, ψ & FS profiles at 6 timesteps, project description |
| **Pressure Head** | `/pressure-head` | ψ(z) profiles with timestep selector, ψ vs time at fixed depth |
| **Factor of Safety** | `/factor-of-safety` | 2D FS heatmap/contour, stability statistics, FS time history |
| **Parameter Explorer** | `/parameters` | Sensitivity tornado chart, current vs default comparison, point query |
| **HYDRUS Comparison** | `/hydrus` | Overlay, error profiles, scatter plot, per-timestep R²/RMSE table |
| **Soil Properties** | `/soil` | Van Genuchten curves: WRC θ(ψ), K(ψ), C(ψ), Se(ψ) |
| **Model Info** | `/model` | Architecture details, network topology diagram, governing equations |

#### Advanced (5 pages)
| Page | Route | Description |
|---|---|---|
| **Animation** | `/animation` | Time-lapse ψ/FS with play/pause, speed control, frame scrubber |
| **Training Loss** | `/training` | Loss curves (total, data, PDE, BC/IC), component breakdown |
| **PDE Residual** | `/pde-residual` | Richards' equation residual heatmap (log₁₀), histogram |
| **Error Analysis** | `/error` | Spatial/temporal error profiles, distribution, relative error heatmap |
| **Validation** | `/validation` | Overall R², RMSE, MAE, NSE, KGE; per-timestep metrics |

#### Research (5 pages)
| Page | Route | Description |
|---|---|---|
| **Uncertainty** | `/uncertainty` | Monte Carlo FS analysis with P(failure), reliability index |
| **Critical Slip** | `/critical-slip` | Minimum FS depth identification, evolution over time |
| **Rainfall Sim** | `/rainfall` | Conceptual rainfall infiltration scenario overlay |
| **Scenario Compare** | `/scenarios` | Multi-soil comparison (clay, silt, sand, residual) |
| **Export** | `/export` | Download prediction grids as CSV or JSON |

### Sidebar Features
- Collapsible navigation organized by section
- Live geotechnical parameter sliders (β, c′, φ′, γ, α, n)
- Parameter changes instantly re-compute all charts
- Mobile-responsive with hamburger menu

---

## Streamlit Dashboard (Legacy)

A 17-page Streamlit dashboard is also included and shares the same inference engine.

```bash
source .venv/bin/activate
pip install streamlit
cd ui
streamlit run app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## Configuration

Two YAML files control the training pipeline:

### `config/config.yaml` — Paths & Artifact Locations

```yaml
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

### `params.yaml` — Architecture, Training & Geotechnical Parameters

```yaml
architecture:
  n_hidden_layers: 7
  neurons_per_layer: 64
  activation: tanh
  input_dim: 2          # (t, z)
  output_dim: 1          # ψ

training:
  epochs: 5000
  learning_rate: 0.001
  device: cpu             # or cuda, mps

geo_params:
  Ks: 1.5e-5             # Saturated hydraulic conductivity (m/s)
  theta_s: 0.45           # Saturated water content
  theta_r: 0.05           # Residual water content
  alpha: 0.8              # Van Genuchten α (1/m)
  n: 1.4                  # Van Genuchten n
  l: 0.5                  # Pore connectivity parameter
  c_prime: 5.0            # Effective cohesion (kPa)
  phi_prime: 28.0         # Effective friction angle (°)
  gamma: 20.0             # Unit weight of soil (kN/m³)
  beta: 42.0              # Slope angle (°)
  gamma_w: 9.81           # Unit weight of water (kN/m³)

loss_weights:
  lambda_physics: 1.0
  lambda_anchor: 10.0
  lambda_initial: 5.0
  lambda_boundary: 5.0
  lambda_failure: 20.0
```

---

## Physics & Governing Equations

### Richards' Equation (1D vertical unsaturated flow)

$$C(\psi) \frac{\partial \psi}{\partial t} = \frac{\partial}{\partial z} \left[ K(\psi) \left( \frac{\partial \psi}{\partial z} + 1 \right) \right]$$

where $C(\psi) = d\theta/d\psi$ is the specific moisture capacity and $K(\psi)$ is the hydraulic conductivity.

### Van Genuchten Soil-Water Retention Model

$$S_e = \left[ 1 + (\alpha |\psi|)^n \right]^{-m}, \quad m = 1 - \frac{1}{n}$$

$$\theta(\psi) = \theta_r + (\theta_s - \theta_r) \cdot S_e$$

$$K(S_e) = K_s \cdot S_e^{0.5} \left[ 1 - \left(1 - S_e^{1/m}\right)^m \right]^2$$

### Infinite-Slope Factor of Safety with Matric Suction

$$FS = \frac{c'}{\gamma z \sin\beta \cos\beta} + \frac{\tan\phi'}{\tan\beta} - \frac{u_w \tan\phi'}{\gamma z \sin\beta \cos\beta}$$

where $u_w = \gamma_w \cdot \psi$ is the pore-water pressure (zero when $\psi < 0$ in the unsaturated zone).

---

## Model Architecture

```
Input (t, z)  →  [64] → [64] → [64] → [64] → [64] → [64] → [64]  →  Output ψ(t, z)
     2               7 hidden layers × 64 neurons (tanh)                    1
                           Xavier initialization
                         ~29,000 total parameters
```

The PINN uses PyTorch autograd to compute $\partial\psi/\partial t$ and $\partial\psi/\partial z$ required by the Richards' equation loss term.

---

## API Reference

The FastAPI backend exposes 16 endpoints under the `/api` prefix:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/defaults` | Default geo/norm params, available times, data stats |
| `GET` | `/api/model-info` | Architecture, parameter count, weight histograms |
| `GET` | `/api/validation` | Overall + per-timestep R², RMSE, MAE, NSE, KGE |
| `GET` | `/api/hydrus-data` | Raw HYDRUS-1D training data |
| `POST` | `/api/predict` | Predict ψ at (z, t) points |
| `POST` | `/api/predict-grid` | Predict ψ on a z × t grid |
| `POST` | `/api/factor-of-safety` | Compute FS at (z, t) points |
| `POST` | `/api/fs-grid` | Compute FS on a z × t grid |
| `POST` | `/api/soil-properties` | Van Genuchten curves (θ, K, C, Se) |
| `POST` | `/api/hydrus-comparison` | PINN vs HYDRUS at selected timesteps |
| `POST` | `/api/pde-residual` | Richards' equation residual grid |
| `POST` | `/api/uncertainty` | Monte Carlo FS analysis |
| `POST` | `/api/critical-slip` | Critical slip depth over time |
| `POST` | `/api/sensitivity` | Tornado chart sensitivity data |
| `POST` | `/api/export` | Export prediction grid as structured data |

All POST endpoints accept JSON with optional `geo` and `norm` parameter overrides. Full request/response schemas are defined in `ui-react/api/schemas.py`.

---

## Artifacts & Logging

| Artifact | Default Path | Description |
|---|---|---|
| Processed CSV | `artifacts/dataset/final_data.csv` | 24K rows of (Depth, Time, Pressure_Head) |
| Tensor batch | `artifacts/dataset/data_batch.pt` | Pre-batched normalized PyTorch tensors |
| Model checkpoint | `artifacts/model/pinn_model.pt` | Trained PINN state dict |
| Training logs | `logs/running_logs.log` | Structured logs for all pipeline stages |

All paths are configurable via `config/config.yaml`.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests where applicable
4. Open a pull request

Keep changes modular and documented.

## License

This repository includes a `LICENSE` file at the project root — please review it for the licensing terms.