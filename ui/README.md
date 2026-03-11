# 🏔️ PINN Landslide Stability Dashboard

Interactive Streamlit dashboard for visualising and analysing the **Physics-Informed Neural Network (PINN)** trained on HYDRUS-1D seepage data for rainfall-induced landslide stability prediction.

---

## Quick Start

```bash
# From the project root
cd ui
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

> **Prerequisites:** The trained model (`artifacts/model/pinn_model.pt`) and processed dataset (`artifacts/dataset/final_data.csv`) must exist. Run `python main.py` from the project root first if they don't.

---

## Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| 1 | **🏠 Overview** | Key metrics (data points, time/depth range, ψ range), quick-look ψ and FS profiles, dataset summary, and parameter table |
| 2 | **📈 Pressure Head Profiles** | Interactive ψ(z) profiles at user-selected time steps, ψ(t) time-series at a fixed depth, optional HYDRUS data overlay |
| 3 | **🗺️ Factor of Safety Map** | 2-D depth × time FS heatmap with FS = 1 contour line, FS time-history at multiple depths, stability summary statistics |
| 4 | **🎛️ Parameter Explorer** | Real-time sensitivity of ψ and FS to geotechnical parameters, tornado chart ranking parameter influence, single-point query with full soil-property readout |
| 5 | **🔬 HYDRUS vs PINN** | Side-by-side overlay of HYDRUS simulation and PINN predictions, error profiles, scatter plot with R², RMSE, MAE per time step |
| 6 | **🧪 Soil Property Curves** | Van Genuchten retention θ(ψ), conductivity K(ψ), capacity C(ψ), and effective saturation Se(ψ) with interactive parameter sliders and equations |
| 7 | **🧠 Model Architecture** | Neural network diagram, per-layer parameter breakdown, weight-distribution histograms, normalisation info, training config, and loss equation |

---

## Sidebar Controls

The sidebar provides **global geotechnical parameters** shared across all pages:

- **Slope Properties** — β (slope angle), c′ (cohesion), φ′ (friction angle), γ (unit weight)
- **Hydraulic Properties** — Kₛ (saturated conductivity), θₛ, θᵣ, α, n (Van Genuchten)

Changing any parameter immediately updates all plots on the current page.

---

## File Structure

```
ui/
├── .streamlit/
│   └── config.toml            # Streamlit theme (dark mode, accent colour)
├── assets/
│   └── style.css              # Custom CSS for cards, metrics, layout
├── pages/
│   ├── __init__.py            # Re-exports all page modules
│   ├── page_overview.py       # Dashboard home
│   ├── page_psi_profiles.py   # Pressure head exploration
│   ├── page_fs_contour.py     # Factor of Safety heatmap
│   ├── page_parameter_explorer.py  # Sensitivity analysis
│   ├── page_hydrus_comparison.py   # HYDRUS vs PINN comparison
│   ├── page_soil_properties.py     # Van Genuchten curves
│   └── page_model_info.py         # Model architecture & config
├── app.py                     # Entry point — navigation, caching, routing
├── model_inference.py         # Inference engine (model loader, VG funcs, FS)
├── __init__.py
└── README.md                  # ← You are here
```

### Key Modules

| Module | Role |
|--------|------|
| `app.py` | Sets page config, loads custom CSS, caches model & data, renders sidebar, routes to selected page |
| `model_inference.py` | Loads `PINN` weights, exposes `predict_psi()` / `predict_grid()`, numpy implementations of Van Genuchten functions (`vg_Se`, `vg_theta`, `vg_K`, `vg_C`), and two FS formulae (`compute_FS`, `compute_FS_original`) |

---

## Dependencies

All dependencies are already in the project's `requirements.txt`. The UI additionally uses:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `plotly` | Interactive charts (heatmaps, scatter, line, histograms) |
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical computation |
| `torch` | Model inference |

Install if missing:

```bash
pip install streamlit plotly
```

---

## Configuration

### Streamlit Theme (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#38bdf8"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
font = "sans serif"
```

### Model & Data Paths

Paths are resolved automatically relative to the project root:

- **Model:** `artifacts/model/pinn_model.pt`
- **Dataset:** `artifacts/dataset/final_data.csv`
- **Config:** `params.yaml`, `config/config.yaml`

These can be overridden by editing `MODEL_PATH` and `DATA_PATH` in `model_inference.py`.

---

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│  app.py     │────▶│ model_inference   │────▶│ PINN model     │
│  (Streamlit)│     │ (predict, VG, FS) │     │ (PyTorch .pt)  │
└──────┬──────┘     └──────────────────┘     └────────────────┘
       │
       ▼
┌──────────────┐
│  pages/*     │  Each page calls predict_psi() / compute_FS()
│  (7 modules) │  with user-controlled parameters from the sidebar
└──────────────┘
```

1. `app.py` loads the trained PINN model and HYDRUS CSV once (cached).
2. Sidebar parameters are collected into a `geo` dictionary.
3. The selected page's `render(model, df_hydrus, geo, norm)` function is called.
4. Each page uses `predict_psi()` / `compute_FS()` from `model_inference.py` to generate predictions and renders interactive Plotly charts.

---

## License

Same as the parent project — see `LICENSE` in the repository root.
