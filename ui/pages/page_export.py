"""
Data Export & Report Generator Page
====================================
Download plots, tables, CSV exports, and a full auto-generated report.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import json
from datetime import datetime

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import (
    predict_psi, predict_grid, compute_FS, compute_hydro_metrics,
    find_critical_depth, DEFAULT_NORM, DEFAULT_GEO,
)


def _generate_prediction_csv(model, norm, geo, z_res=100, t_res=50) -> pd.DataFrame:
    """Generate a full grid of PINN predictions as a flat CSV."""
    z_arr = np.linspace(0.5, norm["z_max"] * 0.9, z_res)
    t_arr = np.linspace(0.0, norm["t_max"], t_res)
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")

    psi_grid = predict_grid(model, z_arr, t_arr, norm)
    fs_grid = compute_FS(psi_grid, Z, geo)

    return pd.DataFrame({
        "Depth_m": Z.flatten(),
        "Time_days": T.flatten(),
        "Psi_PINN_m": psi_grid.flatten(),
        "FS": fs_grid.flatten(),
    })


def _generate_comparison_csv(model, df_hydrus, norm, geo) -> pd.DataFrame:
    """HYDRUS vs PINN comparison at every data point."""
    z_hyd = df_hydrus["Depth_m"].values
    t_hyd = df_hydrus["Time_days"].values
    psi_hyd = df_hydrus["Pressure_Head"].values

    psi_pinn = predict_psi(model, z_hyd, t_hyd, norm)
    fs_pinn = compute_FS(psi_pinn, z_hyd, geo)

    return pd.DataFrame({
        "Depth_m": z_hyd,
        "Time_days": t_hyd,
        "Psi_HYDRUS_m": psi_hyd,
        "Psi_PINN_m": psi_pinn,
        "Error_m": psi_pinn - psi_hyd,
        "Abs_Error_m": np.abs(psi_pinn - psi_hyd),
        "FS_PINN": fs_pinn,
    })


def _generate_text_report(model, df_hydrus, norm, geo) -> str:
    """Generate a markdown-formatted text report."""
    # Compute key metrics
    z_hyd = df_hydrus["Depth_m"].values
    t_hyd = df_hydrus["Time_days"].values
    psi_hyd = df_hydrus["Pressure_Head"].values
    psi_pinn = predict_psi(model, z_hyd, t_hyd, norm)
    metrics = compute_hydro_metrics(psi_hyd, psi_pinn)

    # Critical depth analysis
    t_crit = np.linspace(0, norm["t_max"], 50)
    crit = find_critical_depth(model, t_crit, geo, norm)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# PINN Landslide Stability Analysis — Report
Generated: {now}

## 1. Project Summary
- **Objective:** Predict slope stability under rainfall infiltration using
  a Physics-Informed Neural Network trained on HYDRUS-1D simulation data.
- **Method:** Richards' equation (1D vertical) + Van Genuchten + Infinite slope model
- **Training data:** {len(df_hydrus):,} points from HYDRUS-1D ({len(df_hydrus['Time_days'].unique())} time steps)

## 2. Model Architecture
- **Type:** Fully-connected feedforward MLP
- **Hidden layers:** 7 × 64 neurons
- **Activation:** tanh (Xavier initialisation)
- **Total parameters:** {total_params:,}
- **Training:** Adam (10k epochs, LR=0.001) → L-BFGS (5k max iter, strong Wolfe)

## 3. Data Summary
| Quantity | Range |
|----------|-------|
| Time | {df_hydrus['Time_days'].min():.0f} – {df_hydrus['Time_days'].max():.0f} days |
| Depth | {df_hydrus['Depth_m'].min():.1f} – {df_hydrus['Depth_m'].max():.1f} m |
| Pressure Head (HYDRUS) | {psi_hyd.min():.1f} to {psi_hyd.max():.1f} m |
| Pressure Head (PINN) | {psi_pinn.min():.1f} to {psi_pinn.max():.1f} m |

## 4. Validation Metrics (PINN vs HYDRUS)
| Metric | Value |
|--------|-------|
| R² | {metrics['R2']:.6f} |
| RMSE | {metrics['RMSE']:.4f} m |
| MAE | {metrics['MAE']:.4f} m |
| Nash-Sutcliffe Efficiency | {metrics['NSE']:.6f} |
| Kling-Gupta Efficiency | {metrics['KGE']:.6f} |
| Percent Bias | {metrics['PBIAS']:.2f}% |
| Normalised RMSE | {metrics['nRMSE']:.2f}% |

## 5. Geotechnical Parameters
| Parameter | Value |
|-----------|-------|
| Slope angle β | {geo['beta']}° |
| Cohesion c' | {geo['c_prime']} kPa |
| Friction angle φ' | {geo['phi_prime']}° |
| Unit weight γ | {geo['gamma']} kN/m³ |
| Ks | {geo['Ks']:.2e} m/s |
| θs / θr | {geo['theta_s']} / {geo['theta_r']} |
| α (VG) | {geo['alpha']} 1/m |
| n (VG) | {geo['n']} |

## 6. Stability Analysis
- **Most critical time:** Day {crit['times'][np.argmin(crit['min_fs'])]:.0f}
- **Minimum FS observed:** {np.min(crit['min_fs']):.3f}
- **Critical depth at min FS:** {crit['critical_depths'][np.argmin(crit['min_fs'])]:.1f} m
- **Mean min-FS across all times:** {np.mean(crit['min_fs']):.3f}
"""
    return report


def render(model, df_hydrus, geo, norm):
    st.markdown("## 📥 Data Export & Report Generator")
    st.caption("Download predictions, comparison data, and auto-generated analysis reports")

    tab_export, tab_report, tab_config = st.tabs([
        "📊 Export Data", "📝 Auto Report", "⚙️ Export Config"
    ])

    # ── Tab 1: Data Export ───────────────────────────────────────────
    with tab_export:
        st.markdown("### Download Datasets")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🔮 PINN Predictions (full grid)")
            st.markdown("ψ and FS predictions on a regular depth × time grid")
            z_res = st.slider("Depth points", 20, 200, 60, key="exp_zres")
            t_res = st.slider("Time points", 10, 100, 30, key="exp_tres")

            if st.button("Generate PINN Grid", key="gen_grid"):
                with st.spinner("Computing …"):
                    df_pred = _generate_prediction_csv(model, norm, geo, z_res, t_res)
                    st.session_state["df_pred"] = df_pred

            if "df_pred" in st.session_state:
                df_pred = st.session_state["df_pred"]
                st.dataframe(df_pred.head(20), hide_index=True, width="stretch")
                st.download_button(
                    "⬇️ Download PINN Predictions (.csv)",
                    df_pred.to_csv(index=False),
                    "pinn_predictions.csv", "text/csv",
                )

        with c2:
            st.markdown("#### 🔬 HYDRUS vs PINN Comparison")
            st.markdown("Point-by-point error at every HYDRUS data location")

            if st.button("Generate Comparison", key="gen_comp"):
                with st.spinner("Computing …"):
                    df_comp = _generate_comparison_csv(model, df_hydrus, norm, geo)
                    st.session_state["df_comp"] = df_comp

            if "df_comp" in st.session_state:
                df_comp = st.session_state["df_comp"]
                st.dataframe(df_comp.head(20), hide_index=True, width="stretch")
                st.download_button(
                    "⬇️ Download Comparison (.csv)",
                    df_comp.to_csv(index=False),
                    "hydrus_vs_pinn_comparison.csv", "text/csv",
                )

        st.divider()
        st.markdown("#### 📊 Original HYDRUS Dataset")
        st.download_button(
            "⬇️ Download HYDRUS Data (.csv)",
            df_hydrus.to_csv(index=False),
            "hydrus_original_data.csv", "text/csv",
        )

    # ── Tab 2: Auto Report ───────────────────────────────────────────
    with tab_report:
        st.markdown("### 📝 Auto-Generated Analysis Report")
        st.markdown("Click below to generate a full markdown report with all metrics:")

        if st.button("🚀 Generate Report", key="gen_report"):
            with st.spinner("Generating comprehensive report …"):
                report_text = _generate_text_report(model, df_hydrus, norm, geo)
                st.session_state["report_text"] = report_text

        if "report_text" in st.session_state:
            st.markdown(st.session_state["report_text"])

            st.divider()
            st.download_button(
                "⬇️ Download Report (.md)",
                st.session_state["report_text"],
                "pinn_analysis_report.md", "text/markdown",
            )

    # ── Tab 3: Configuration Export ──────────────────────────────────
    with tab_config:
        st.markdown("### ⚙️ Export Configuration")
        st.markdown("Download the current parameter configuration as JSON:")

        config_dict = {
            "normalization": norm,
            "geotechnical_parameters": geo,
            "architecture": {
                "n_hidden_layers": 7,
                "neurons_per_layer": 64,
                "activation": "tanh",
                "input_dim": 2,
                "output_dim": 1,
            },
            "training": {
                "adam_epochs": 10000,
                "learning_rate": 0.001,
                "lbfgs_max_iter": 5000,
            },
            "exported_at": datetime.now().isoformat(),
        }

        st.json(config_dict)
        st.download_button(
            "⬇️ Download Config (.json)",
            json.dumps(config_dict, indent=2, default=str),
            "pinn_config.json", "application/json",
        )
