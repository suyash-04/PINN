"""
Overview / Home Page
====================
Dashboard summary with key metrics, quick-look plots, and project description.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import predict_psi, predict_grid, compute_FS, vg_Se, vg_theta


def render(model, df_hydrus, geo, norm):
    # ── Header Banner ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="header-banner">
            <h1>🏔️ PINN Landslide Stability Dashboard</h1>
            <p>Physics-Informed Neural Network for predicting slope stability under rainfall infiltration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key Metrics Row ──────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    n_points = len(df_hydrus)
    time_range = f"{df_hydrus['Time_days'].min():.0f} – {df_hydrus['Time_days'].max():.0f}"
    depth_range = f"{df_hydrus['Depth_m'].min():.1f} – {df_hydrus['Depth_m'].max():.1f}"
    psi_range_str = f"{df_hydrus['Pressure_Head'].min():.0f} to {df_hydrus['Pressure_Head'].max():.1f}"

    # Quick PINN prediction at a sample point
    sample_z = np.array([5.0])
    sample_t = np.array([96.0])
    sample_psi = predict_psi(model, sample_z, sample_t, norm)
    sample_fs = compute_FS(sample_psi, sample_z, geo)

    col1.metric("Data Points", f"{n_points:,}")
    col2.metric("Time Span (days)", time_range)
    col3.metric("Depth Range (m)", depth_range)
    col4.metric("ψ at 5m, Day 96", f"{sample_psi[0]:.1f} m")

    # Color the FS metric
    fs_val = sample_fs[0]
    if fs_val > 1.5:
        col5.metric("FS at 5m, Day 96", f"{fs_val:.2f}", delta="Safe", delta_color="normal")
    elif fs_val > 1.0:
        col5.metric("FS at 5m, Day 96", f"{fs_val:.2f}", delta="Marginal", delta_color="off")
    else:
        col5.metric("FS at 5m, Day 96", f"{fs_val:.2f}", delta="Unstable", delta_color="inverse")

    st.divider()

    # ── Quick-look Plots ─────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 📊 Pressure Head Profile (PINN Prediction)")
        depths = np.linspace(0.5, 50.0, 200)
        times_to_show = [0, 30, 60, 90, 96, 123]

        fig_psi = go.Figure()
        colors = ["#38bdf8", "#818cf8", "#a78bfa", "#f472b6", "#fb923c", "#ef4444"]
        for i, t_val in enumerate(times_to_show):
            t_arr = np.full_like(depths, t_val)
            psi_pred = predict_psi(model, depths, t_arr, norm)
            fig_psi.add_trace(go.Scatter(
                x=psi_pred, y=depths, mode="lines", name=f"Day {t_val}",
                line=dict(color=colors[i % len(colors)], width=2),
            ))

        fig_psi.update_layout(
            xaxis_title="Pressure Head ψ (m)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=450,
            margin=dict(l=60, r=20, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_psi, width="stretch")

    with right:
        st.markdown("#### 🛡️ Factor of Safety Profile")

        fig_fs = go.Figure()
        for i, t_val in enumerate(times_to_show):
            t_arr = np.full_like(depths, t_val)
            psi_pred = predict_psi(model, depths, t_arr, norm)
            fs_pred = compute_FS(psi_pred, depths, geo)
            fig_fs.add_trace(go.Scatter(
                x=fs_pred, y=depths, mode="lines", name=f"Day {t_val}",
                line=dict(color=colors[i % len(colors)], width=2),
            ))

        # FS = 1 reference line
        fig_fs.add_vline(x=1.0, line_dash="dash", line_color="#ef4444",
                         annotation_text="FS = 1 (Failure)", annotation_position="top right")

        fig_fs.update_layout(
            xaxis_title="Factor of Safety",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(range=[0, min(10, float(np.max(fs_pred)) + 1)]),
            template="plotly_white",
            height=450,
            margin=dict(l=60, r=20, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_fs, width="stretch")

    st.divider()

    # ── Project Description ──────────────────────────────────────────
    st.markdown("### 📖 About This Project")

    desc_left, desc_right = st.columns([2, 1])

    with desc_left:
        st.markdown("""
        This dashboard visualises a **Physics-Informed Neural Network (PINN)** trained
        to predict **subsurface pore-water pressure** and **slope stability** (Factor of Safety)
        during a rainfall event on a natural hillslope.

        **How it works:**
        1. **HYDRUS-1D** simulates 1-D vertical water flow governed by **Richards' Equation**.
        2. The PINN learns the pressure-head field ψ(z, t) while being constrained to obey the PDE.
        3. Geotechnical parameters are used to compute the **Factor of Safety** via the infinite-slope model.

        **Key physics encoded:**
        - **Richards' Equation** (unsaturated water flow PDE)
        - **Van Genuchten** soil–water retention model
        - **Mohr–Coulomb** shear-strength criterion
        - **Infinite-slope** stability model with matric suction
        """)

    with desc_right:
        st.markdown("#### Current Parameters")
        param_data = {
            "Parameter": ["β (slope)", "c′ (cohesion)", "φ′ (friction)", "γ (unit wt)",
                          "Ks", "θs", "θr", "α", "n"],
            "Value": [f"{geo['beta']:.1f}°", f"{geo['c_prime']:.1f} kPa",
                      f"{geo['phi_prime']:.1f}°", f"{geo['gamma']:.1f} kN/m³",
                      f"{geo['Ks']:.2e} m/s", f"{geo['theta_s']:.2f}",
                      f"{geo['theta_r']:.2f}", f"{geo['alpha']:.1f} 1/m",
                      f"{geo['n']:.2f}"],
        }
        st.dataframe(pd.DataFrame(param_data), hide_index=True, width="stretch")
