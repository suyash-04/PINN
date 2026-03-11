"""
Rainfall Event Simulator Page
==============================
Let user design a rainfall hyetograph and see how FS responds.
Translates rainfall intensity → surface ψ → PINN propagation → FS.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import predict_psi, compute_FS


def render(model, df_hydrus, geo, norm):
    st.markdown("## 📉 Rainfall Event Simulator")
    st.caption("Design a rainfall hyetograph and observe the slope's dynamic response")

    st.info(
        "💡 The PINN was trained on a fixed HYDRUS-1D boundary condition. This simulator "
        "**approximates** the effect of different rainfall patterns by scaling the time "
        "axis to match peak infiltration periods. For rigorous new-rainfall analysis, "
        "re-run HYDRUS-1D with the new boundary and retrain the PINN."
    )

    # ── Rainfall Design ──────────────────────────────────────────────
    st.markdown("### 🌧️ Design Your Rainfall Event")

    c1, c2, c3 = st.columns(3)
    with c1:
        pattern = st.selectbox("Rainfall pattern", [
            "Uniform", "Early peak", "Late peak", "Double peak", "Custom"
        ], key="rain_pattern")
    with c2:
        duration = st.slider("Duration (days)", 5, 120, 60, 5, key="rain_dur")
    with c3:
        total_rain = st.slider("Total rainfall (mm)", 50, 1000, 300, 10, key="rain_total")

    # Generate hyetograph based on pattern
    n_steps = 24
    t_rain = np.linspace(0, duration, n_steps)
    dt = duration / n_steps

    if pattern == "Uniform":
        intensity = np.ones(n_steps) * total_rain / duration
    elif pattern == "Early peak":
        x = np.linspace(0, 3, n_steps)
        raw = np.exp(-x)
        intensity = raw / raw.sum() * total_rain / dt
    elif pattern == "Late peak":
        x = np.linspace(0, 3, n_steps)
        raw = np.exp(-(3 - x))
        intensity = raw / raw.sum() * total_rain / dt
    elif pattern == "Double peak":
        x = np.linspace(0, 2 * np.pi, n_steps)
        raw = np.abs(np.sin(x)) + 0.3
        intensity = raw / raw.sum() * total_rain / dt
    else:  # Custom
        st.markdown("Enter comma-separated intensity values (mm/day):")
        custom_str = st.text_input("Intensities", "5,10,20,40,40,30,15,5", key="rain_custom")
        try:
            vals = [float(v.strip()) for v in custom_str.split(",")]
            t_rain = np.linspace(0, duration, len(vals))
            intensity = np.array(vals)
        except Exception:
            st.error("Invalid input. Use comma-separated numbers.")
            return

    cumulative = np.cumsum(intensity * dt)

    # ── Hyetograph Plot ──────────────────────────────────────────────
    fig_rain = make_subplots(specs=[[{"secondary_y": True}]])

    fig_rain.add_trace(go.Bar(
        x=t_rain, y=intensity, name="Intensity (mm/day)",
        marker_color="#38bdf8", opacity=0.8,
    ), secondary_y=False)

    fig_rain.add_trace(go.Scatter(
        x=t_rain, y=cumulative, name="Cumulative (mm)",
        line=dict(color="#f472b6", width=2.5),
    ), secondary_y=True)

    fig_rain.update_layout(
        xaxis_title="Time (days)",
        template="plotly_white", height=300,
        margin=dict(l=60, r=60, t=30, b=60),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
    )
    fig_rain.update_yaxes(title_text="Intensity (mm/day)", secondary_y=False)
    fig_rain.update_yaxes(title_text="Cumulative (mm)", secondary_y=True)
    st.plotly_chart(fig_rain, width="stretch")

    st.divider()

    # ── Map rainfall to PINN time domain ─────────────────────────────
    st.markdown("### 🔮 Slope Response Prediction")

    # Strategy: Map the rainfall peak to the HYDRUS training peak (around day 90-96)
    # Scale the time axis: t_pinn = t_rain * (norm["t_max"] / duration)
    scale_factor = norm["t_max"] / max(duration, 1)

    # Monitor depths
    monitor_depths = st.multiselect(
        "Monitor depths (m)", [1, 2, 5, 10, 15, 20, 25, 30, 40, 50],
        default=[2, 5, 10, 20, 30], key="rain_depths"
    )

    if not monitor_depths:
        st.warning("Select at least one depth.")
        return

    # Compute ψ and FS at monitor depths over the rainfall duration
    t_eval = np.linspace(0, duration, 50)
    t_pinn = t_eval * scale_factor  # scaled to training domain

    palette = ["#38bdf8", "#818cf8", "#f472b6", "#fb923c", "#22c55e",
               "#ef4444", "#eab308", "#a78bfa", "#06b6d4", "#d946ef"]

    fig_resp = make_subplots(rows=1, cols=2,
                             subplot_titles=["Pressure Head ψ(t)", "Factor of Safety FS(t)"],
                             horizontal_spacing=0.08)

    for i, z_val in enumerate(monitor_depths):
        color = palette[i % len(palette)]
        z_arr = np.full(len(t_pinn), float(z_val))
        psi = predict_psi(model, z_arr, t_pinn, norm)
        fs = compute_FS(psi, z_arr, geo)

        fig_resp.add_trace(go.Scatter(
            x=t_eval, y=psi, mode="lines", name=f"z={z_val}m",
            line=dict(color=color, width=2), legendgroup=f"z{z_val}",
        ), row=1, col=1)

        fig_resp.add_trace(go.Scatter(
            x=t_eval, y=fs, mode="lines", name=f"z={z_val}m",
            line=dict(color=color, width=2), legendgroup=f"z{z_val}",
            showlegend=False,
        ), row=1, col=2)

    fig_resp.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", row=1, col=2,
                       annotation_text="FS = 1")

    fig_resp.update_xaxes(title_text="Time (days)")
    fig_resp.update_yaxes(title_text="ψ (m)", row=1, col=1)
    fig_resp.update_yaxes(title_text="FS", row=1, col=2)

    fig_resp.update_layout(
        template="plotly_white", height=450,
        margin=dict(l=60, r=20, t=40, b=60),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    st.plotly_chart(fig_resp, width="stretch")

    # ── Risk Summary ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚠️ Risk Summary")

    all_min_fs = []
    for z_val in monitor_depths:
        z_arr = np.full(len(t_pinn), float(z_val))
        psi = predict_psi(model, z_arr, t_pinn, norm)
        fs = compute_FS(psi, z_arr, geo)
        all_min_fs.append({"Depth (m)": z_val, "Min FS": fs.min(),
                           "Time of Min FS (day)": t_eval[np.argmin(fs)],
                           "Status": "🔴 Failure" if fs.min() < 1.0 else
                                     "🟡 Marginal" if fs.min() < 1.5 else "🟢 Safe"})

    import pandas as pd
    st.dataframe(pd.DataFrame(all_min_fs), hide_index=True, width="stretch")
