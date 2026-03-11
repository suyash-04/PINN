"""
Pressure Head Profiles Page
============================
Interactive ψ(z) profiles at user-selected time steps.
Shows PINN predictions with depth as vertical axis.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import predict_psi


def render(model, df_hydrus, geo, norm):
    st.markdown("## 📈 Pressure Head Profiles — ψ(z, t)")
    st.caption("Visualise how pore-water pressure changes with depth at selected time steps")

    # ── Controls ─────────────────────────────────────────────────────
    ctrl_left, ctrl_mid, ctrl_right = st.columns([2, 1, 1])

    available_times = sorted(df_hydrus["Time_days"].unique())

    with ctrl_left:
        selected_times = st.multiselect(
            "Select time steps (days)",
            options=available_times,
            default=[0.0, 30.0, 60.0, 90.0, 96.0, 123.0],
            help="Choose time steps to plot ψ(z) profiles",
        )
    with ctrl_mid:
        z_min = st.number_input("Min depth (m)", 0.0, 54.0, 0.5, 0.5)
        z_max_ui = st.number_input("Max depth (m)", 1.0, 55.0, 50.0, 1.0)
    with ctrl_right:
        n_points = st.slider("Resolution (points)", 50, 500, 200, 50)
        show_hydrus = st.checkbox("Show HYDRUS data", value=True)

    if not selected_times:
        st.warning("Please select at least one time step.")
        return

    depths = np.linspace(z_min, z_max_ui, n_points)

    # ── Color palette ────────────────────────────────────────────────
    palette = [
        "#38bdf8", "#818cf8", "#a78bfa", "#f472b6", "#fb923c",
        "#ef4444", "#22c55e", "#eab308", "#06b6d4", "#8b5cf6",
        "#f43f5e", "#14b8a6", "#d946ef", "#64748b",
    ]

    # ── Main Plot ────────────────────────────────────────────────────
    fig = go.Figure()

    for i, t_val in enumerate(selected_times):
        color = palette[i % len(palette)]

        # PINN prediction
        t_arr = np.full_like(depths, t_val)
        psi_pred = predict_psi(model, depths, t_arr, norm)
        fig.add_trace(go.Scatter(
            x=psi_pred, y=depths,
            mode="lines", name=f"PINN Day {t_val:.0f}",
            line=dict(color=color, width=2.5),
            legendgroup=f"t{t_val}",
        ))

        # HYDRUS data overlay
        if show_hydrus:
            hyd = df_hydrus[df_hydrus["Time_days"] == t_val].sort_values("Depth_m")
            if len(hyd) > 0:
                fig.add_trace(go.Scatter(
                    x=hyd["Pressure_Head"].values, y=hyd["Depth_m"].values,
                    mode="markers", name=f"HYDRUS Day {t_val:.0f}",
                    marker=dict(color=color, size=4, symbol="circle-open", line=dict(width=1)),
                    legendgroup=f"t{t_val}",
                    showlegend=True,
                ))

    fig.update_layout(
        xaxis_title="Pressure Head ψ (m)",
        yaxis_title="Depth z (m)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=600,
        margin=dict(l=60, r=20, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
        hovermode="closest",
    )

    st.plotly_chart(fig, width="stretch")

    # ── ψ vs Time at fixed depth ─────────────────────────────────────
    st.divider()
    st.markdown("### ψ vs Time at Fixed Depth")

    depth_select = st.slider("Select depth (m)", 0.5, 50.0, 5.0, 0.5,
                             key="psi_time_depth")

    times_dense = np.linspace(0, norm["t_max"], 300)
    z_arr = np.full_like(times_dense, depth_select)
    psi_time = predict_psi(model, z_arr, times_dense, norm)

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=times_dense, y=psi_time,
        mode="lines", name=f"PINN ψ at z={depth_select}m",
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.1)",
    ))

    # HYDRUS points at this depth
    hyd_depth = df_hydrus[np.abs(df_hydrus["Depth_m"] - depth_select) < 0.1]
    if len(hyd_depth) > 0:
        fig_time.add_trace(go.Scatter(
            x=hyd_depth["Time_days"].values, y=hyd_depth["Pressure_Head"].values,
            mode="markers", name="HYDRUS",
            marker=dict(color="#f472b6", size=8, symbol="diamond"),
        ))

    fig_time.update_layout(
        xaxis_title="Time (days)",
        yaxis_title="Pressure Head ψ (m)",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=20, t=30, b=50),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
    )
    st.plotly_chart(fig_time, width="stretch")

    # ── Summary Table ────────────────────────────────────────────────
    with st.expander("📋 Numerical Values at Selected Times"):
        import pandas as pd
        rows = []
        for t_val in selected_times:
            for z_probe in [1.0, 5.0, 10.0, 20.0, 40.0]:
                if z_probe <= z_max_ui:
                    psi_val = predict_psi(model, np.array([z_probe]), np.array([t_val]), norm)[0]
                    rows.append({"Time (days)": t_val, "Depth (m)": z_probe,
                                 "ψ PINN (m)": f"{psi_val:.2f}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
