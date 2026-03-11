"""
Factor of Safety Contour Map Page
==================================
2-D heatmap of FS(z, t) with interactive controls.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import predict_grid, compute_FS


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🗺️ Factor of Safety Contour Map")
    st.caption("2-D heatmap of slope stability across depth and time")

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        z_res = st.slider("Depth resolution", 20, 100, 50, 10, key="fs_zres")
    with c2:
        t_res = st.slider("Time resolution", 20, 100, 50, 10, key="fs_tres")
    with c3:
        z_max_plot = st.slider("Max depth (m)", 5.0, 55.0, 40.0, 5.0, key="fs_zmax")
    with c4:
        fs_clip = st.slider("FS display max", 1.5, 10.0, 5.0, 0.5, key="fs_clip")

    # ── Compute FS grid ──────────────────────────────────────────────
    z_arr = np.linspace(0.5, z_max_plot, z_res)
    t_arr = np.linspace(0.0, norm["t_max"], t_res)

    with st.spinner("Computing FS grid from PINN …"):
        psi_grid = predict_grid(model, z_arr, t_arr, norm)   # shape (z, t)
        Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
        fs_grid = compute_FS(psi_grid, Z, geo)
        fs_display = np.clip(fs_grid, 0.0, fs_clip)

    # ── Contour Heatmap ──────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=t_arr, y=z_arr, z=fs_display,
        colorscale=[
            [0.0, "#7f1d1d"],    # dark red  (FS ≈ 0)
            [0.15, "#ef4444"],   # red       (FS < 1)
            [0.20, "#f97316"],   # orange    (FS ≈ 1)
            [0.30, "#eab308"],   # yellow    (FS ≈ 1.5)
            [0.50, "#22c55e"],   # green     (FS ≈ 2.5)
            [1.0, "#0ea5e9"],    # blue      (FS high)
        ],
        colorbar=dict(title=dict(text="FS", side="right")),
        hovertemplate="Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>FS: %{z:.3f}<extra></extra>",
    ))

    # FS = 1 contour line
    fig.add_trace(go.Contour(
        x=t_arr, y=z_arr, z=fs_grid,
        contours=dict(start=1.0, end=1.0, size=0.1, coloring="none"),
        line=dict(color="#1e293b", width=3, dash="dash"),
        showscale=False, showlegend=True, name="FS = 1.0",
        hoverinfo="skip",
    ))

    fig.update_layout(
        xaxis_title="Time (days)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=550,
        margin=dict(l=60, r=20, t=30, b=60),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
    )

    st.plotly_chart(fig, width="stretch")

    # ── FS at specific depth over time ───────────────────────────────
    st.divider()
    st.markdown("### 📉 FS Time History at Specific Depths")

    depths_to_plot = st.multiselect(
        "Select depths (m)", [1, 2, 5, 10, 15, 20, 30, 40],
        default=[2, 5, 10, 20], key="fs_depths",
    )

    if depths_to_plot:
        t_dense = np.linspace(0, norm["t_max"], 200)
        palette = ["#38bdf8", "#818cf8", "#f472b6", "#fb923c", "#22c55e", "#ef4444", "#eab308", "#a78bfa"]

        fig_ts = go.Figure()
        for i, d in enumerate(depths_to_plot):
            z_const = np.full_like(t_dense, float(d))
            psi_ts = predict_grid(model, np.array([float(d)]), t_dense, norm).flatten()
            fs_ts = compute_FS(psi_ts, z_const[:len(psi_ts)], geo)
            fig_ts.add_trace(go.Scatter(
                x=t_dense[:len(fs_ts)], y=fs_ts,
                mode="lines", name=f"z = {d} m",
                line=dict(color=palette[i % len(palette)], width=2.5),
            ))

        fig_ts.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                         annotation_text="FS = 1 (Failure threshold)")
        fig_ts.add_hline(y=1.5, line_dash="dot", line_color="#eab308",
                         annotation_text="FS = 1.5 (Marginal)")

        fig_ts.update_layout(
            xaxis_title="Time (days)",
            yaxis_title="Factor of Safety",
            yaxis=dict(range=[0, min(fs_clip, 8)]),
            template="plotly_white",
            height=450,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ts, width="stretch")

    # ── Stability summary ────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Stability Summary")

    total_cells = fs_grid.size
    unstable = np.sum(fs_grid < 1.0)
    marginal = np.sum((fs_grid >= 1.0) & (fs_grid < 1.5))
    safe = np.sum(fs_grid >= 1.5)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Grid Cells", f"{total_cells:,}")
    s2.metric("🔴 Unstable (FS < 1)", f"{unstable:,}  ({100*unstable/total_cells:.1f}%)")
    s3.metric("🟡 Marginal (1 ≤ FS < 1.5)", f"{marginal:,}  ({100*marginal/total_cells:.1f}%)")
    s4.metric("🟢 Safe (FS ≥ 1.5)", f"{safe:,}  ({100*safe/total_cells:.1f}%)")
