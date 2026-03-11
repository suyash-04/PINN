"""
Critical Slip Surface Finder Page
===================================
Auto-detect depth where FS is minimum for each time step.
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

from model_inference import predict_psi, compute_FS, find_critical_depth, predict_grid


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🗺️ Critical Slip Surface Finder")
    st.caption("Identify the most vulnerable depth at each time step — the key geotechnical output")

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        t_res = st.slider("Time resolution", 20, 100, 50, 5, key="crit_tres")
    with c2:
        z_res = st.slider("Depth resolution", 50, 300, 150, 25, key="crit_zres")

    t_arr = np.linspace(0.0, norm["t_max"], t_res)

    # ── Compute ──────────────────────────────────────────────────────
    with st.spinner("Finding critical slip surfaces …"):
        crit = find_critical_depth(model, t_arr, geo, norm, z_res)

    tab_evolution, tab_map, tab_table = st.tabs([
        "📈 Critical Depth Evolution", "🗺️ FS Map with Slip Line", "📋 Data Table"
    ])

    # ── Tab 1: Critical depth and min FS over time ───────────────────
    with tab_evolution:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Critical Depth vs Time",
                                            "Minimum Factor of Safety vs Time"],
                            vertical_spacing=0.08)

        # Critical depth
        fig.add_trace(go.Scatter(
            x=crit["times"], y=crit["critical_depths"],
            mode="lines+markers", line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5), name="Critical Depth",
        ), row=1, col=1)

        # Min FS
        fig.add_trace(go.Scatter(
            x=crit["times"], y=crit["min_fs"],
            mode="lines+markers", line=dict(color="#22c55e", width=2.5),
            marker=dict(size=5), name="Min FS",
        ), row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                      annotation_text="FS = 1 (Failure)", row=2, col=1)
        fig.add_hline(y=1.5, line_dash="dot", line_color="#eab308",
                      annotation_text="FS = 1.5 (Marginal)", row=2, col=1)

        fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
        fig.update_yaxes(title_text="Min FS", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)

        fig.update_layout(
            template="plotly_white", height=600,
            margin=dict(l=60, r=20, t=40, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        # Key findings
        st.divider()
        idx_worst = np.argmin(crit["min_fs"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Most Critical Time", f"Day {crit['times'][idx_worst]:.1f}")
        c2.metric("Minimum FS", f"{crit['min_fs'][idx_worst]:.4f}")
        c3.metric("Critical Depth", f"{crit['critical_depths'][idx_worst]:.1f} m")
        c4.metric("FS Range", f"{crit['min_fs'].min():.3f} – {crit['min_fs'].max():.3f}")

        # Stability classification
        worst_fs = crit["min_fs"].min()
        if worst_fs > 1.5:
            st.success("✅ Slope is **stable** throughout the analysis period (all FS > 1.5)")
        elif worst_fs > 1.0:
            st.warning(f"⚠️ Slope is **marginally stable** — minimum FS = {worst_fs:.3f} (1.0 < FS < 1.5)")
        else:
            st.error(f"🚨 Slope **failure predicted** — minimum FS = {worst_fs:.3f} at Day {crit['times'][idx_worst]:.0f}, Depth {crit['critical_depths'][idx_worst]:.1f} m")

    # ── Tab 2: FS Map with critical line overlay ─────────────────────
    with tab_map:
        z_map = np.linspace(0.5, norm["z_max"] * 0.9, 60)
        t_map = np.linspace(0.0, norm["t_max"], 60)

        with st.spinner("Computing FS map …"):
            psi_grid = predict_grid(model, z_map, t_map, norm)
            Z, T = np.meshgrid(z_map, t_map, indexing="ij")
            fs_grid = compute_FS(psi_grid, Z, geo)

        fig2 = go.Figure()

        # FS heatmap
        fig2.add_trace(go.Heatmap(
            x=t_map, y=z_map, z=np.clip(fs_grid, 0, 5),
            colorscale=[
                [0.0, "#7f1d1d"], [0.15, "#ef4444"], [0.20, "#f97316"],
                [0.30, "#eab308"], [0.50, "#22c55e"], [1.0, "#0ea5e9"],
            ],
            colorbar=dict(title="FS"),
            hovertemplate="Time: %{x:.1f}d<br>Depth: %{y:.1f}m<br>FS: %{z:.3f}<extra></extra>",
        ))

        # Critical slip line
        fig2.add_trace(go.Scatter(
            x=crit["times"], y=crit["critical_depths"],
            mode="lines+markers",
            line=dict(color="#1e293b", width=3, dash="dash"),
            marker=dict(size=5, color="#1e293b", symbol="diamond"),
            name="Critical Slip Surface",
        ))

        # Mark the worst point
        fig2.add_trace(go.Scatter(
            x=[crit["times"][idx_worst]],
            y=[crit["critical_depths"][idx_worst]],
            mode="markers",
            marker=dict(size=15, color="#ef4444", symbol="x", line=dict(width=2, color="#1e293b")),
            name=f"Worst FS = {crit['min_fs'][idx_worst]:.3f}",
        ))

        fig2.update_layout(
            xaxis_title="Time (days)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            template="plotly_white", height=500,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig2, width="stretch")

    # ── Tab 3: Data table ────────────────────────────────────────────
    with tab_table:
        df_crit = pd.DataFrame({
            "Time (days)": crit["times"],
            "Critical Depth (m)": crit["critical_depths"],
            "Min FS": crit["min_fs"],
            "Status": ["🟢 Safe" if fs > 1.5 else "🟡 Marginal" if fs > 1.0 else "🔴 Failure"
                       for fs in crit["min_fs"]],
        })
        st.dataframe(df_crit, hide_index=True, width="stretch")

        st.download_button(
            "⬇️ Download Critical Slip Data (.csv)",
            df_crit.to_csv(index=False),
            "critical_slip_surface.csv", "text/csv",
        )
