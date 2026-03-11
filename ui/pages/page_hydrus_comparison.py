"""
HYDRUS vs PINN Comparison Page
===============================
Side-by-side and overlay comparison of the HYDRUS-1D simulation data
with the trained PINN predictions.
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

from model_inference import predict_psi


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🔬 HYDRUS-1D vs PINN Comparison")
    st.caption("Quantitative comparison of simulation data against neural network predictions")

    available_times = sorted(df_hydrus["Time_days"].unique())

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        selected_times = st.multiselect(
            "Time steps to compare (days)", available_times,
            default=[0.0, 60.0, 90.0, 96.0, 123.0],
            key="comp_times",
        )
    with c2:
        metric_type = st.radio("Error metric", ["Absolute Error", "Relative Error (%)"],
                               horizontal=True, key="comp_metric")

    if not selected_times:
        st.warning("Select at least one time step.")
        return

    # ── Tab layout ───────────────────────────────────────────────────
    tab_overlay, tab_error, tab_scatter, tab_table = st.tabs([
        "📈 Overlay", "📊 Error Profile", "🔘 Scatter Plot", "📋 Statistics"
    ])

    palette = ["#38bdf8", "#818cf8", "#f472b6", "#fb923c", "#22c55e",
               "#ef4444", "#eab308", "#a78bfa", "#06b6d4", "#d946ef"]

    # ── TAB 1: Overlay ───────────────────────────────────────────────
    with tab_overlay:
        fig_ov = go.Figure()

        for i, t_val in enumerate(selected_times):
            color = palette[i % len(palette)]

            # HYDRUS data
            hyd = df_hydrus[df_hydrus["Time_days"] == t_val].sort_values("Depth_m")
            z_hyd = hyd["Depth_m"].values
            psi_hyd = hyd["Pressure_Head"].values

            fig_ov.add_trace(go.Scatter(
                x=psi_hyd, y=z_hyd,
                mode="markers", name=f"HYDRUS t={t_val:.0f}d",
                marker=dict(color=color, size=5, symbol="circle-open"),
                legendgroup=f"t{t_val}",
            ))

            # PINN prediction at same points
            t_arr = np.full_like(z_hyd, t_val)
            psi_pinn = predict_psi(model, z_hyd, t_arr, norm)

            fig_ov.add_trace(go.Scatter(
                x=psi_pinn, y=z_hyd,
                mode="lines", name=f"PINN t={t_val:.0f}d",
                line=dict(color=color, width=2.5),
                legendgroup=f"t{t_val}",
            ))

        fig_ov.update_layout(
            xaxis_title="Pressure Head ψ (m)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            template="plotly_white", height=550,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_ov, width="stretch")

    # ── TAB 2: Error Profile ─────────────────────────────────────────
    with tab_error:
        fig_err = go.Figure()

        for i, t_val in enumerate(selected_times):
            color = palette[i % len(palette)]
            hyd = df_hydrus[df_hydrus["Time_days"] == t_val].sort_values("Depth_m")
            z_hyd = hyd["Depth_m"].values
            psi_hyd = hyd["Pressure_Head"].values
            t_arr = np.full_like(z_hyd, t_val)
            psi_pinn = predict_psi(model, z_hyd, t_arr, norm)

            if "Relative" in metric_type:
                err = np.abs((psi_pinn - psi_hyd) / np.maximum(np.abs(psi_hyd), 1e-6)) * 100
                x_label = "Relative Error (%)"
            else:
                err = np.abs(psi_pinn - psi_hyd)
                x_label = "Absolute Error |ψ_PINN − ψ_HYDRUS| (m)"

            fig_err.add_trace(go.Scatter(
                x=err, y=z_hyd,
                mode="lines+markers", name=f"t={t_val:.0f}d",
                line=dict(color=color, width=2),
                marker=dict(size=3),
            ))

        fig_err.update_layout(
            xaxis_title=x_label,
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            template="plotly_white", height=500,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_err, width="stretch")

    # ── TAB 3: Scatter Plot (HYDRUS vs PINN) ────────────────────────
    with tab_scatter:
        all_psi_hyd = []
        all_psi_pinn = []
        all_labels = []

        for t_val in selected_times:
            hyd = df_hydrus[df_hydrus["Time_days"] == t_val].sort_values("Depth_m")
            z_hyd = hyd["Depth_m"].values
            psi_hyd = hyd["Pressure_Head"].values
            t_arr = np.full_like(z_hyd, t_val)
            psi_pinn = predict_psi(model, z_hyd, t_arr, norm)
            all_psi_hyd.extend(psi_hyd)
            all_psi_pinn.extend(psi_pinn)
            all_labels.extend([f"t={t_val:.0f}d"] * len(psi_hyd))

        all_psi_hyd = np.array(all_psi_hyd)
        all_psi_pinn = np.array(all_psi_pinn)

        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=all_psi_hyd, y=all_psi_pinn,
            mode="markers", text=all_labels,
            marker=dict(color="#38bdf8", size=4, opacity=0.6),
            name="Data points",
        ))

        # Perfect prediction line
        psi_range = [min(all_psi_hyd.min(), all_psi_pinn.min()),
                     max(all_psi_hyd.max(), all_psi_pinn.max())]
        fig_scat.add_trace(go.Scatter(
            x=psi_range, y=psi_range,
            mode="lines", name="Perfect prediction",
            line=dict(color="#ef4444", dash="dash", width=2),
        ))

        # R² calculation
        ss_res = np.sum((all_psi_pinn - all_psi_hyd) ** 2)
        ss_tot = np.sum((all_psi_hyd - np.mean(all_psi_hyd)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        rmse = np.sqrt(np.mean((all_psi_pinn - all_psi_hyd) ** 2))
        mae = np.mean(np.abs(all_psi_pinn - all_psi_hyd))

        fig_scat.update_layout(
            xaxis_title="ψ HYDRUS (m)",
            yaxis_title="ψ PINN (m)",
            template="plotly_white", height=500,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            annotations=[dict(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text=f"R² = {r2:.4f}<br>RMSE = {rmse:.2f} m<br>MAE = {mae:.2f} m",
                showarrow=False, font=dict(size=14, color="#1e293b"),
                bgcolor="rgba(241,245,249,0.9)", bordercolor="#cbd5e1", borderpad=8,
            )],
        )
        st.plotly_chart(fig_scat, width="stretch")

    # ── TAB 4: Statistics Table ──────────────────────────────────────
    with tab_table:
        rows = []
        for t_val in selected_times:
            hyd = df_hydrus[df_hydrus["Time_days"] == t_val]
            z_hyd = hyd["Depth_m"].values
            psi_hyd = hyd["Pressure_Head"].values
            t_arr = np.full_like(z_hyd, t_val)
            psi_pinn = predict_psi(model, z_hyd, t_arr, norm)

            err = psi_pinn - psi_hyd
            abs_err = np.abs(err)
            rel_err = abs_err / np.maximum(np.abs(psi_hyd), 1e-6) * 100

            ss_res = np.sum(err ** 2)
            ss_tot = np.sum((psi_hyd - np.mean(psi_hyd)) ** 2)
            r2_t = 1 - ss_res / max(ss_tot, 1e-10)

            rows.append({
                "Time (days)": f"{t_val:.0f}",
                "N points": len(psi_hyd),
                "MAE (m)": f"{np.mean(abs_err):.3f}",
                "RMSE (m)": f"{np.sqrt(np.mean(err**2)):.3f}",
                "Max Error (m)": f"{np.max(abs_err):.3f}",
                "Mean Rel. Error (%)": f"{np.mean(rel_err):.2f}",
                "R²": f"{r2_t:.5f}",
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # Overall metrics
        st.divider()
        o1, o2, o3 = st.columns(3)
        o1.metric("Overall R²", f"{r2:.5f}")
        o2.metric("Overall RMSE (m)", f"{rmse:.3f}")
        o3.metric("Overall MAE (m)", f"{mae:.3f}")
