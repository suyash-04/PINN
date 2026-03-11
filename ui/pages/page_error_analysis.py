"""
Convergence & Error Analysis Page
==================================
Detailed error analysis with advanced hydrology metrics.
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

from model_inference import predict_psi, compute_hydro_metrics


def render(model, df_hydrus, geo, norm):
    st.markdown("## 📈 Convergence & Error Analysis")
    st.caption("Advanced validation metrics from hydrology: NSE, KGE, PBIAS, and per-timestep error decomposition")

    available_times = sorted(df_hydrus["Time_days"].unique())

    # ── Overall Metrics ──────────────────────────────────────────────
    with st.spinner("Computing global validation metrics …"):
        z_all = df_hydrus["Depth_m"].values
        t_all = df_hydrus["Time_days"].values
        psi_obs = df_hydrus["Pressure_Head"].values
        psi_pred = predict_psi(model, z_all, t_all, norm)
        metrics = compute_hydro_metrics(psi_obs, psi_pred)

    st.markdown("### 🏆 Global Validation Metrics")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    metric_colors = {
        "R²": ("#22c55e" if metrics["R2"] > 0.95 else "#eab308" if metrics["R2"] > 0.8 else "#ef4444"),
        "NSE": ("#22c55e" if metrics["NSE"] > 0.75 else "#eab308" if metrics["NSE"] > 0.5 else "#ef4444"),
        "KGE": ("#22c55e" if metrics["KGE"] > 0.75 else "#eab308" if metrics["KGE"] > 0.5 else "#ef4444"),
    }

    m1.metric("R²", f"{metrics['R2']:.5f}")
    m2.metric("RMSE (m)", f"{metrics['RMSE']:.3f}")
    m3.metric("MAE (m)", f"{metrics['MAE']:.3f}")
    m4.metric("NSE", f"{metrics['NSE']:.5f}")
    m5.metric("KGE", f"{metrics['KGE']:.5f}")
    m6.metric("PBIAS (%)", f"{metrics['PBIAS']:.2f}")
    m7.metric("nRMSE (%)", f"{metrics['nRMSE']:.2f}")

    # Rating box
    if metrics["NSE"] > 0.75 and metrics["KGE"] > 0.75:
        st.success("✅ **Excellent** — Model performance rated 'Very Good' by Moriasi et al. (2007) criteria")
    elif metrics["NSE"] > 0.5:
        st.warning("⚠️ **Satisfactory** — NSE > 0.5, but there is room for improvement")
    else:
        st.error("❌ **Unsatisfactory** — NSE < 0.5 indicates poor model performance")

    st.info("""
    **Metric guide:**
    - **R²**: Coefficient of determination (≥ 0.95 = excellent)
    - **NSE**: Nash-Sutcliffe Efficiency (1 = perfect, 0 = mean, < 0 = worse than mean) — *standard in hydrology*
    - **KGE**: Kling-Gupta Efficiency (1 = perfect) — *decomposes into correlation, variability, bias*
    - **PBIAS**: Percent Bias (0% = unbiased, positive = overestimation)
    - **nRMSE**: Normalised RMSE (< 10% = excellent, < 20% = good)
    """)

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_temporal, tab_spatial, tab_dist, tab_taylor = st.tabs([
        "⏱️ Per-Timestep Metrics", "📏 Per-Depth Metrics",
        "📊 Error Distribution", "🎯 Taylor Diagram"
    ])

    palette = ["#38bdf8", "#818cf8", "#f472b6", "#fb923c", "#22c55e",
               "#ef4444", "#eab308", "#a78bfa", "#06b6d4", "#d946ef"]

    # ── Tab 1: Per-timestep ──────────────────────────────────────────
    with tab_temporal:
        rows = []
        for t_val in available_times:
            mask = df_hydrus["Time_days"] == t_val
            obs = df_hydrus.loc[mask, "Pressure_Head"].values
            z_t = df_hydrus.loc[mask, "Depth_m"].values
            t_t = np.full_like(z_t, t_val)
            pred = predict_psi(model, z_t, t_t, norm)
            m = compute_hydro_metrics(obs, pred)
            m["Time (days)"] = t_val
            m["N"] = len(obs)
            rows.append(m)

        df_metrics = pd.DataFrame(rows)
        col_order = ["Time (days)", "N", "R2", "RMSE", "MAE", "NSE", "KGE", "PBIAS", "nRMSE"]
        st.dataframe(df_metrics[col_order].round(5), hide_index=True, width="stretch")

        # Plot NSE and KGE over time
        fig_tm = make_subplots(rows=1, cols=2,
                               subplot_titles=["NSE over Time", "KGE over Time"])

        fig_tm.add_trace(go.Scatter(
            x=df_metrics["Time (days)"], y=df_metrics["NSE"],
            mode="lines+markers", line=dict(color="#38bdf8", width=2),
            name="NSE", marker=dict(size=6),
        ), row=1, col=1)
        fig_tm.add_hline(y=0.75, line_dash="dash", line_color="#22c55e", row=1, col=1,
                         annotation_text="Good (0.75)")
        fig_tm.add_hline(y=0.5, line_dash="dot", line_color="#eab308", row=1, col=1,
                         annotation_text="Satisfactory (0.5)")

        fig_tm.add_trace(go.Scatter(
            x=df_metrics["Time (days)"], y=df_metrics["KGE"],
            mode="lines+markers", line=dict(color="#f472b6", width=2),
            name="KGE", marker=dict(size=6),
        ), row=1, col=2)
        fig_tm.add_hline(y=0.75, line_dash="dash", line_color="#22c55e", row=1, col=2)

        fig_tm.update_layout(
            template="plotly_white", height=380,
            margin=dict(l=60, r=20, t=40, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_tm, width="stretch")

    # ── Tab 2: Per-depth ─────────────────────────────────────────────
    with tab_spatial:
        # Bin depths
        n_bins = st.slider("Depth bins", 5, 30, 15, key="err_zbins")
        df_hydrus_copy = df_hydrus.copy()
        df_hydrus_copy["Depth_bin"] = pd.cut(df_hydrus_copy["Depth_m"], bins=n_bins)

        z_all_d = df_hydrus_copy["Depth_m"].values
        t_all_d = df_hydrus_copy["Time_days"].values
        psi_pred_d = predict_psi(model, z_all_d, t_all_d, norm)
        df_hydrus_copy["Psi_PINN"] = psi_pred_d
        df_hydrus_copy["Error"] = psi_pred_d - df_hydrus_copy["Pressure_Head"].values

        depth_stats = df_hydrus_copy.groupby("Depth_bin", observed=True).agg(
            Mean_Error=("Error", "mean"),
            RMSE=("Error", lambda x: np.sqrt(np.mean(x**2))),
            MAE=("Error", lambda x: np.mean(np.abs(x))),
            Count=("Error", "count"),
        ).reset_index()
        depth_stats["Depth_mid"] = depth_stats["Depth_bin"].apply(lambda x: x.mid)

        fig_sp = make_subplots(rows=1, cols=2,
                               subplot_titles=["RMSE by Depth", "Mean Error (Bias) by Depth"])

        fig_sp.add_trace(go.Scatter(
            x=depth_stats["RMSE"], y=depth_stats["Depth_mid"],
            mode="lines+markers", line=dict(color="#818cf8", width=2),
            name="RMSE",
        ), row=1, col=1)

        fig_sp.add_trace(go.Scatter(
            x=depth_stats["Mean_Error"], y=depth_stats["Depth_mid"],
            mode="lines+markers", line=dict(color="#fb923c", width=2),
            name="Bias",
        ), row=1, col=2)
        fig_sp.add_vline(x=0, line_dash="dash", line_color="#64748b", row=1, col=2)

        fig_sp.update_yaxes(autorange="reversed", title_text="Depth (m)")
        fig_sp.update_layout(
            template="plotly_white", height=400,
            margin=dict(l=60, r=20, t=40, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_sp, width="stretch")

    # ── Tab 3: Error Distribution ────────────────────────────────────
    with tab_dist:
        errors = psi_pred - psi_obs

        fig_dist = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Error Histogram", "Q-Q Plot (vs Normal)"])

        fig_dist.add_trace(go.Histogram(
            x=errors, nbinsx=60, marker_color="#38bdf8", name="Error",
        ), row=1, col=1)
        fig_dist.add_vline(x=0, line_dash="dash", line_color="#ef4444", row=1, col=1)

        # Q-Q plot
        sorted_err = np.sort(errors)
        n_qq = len(sorted_err)
        theoretical = np.sort(np.random.normal(np.mean(errors), np.std(errors), n_qq))
        fig_dist.add_trace(go.Scatter(
            x=theoretical, y=sorted_err, mode="markers",
            marker=dict(color="#818cf8", size=2), name="Q-Q",
        ), row=1, col=2)
        qq_range = [min(theoretical.min(), sorted_err.min()),
                    max(theoretical.max(), sorted_err.max())]
        fig_dist.add_trace(go.Scatter(
            x=qq_range, y=qq_range, mode="lines",
            line=dict(color="#ef4444", dash="dash"), name="1:1",
        ), row=1, col=2)

        fig_dist.update_layout(
            template="plotly_white", height=400,
            margin=dict(l=60, r=20, t=40, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_dist, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Error", f"{np.mean(errors):.4f} m")
        c2.metric("Std Error", f"{np.std(errors):.4f} m")
        c3.metric("Skewness", f"{float(pd.Series(errors).skew()):.4f}")
        c4.metric("Kurtosis", f"{float(pd.Series(errors).kurtosis()):.4f}")

    # ── Tab 4: Taylor Diagram (simplified) ───────────────────────────
    with tab_taylor:
        st.markdown("### 🎯 Taylor-style Decomposition")
        st.markdown("""
        A Taylor diagram summarises three metrics simultaneously:
        - **Standard deviation ratio** (model / observed)
        - **Correlation** (Pearson r)
        - **Centred RMSE** (distance from perfect)
        """)

        rows_t = []
        for t_val in available_times:
            mask = df_hydrus["Time_days"] == t_val
            obs = df_hydrus.loc[mask, "Pressure_Head"].values
            z_t = df_hydrus.loc[mask, "Depth_m"].values
            pred = predict_psi(model, z_t, np.full_like(z_t, t_val), norm)
            r = np.corrcoef(obs, pred)[0, 1]
            std_ratio = np.std(pred) / max(np.std(obs), 1e-10)
            crmse = np.sqrt(np.mean(((pred - np.mean(pred)) - (obs - np.mean(obs)))**2))
            rows_t.append(dict(time=t_val, r=r, std_ratio=std_ratio, crmse=crmse))

        df_taylor = pd.DataFrame(rows_t)

        fig_tay = go.Figure()
        fig_tay.add_trace(go.Scatter(
            x=df_taylor["std_ratio"] * df_taylor["r"],
            y=df_taylor["std_ratio"] * np.sqrt(1 - df_taylor["r"]**2),
            mode="markers+text",
            text=[f"d{t:.0f}" for t in df_taylor["time"]],
            textposition="top center",
            marker=dict(size=8, color=df_taylor["crmse"],
                       colorscale="Viridis_r", colorbar=dict(title="CRMSE"),
                       showscale=True),
            hovertemplate="Time: %{text}<br>Correlation: %{customdata[0]:.4f}<br>Std ratio: %{customdata[1]:.4f}<extra></extra>",
            customdata=np.column_stack([df_taylor["r"], df_taylor["std_ratio"]]),
        ))

        # Perfect point
        fig_tay.add_trace(go.Scatter(
            x=[1], y=[0], mode="markers",
            marker=dict(size=15, color="#ef4444", symbol="star"),
            name="Perfect Model", showlegend=True,
        ))

        fig_tay.update_layout(
            xaxis_title="σ_model/σ_obs × r (correlation component)",
            yaxis_title="σ_model/σ_obs × √(1-r²) (error component)",
            template="plotly_white", height=450,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_tay, width="stretch")

        st.dataframe(df_taylor.round(5), hide_index=True, width="stretch")
