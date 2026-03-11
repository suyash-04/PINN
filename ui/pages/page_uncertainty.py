"""
Monte Carlo Uncertainty Quantification Page
============================================
Parameter perturbation → FS confidence bands.
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

from model_inference import predict_psi, compute_FS, monte_carlo_fs


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🎯 Uncertainty Quantification (Monte Carlo)")
    st.caption("Propagate geotechnical parameter uncertainty through the PINN → FS confidence bands")

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_samples = st.slider("Monte Carlo samples", 50, 500, 200, 50, key="mc_n")
    with c2:
        cov_frac = st.slider("CoV (% uncertainty)", 1, 30, 10, 1, key="mc_cov") / 100.0
    with c3:
        t_val = st.selectbox("Time (days)",
                              sorted(df_hydrus["Time_days"].unique()),
                              index=18, key="mc_t")  # default ~day 96
    with c4:
        z_max_plot = st.slider("Max depth (m)", 5.0, 55.0, 40.0, 5.0, key="mc_zmax")

    z_arr = np.linspace(0.5, z_max_plot, 120)

    # ── Run Monte Carlo ──────────────────────────────────────────────
    with st.spinner(f"Running {n_samples} Monte Carlo samples …"):
        mc = monte_carlo_fs(model, z_arr, t_val, geo, norm, n_samples, cov_frac)

    tab_band, tab_hist, tab_stats = st.tabs([
        "📈 Confidence Bands", "📊 Histograms", "📋 Statistics"
    ])

    # ── Tab 1: Confidence Band Plot ──────────────────────────────────
    with tab_band:
        fig = go.Figure()

        # 90% confidence band (P5-P95)
        fig.add_trace(go.Scatter(
            x=np.concatenate([mc["p95"], mc["p5"][::-1]]),
            y=np.concatenate([z_arr, z_arr[::-1]]),
            fill="toself", fillcolor="rgba(56,189,248,0.15)",
            line=dict(width=0), name="90% CI",
            hoverinfo="skip",
        ))

        # ±1σ band
        fig.add_trace(go.Scatter(
            x=np.concatenate([mc["mean"] + mc["std"], (mc["mean"] - mc["std"])[::-1]]),
            y=np.concatenate([z_arr, z_arr[::-1]]),
            fill="toself", fillcolor="rgba(129,140,248,0.2)",
            line=dict(width=0), name="±1σ",
            hoverinfo="skip",
        ))

        # Mean FS
        fig.add_trace(go.Scatter(
            x=mc["mean"], y=z_arr, mode="lines",
            line=dict(color="#38bdf8", width=3), name="Mean FS",
        ))

        # Deterministic FS (no perturbation)
        t_arr_det = np.full_like(z_arr, t_val)
        psi_det = predict_psi(model, z_arr, t_arr_det, norm)
        fs_det = compute_FS(psi_det, z_arr, geo)
        fig.add_trace(go.Scatter(
            x=fs_det, y=z_arr, mode="lines",
            line=dict(color="#fb923c", width=2, dash="dash"), name="Deterministic FS",
        ))

        # FS = 1 reference
        fig.add_vline(x=1.0, line_dash="dot", line_color="#ef4444",
                      annotation_text="FS = 1")

        fig.update_layout(
            xaxis_title="Factor of Safety",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(range=[0, min(float(mc["p95"].max()) + 1, 15)]),
            template="plotly_white", height=550,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Tab 2: Histograms at specific depths ─────────────────────────
    with tab_hist:
        sel_depths = st.multiselect("Depths for histograms (m)",
                                     [float(f"{z:.1f}") for z in z_arr[::10]],
                                     default=[float(f"{z_arr[5]:.1f}"),
                                              float(f"{z_arr[30]:.1f}"),
                                              float(f"{z_arr[60]:.1f}")],
                                     key="mc_hist_z")

        if sel_depths:
            n_hist = len(sel_depths)
            fig_h = make_subplots(rows=1, cols=n_hist,
                                  subplot_titles=[f"z = {d:.1f} m" for d in sel_depths])
            colors = ["#38bdf8", "#818cf8", "#f472b6", "#22c55e", "#fb923c"]

            for i, zv in enumerate(sel_depths):
                z_idx = np.argmin(np.abs(z_arr - zv))
                fs_samples = mc["samples"][:, z_idx]
                fig_h.add_trace(go.Histogram(
                    x=fs_samples, nbinsx=30,
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ), row=1, col=i+1)
                fig_h.add_vline(x=1.0, line_dash="dash", line_color="#ef4444",
                                row=1, col=i+1)

            fig_h.update_layout(
                template="plotly_white", height=350,
                margin=dict(l=60, r=20, t=40, b=60),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig_h, width="stretch")

    # ── Tab 3: Statistics ────────────────────────────────────────────
    with tab_stats:
        # Probability of failure at each depth
        prob_failure = np.mean(mc["samples"] < 1.0, axis=0)

        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(
            x=prob_failure * 100, y=z_arr, mode="lines",
            line=dict(color="#ef4444", width=3),
            fill="tozerox", fillcolor="rgba(239,68,68,0.15)",
            name="P(FS < 1)",
        ))
        fig_pf.update_layout(
            xaxis_title="Probability of Failure P(FS < 1) [%]",
            yaxis_title="Depth (m)", yaxis=dict(autorange="reversed"),
            xaxis=dict(range=[0, 100]),
            template="plotly_white", height=400,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_pf, width="stretch")

        # Summary metrics
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max P(failure)", f"{prob_failure.max()*100:.1f}%")
        c2.metric("Depth of max P(f)", f"{z_arr[np.argmax(prob_failure)]:.1f} m")
        c3.metric("Mean CoV of FS", f"{np.mean(mc['std']/np.maximum(mc['mean'],1e-6))*100:.1f}%")
        c4.metric("Min P5 FS", f"{mc['p5'].min():.3f}")

        st.info(f"""
        **Parameters perturbed:** c′, φ′, β, α, n, Ks
        **CoV applied:** ±{cov_frac*100:.0f}% (Gaussian)
        **Samples:** {n_samples}
        """)
