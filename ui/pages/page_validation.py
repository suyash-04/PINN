"""
Validation Metrics Page (Dimensional Analysis)
================================================
Standard hydrology validation: NSE, KGE, PBIAS, nRMSE.
(Thin wrapper that redirects to page_error_analysis if needed,
 but also provides per-variable dimensional checks.)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import predict_psi, compute_FS, compute_hydro_metrics


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🧮 Dimensional Analysis & Validation")
    st.caption("Rigorous dimensional checks + standard hydrology validation criteria (Moriasi et al. 2007)")

    tab_dim, tab_bench, tab_perf = st.tabs([
        "📐 Dimensional Check", "📊 Benchmark Criteria", "⚡ Performance"
    ])

    # ── Tab 1: Dimensional Consistency ───────────────────────────────
    with tab_dim:
        st.markdown("### Physical Dimensional Consistency")
        st.markdown("""
        The Richards' equation and the infinite slope model involve several physical
        quantities. Here we verify that the PINN predictions stay within physically
        plausible ranges.
        """)

        # Sample predictions
        z_check = np.linspace(0.5, norm["z_max"] * 0.9, 200)
        t_check = np.array([0, 30, 60, 90, 96, 123])

        checks = []
        for t_val in t_check:
            t_arr = np.full_like(z_check, t_val)
            psi = predict_psi(model, z_check, t_arr, norm)
            fs = compute_FS(psi, z_check, geo)

            checks.append({
                "Time (day)": t_val,
                "ψ_min (m)": f"{psi.min():.2f}",
                "ψ_max (m)": f"{psi.max():.2f}",
                "ψ < 0 (unsaturated)": f"{np.mean(psi < 0)*100:.1f}%",
                "FS_min": f"{fs.min():.4f}",
                "FS_max": f"{fs.max():.2f}",
                "FS > 0 (physical)": "✅" if np.all(fs >= 0) else "❌",
            })

        st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")

        # Physical range checks
        st.divider()
        st.markdown("### 🔍 Physical Range Validation")

        all_psi = predict_psi(model, df_hydrus["Depth_m"].values,
                              df_hydrus["Time_days"].values, norm)

        range_checks = [
            ("Pressure head ψ",
             f"{all_psi.min():.1f} to {all_psi.max():.1f} m",
             "ψ < 0 for unsaturated soil" if np.all(all_psi < 0) else "⚠️ Some ψ ≥ 0 (saturated zones exist)",
             "✅" if all_psi.min() > -1000 else "⚠️"),
            ("HYDRUS ψ range",
             f"{df_hydrus['Pressure_Head'].min():.1f} to {df_hydrus['Pressure_Head'].max():.1f} m",
             "Training data range",
             "✅"),
            ("Normalisation: t_max",
             f"{norm['t_max']} days",
             f"Last time step = {df_hydrus['Time_days'].max()} days",
             "✅" if norm['t_max'] >= df_hydrus['Time_days'].max() else "❌"),
            ("Normalisation: z_max",
             f"{norm['z_max']} m",
             f"Max depth = {df_hydrus['Depth_m'].max():.1f} m",
             "✅" if norm['z_max'] >= df_hydrus['Depth_m'].max() else "❌"),
        ]

        for name, val, note, status in range_checks:
            c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
            c1.write(f"**{name}**")
            c2.code(val)
            c3.write(note)
            c4.write(status)

    # ── Tab 2: Benchmark Criteria ────────────────────────────────────
    with tab_bench:
        st.markdown("### 📊 Moriasi et al. (2007) Performance Ratings")
        st.markdown("""
        The standard classification for hydrological model performance:

        | Rating | NSE | RSR | PBIAS (%) |
        |--------|-----|-----|-----------|
        | **Very Good** | 0.75 < NSE ≤ 1.00 | 0.00 ≤ RSR ≤ 0.50 | PBIAS < ±10 |
        | **Good** | 0.65 < NSE ≤ 0.75 | 0.50 < RSR ≤ 0.60 | ±10 ≤ PBIAS < ±15 |
        | **Satisfactory** | 0.50 < NSE ≤ 0.65 | 0.60 < RSR ≤ 0.70 | ±15 ≤ PBIAS < ±25 |
        | **Unsatisfactory** | NSE ≤ 0.50 | RSR > 0.70 | PBIAS ≥ ±25 |
        """)

        # Compute global metrics
        obs = df_hydrus["Pressure_Head"].values
        pred = predict_psi(model, df_hydrus["Depth_m"].values,
                           df_hydrus["Time_days"].values, norm)
        metrics = compute_hydro_metrics(obs, pred)

        rmse = metrics["RMSE"]
        obs_std = np.std(obs)
        rsr = rmse / max(obs_std, 1e-10)

        # Rating determination
        def rate_nse(v):
            if v > 0.75: return "🟢 Very Good"
            if v > 0.65: return "🔵 Good"
            if v > 0.50: return "🟡 Satisfactory"
            return "🔴 Unsatisfactory"

        def rate_rsr(v):
            if v <= 0.50: return "🟢 Very Good"
            if v <= 0.60: return "🔵 Good"
            if v <= 0.70: return "🟡 Satisfactory"
            return "🔴 Unsatisfactory"

        def rate_pbias(v):
            v = abs(v)
            if v < 10: return "🟢 Very Good"
            if v < 15: return "🔵 Good"
            if v < 25: return "🟡 Satisfactory"
            return "🔴 Unsatisfactory"

        results_df = pd.DataFrame([
            {"Metric": "NSE", "Value": f"{metrics['NSE']:.5f}", "Rating": rate_nse(metrics['NSE'])},
            {"Metric": "RSR", "Value": f"{rsr:.5f}", "Rating": rate_rsr(rsr)},
            {"Metric": "PBIAS (%)", "Value": f"{metrics['PBIAS']:.2f}", "Rating": rate_pbias(metrics['PBIAS'])},
            {"Metric": "R²", "Value": f"{metrics['R2']:.5f}", "Rating": rate_nse(metrics['R2'])},
            {"Metric": "KGE", "Value": f"{metrics['KGE']:.5f}", "Rating": rate_nse(metrics['KGE'])},
        ])

        st.dataframe(results_df, hide_index=True, width="stretch")

        # Visual gauge
        fig_gauge = make_subplots(rows=1, cols=3,
                                  specs=[[{"type": "indicator"}]*3],
                                  subplot_titles=["NSE", "KGE", "R²"])

        for i, (name, val) in enumerate([("NSE", metrics["NSE"]),
                                           ("KGE", metrics["KGE"]),
                                           ("R²", metrics["R2"])]):
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number",
                value=val,
                gauge=dict(
                    axis=dict(range=[0, 1]),
                    bar=dict(color="#38bdf8"),
                    steps=[
                        dict(range=[0, 0.5], color="#7f1d1d"),
                        dict(range=[0.5, 0.65], color="#92400e"),
                        dict(range=[0.65, 0.75], color="#854d0e"),
                        dict(range=[0.75, 1.0], color="#14532d"),
                    ],
                ),
            ), row=1, col=i+1)

        fig_gauge.update_layout(
            height=250,
            paper_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#1e293b"),
        )
        st.plotly_chart(fig_gauge, width="stretch")

    # ── Tab 3: Computational Performance ─────────────────────────────
    with tab_perf:
        st.markdown("### ⚡ Inference Performance Benchmark")
        st.markdown("Compare PINN inference speed vs typical FEM/FDM solve times")

        # Benchmark PINN inference
        n_bench = 10000
        z_bench = np.random.uniform(0.5, norm["z_max"] * 0.9, n_bench)
        t_bench = np.random.uniform(0, norm["t_max"], n_bench)

        start = time.perf_counter()
        _ = predict_psi(model, z_bench, t_bench, norm)
        elapsed = time.perf_counter() - start

        total_params = sum(p.numel() for p in model.parameters())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Inference Time", f"{elapsed*1000:.1f} ms")
        c2.metric("Points Evaluated", f"{n_bench:,}")
        c3.metric("Throughput", f"{n_bench/elapsed:,.0f} pts/s")
        c4.metric("Model Size", f"{total_params*4/1024:.1f} KB")

        st.divider()

        comparison = pd.DataFrame([
            {"Method": "PINN (this model)", "Solve Time": f"{elapsed*1000:.1f} ms",
             "Points": f"{n_bench:,}", "Reusable": "✅ Yes"},
            {"Method": "HYDRUS-1D (estimated)", "Solve Time": "~30-120 s",
             "Points": "1,000 nodes", "Reusable": "❌ Per-scenario"},
            {"Method": "FEM (PLAXIS/GeoStudio)", "Solve Time": "~5-30 min",
             "Points": "~10k elements", "Reusable": "❌ Per-scenario"},
        ])
        st.dataframe(comparison, hide_index=True, width="stretch")

        st.markdown(f"""
        **Key advantage:** The PINN evaluates **{n_bench:,} points in {elapsed*1000:.1f} ms** —
        orders of magnitude faster than numerical solvers. Once trained, it acts as a
        **surrogate model** for real-time parametric studies.
        """)
