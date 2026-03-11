"""
What-If Scenario Comparator Page
=================================
Save parameter snapshots as scenarios and compare side-by-side.
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

from model_inference import predict_psi, compute_FS


# Pre-defined scenarios
PRESET_SCENARIOS = {
    "Current (sidebar params)": None,  # Will use current geo
    "Dry Slope (high suction)": dict(
        Ks=1.5e-5, theta_s=0.45, theta_r=0.05, alpha=0.8, n=1.4, l=0.5,
        c_prime=8.0, phi_prime=32.0, gamma=19.0, beta=30.0, gamma_w=9.81,
    ),
    "Wet Slope (near saturation)": dict(
        Ks=1.5e-5, theta_s=0.45, theta_r=0.05, alpha=0.8, n=1.4, l=0.5,
        c_prime=3.0, phi_prime=24.0, gamma=21.0, beta=42.0, gamma_w=9.81,
    ),
    "Steep Clay Slope": dict(
        Ks=5e-7, theta_s=0.50, theta_r=0.08, alpha=0.5, n=1.2, l=0.5,
        c_prime=15.0, phi_prime=20.0, gamma=18.0, beta=55.0, gamma_w=9.81,
    ),
    "Gentle Sandy Slope": dict(
        Ks=1e-4, theta_s=0.38, theta_r=0.03, alpha=3.0, n=2.5, l=0.5,
        c_prime=0.0, phi_prime=35.0, gamma=17.0, beta=20.0, gamma_w=9.81,
    ),
    "Critical (marginal stability)": dict(
        Ks=1.5e-5, theta_s=0.45, theta_r=0.05, alpha=0.8, n=1.4, l=0.5,
        c_prime=2.0, phi_prime=22.0, gamma=22.0, beta=48.0, gamma_w=9.81,
    ),
}


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🔄 What-If Scenario Comparator")
    st.caption("Compare multiple parameter scenarios side-by-side to understand slope sensitivity")

    # ── Scenario Selection ───────────────────────────────────────────
    st.markdown("### 📋 Select Scenarios to Compare")

    available = list(PRESET_SCENARIOS.keys())
    selected = st.multiselect(
        "Choose 2-4 scenarios", available,
        default=["Current (sidebar params)", "Dry Slope (high suction)", "Wet Slope (near saturation)"],
        key="whatif_scenarios",
    )

    if len(selected) < 2:
        st.warning("Select at least 2 scenarios to compare.")
        return

    # Build geo dicts
    scenarios = {}
    for name in selected:
        if PRESET_SCENARIOS[name] is None:
            scenarios[name] = dict(geo)  # Current sidebar values
        else:
            scenarios[name] = dict(PRESET_SCENARIOS[name])

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        t_val = st.selectbox("Time (days)", sorted(df_hydrus["Time_days"].unique()),
                              index=18, key="whatif_t")
    with c2:
        z_max_plot = st.slider("Max depth (m)", 5.0, 55.0, 40.0, 5.0, key="whatif_zmax")
    with c3:
        z_res = st.slider("Depth points", 30, 200, 100, 10, key="whatif_zres")

    z_arr = np.linspace(0.5, z_max_plot, z_res)
    t_arr = np.full_like(z_arr, t_val)

    # PINN prediction is the same for all scenarios (same model)
    psi_pred = predict_psi(model, z_arr, t_arr, norm)

    palette = ["#38bdf8", "#f472b6", "#22c55e", "#fb923c", "#818cf8", "#eab308"]

    tab_profiles, tab_params, tab_summary = st.tabs([
        "📈 Profile Comparison", "📋 Parameter Table", "📊 Summary"
    ])

    # ── Tab 1: Side-by-side profiles ─────────────────────────────────
    with tab_profiles:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Pressure Head ψ(z)", "Factor of Safety FS(z)"],
                            horizontal_spacing=0.08)

        for i, (name, geo_s) in enumerate(scenarios.items()):
            color = palette[i % len(palette)]
            fs = compute_FS(psi_pred, z_arr, geo_s)

            fig.add_trace(go.Scatter(
                x=psi_pred, y=z_arr, mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=name,
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=fs, y=z_arr, mode="lines", name=name,
                line=dict(color=color, width=2.5),
                legendgroup=name, showlegend=False,
            ), row=1, col=2)

        # FS=1 reference
        fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444", row=1, col=2,
                      annotation_text="FS=1")

        fig.update_xaxes(title_text="ψ (m)", row=1, col=1)
        fig.update_xaxes(title_text="FS", row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)

        fig.update_layout(
            template="plotly_white", height=550,
            margin=dict(l=60, r=20, t=40, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")

        # ψ is the same because the PINN model is fixed
        st.info("💡 **Note:** ψ(z) is the same across scenarios because the PINN model is fixed. "
                "Only FS differs because it depends on geotechnical parameters (c′, φ′, β, γ).")

    # ── Tab 2: Parameter comparison table ────────────────────────────
    with tab_params:
        param_names = ["c_prime", "phi_prime", "beta", "gamma", "Ks", "theta_s",
                       "theta_r", "alpha", "n"]
        param_labels = ["c′ (kPa)", "φ′ (°)", "β (°)", "γ (kN/m³)", "Ks (m/s)",
                        "θs", "θr", "α (1/m)", "n"]

        rows = []
        for label, key in zip(param_labels, param_names):
            row = {"Parameter": label}
            for name, geo_s in scenarios.items():
                val = geo_s.get(key, "—")
                row[name] = f"{val:.2e}" if isinstance(val, float) and abs(val) < 0.001 else f"{val}"
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ── Tab 3: Summary comparison ────────────────────────────────────
    with tab_summary:
        summary_rows = []
        for name, geo_s in scenarios.items():
            fs = compute_FS(psi_pred, z_arr, geo_s)
            summary_rows.append({
                "Scenario": name,
                "Min FS": f"{fs.min():.4f}",
                "Mean FS": f"{np.mean(fs):.3f}",
                "Max FS": f"{np.max(fs):.2f}",
                "% Unstable (FS<1)": f"{np.mean(fs<1)*100:.1f}%",
                "Depth of Min FS": f"{z_arr[np.argmin(fs)]:.1f} m",
                "Status": "🔴 Failure" if fs.min() < 1.0 else
                          "🟡 Marginal" if fs.min() < 1.5 else "🟢 Safe",
            })

        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

        # Bar chart of min FS per scenario
        fig_bar = go.Figure()
        names = [r["Scenario"] for r in summary_rows]
        min_fs = [float(r["Min FS"]) for r in summary_rows]
        colors = ["#22c55e" if v > 1.5 else "#eab308" if v > 1.0 else "#ef4444" for v in min_fs]

        fig_bar.add_trace(go.Bar(
            x=names, y=min_fs, marker_color=colors,
            text=[f"{v:.3f}" for v in min_fs], textposition="auto",
        ))
        fig_bar.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                          annotation_text="FS = 1")
        fig_bar.add_hline(y=1.5, line_dash="dot", line_color="#eab308",
                          annotation_text="FS = 1.5")

        fig_bar.update_layout(
            yaxis_title="Minimum Factor of Safety",
            template="plotly_white", height=350,
            margin=dict(l=60, r=20, t=30, b=100),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_bar, width="stretch")
