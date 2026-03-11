"""
Interactive Parameter Explorer
===============================
Lets the user change geotechnical / hydraulic parameters via sidebar
and see the real-time effect on FS profiles and contour maps.
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

from model_inference import predict_psi, predict_grid, compute_FS, compute_FS_original


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🎛️ Interactive Parameter Explorer")
    st.caption("Adjust geotechnical parameters in the sidebar and observe real-time effects on stability")

    st.info(
        "💡 **Tip:** Change slope angle, cohesion, friction angle, and hydraulic "
        "parameters in the **left sidebar** to see how they affect the Factor of Safety.",
        icon="💡",
    )

    # ── Configuration ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        time_val = st.slider("Time step (days)", 0.0, float(norm["t_max"]), 96.0, 1.0,
                             key="pe_time")
    with c2:
        z_max_plot = st.slider("Max depth (m)", 5.0, 55.0, 40.0, 5.0, key="pe_zmax")

    depths = np.linspace(0.5, z_max_plot, 200)
    t_arr = np.full_like(depths, time_val)
    psi_pred = predict_psi(model, depths, t_arr, norm)

    # ── Compute FS with current and default parameters ───────────────
    fs_current = compute_FS(psi_pred, depths, geo)
    fs_default = compute_FS(psi_pred, depths, {**geo, **dict(
        c_prime=5.0, phi_prime=28.0, gamma=20.0, beta=42.0,
    )})

    # ── Dual Plot: ψ profile + FS profile ────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Pressure Head ψ(z)", "Factor of Safety FS(z)"],
        horizontal_spacing=0.12,
    )

    # ψ profile
    fig.add_trace(go.Scatter(
        x=psi_pred, y=depths, mode="lines", name="ψ (PINN)",
        line=dict(color="#38bdf8", width=2.5),
    ), row=1, col=1)

    # FS profiles — current vs default
    fig.add_trace(go.Scatter(
        x=fs_current, y=depths, mode="lines", name="FS (current params)",
        line=dict(color="#22c55e", width=2.5),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=fs_default, y=depths, mode="lines", name="FS (default params)",
        line=dict(color="#64748b", width=2, dash="dash"),
    ), row=1, col=2)

    # FS = 1 reference
    fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444", row=1, col=2,
                  annotation_text="FS=1")

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="ψ (m)", row=1, col=1)
    fig.update_xaxes(title_text="Factor of Safety", row=1, col=2,
                     range=[0, min(10, float(np.max(fs_current)) + 1)])

    fig.update_layout(
        template="plotly_white", height=550,
        margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.08),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Sensitivity bars ─────────────────────────────────────────────
    st.markdown("### 📊 Parameter Sensitivity (FS at 10m depth)")

    z_probe = 10.0
    t_probe = time_val
    psi_probe = predict_psi(model, np.array([z_probe]), np.array([t_probe]), norm)
    fs_base = compute_FS(psi_probe, np.array([z_probe]), geo)[0]

    params_to_test = {
        "β (slope angle)": ("beta", geo["beta"], geo["beta"] - 5, geo["beta"] + 5),
        "c′ (cohesion)": ("c_prime", geo["c_prime"], max(0, geo["c_prime"] - 2), geo["c_prime"] + 2),
        "φ′ (friction)": ("phi_prime", geo["phi_prime"], geo["phi_prime"] - 3, geo["phi_prime"] + 3),
        "γ (unit weight)": ("gamma", geo["gamma"], geo["gamma"] - 2, geo["gamma"] + 2),
        "α (VG alpha)": ("alpha", geo["alpha"], max(0.1, geo["alpha"] - 0.3), geo["alpha"] + 0.3),
        "n (VG n)": ("n", geo["n"], max(1.05, geo["n"] - 0.2), geo["n"] + 0.2),
    }

    param_names = []
    fs_low = []
    fs_high = []

    for label, (key, base_val, low_val, high_val) in params_to_test.items():
        geo_low = {**geo, key: low_val}
        geo_high = {**geo, key: high_val}
        fs_l = compute_FS(psi_probe, np.array([z_probe]), geo_low)[0]
        fs_h = compute_FS(psi_probe, np.array([z_probe]), geo_high)[0]
        param_names.append(label)
        fs_low.append(fs_l - fs_base)
        fs_high.append(fs_h - fs_base)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Bar(
        y=param_names, x=fs_low, orientation="h",
        name="Decrease param", marker_color="#ef4444",
    ))
    fig_sens.add_trace(go.Bar(
        y=param_names, x=fs_high, orientation="h",
        name="Increase param", marker_color="#22c55e",
    ))

    fig_sens.update_layout(
        barmode="relative",
        xaxis_title="ΔFS from baseline",
        template="plotly_white", height=350,
        margin=dict(l=140, r=20, t=30, b=50),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_sens.add_vline(x=0, line_color="#64748b", line_width=1)

    st.plotly_chart(fig_sens, width="stretch")

    # ── Numerical readout ────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔢 Point Query")

    q1, q2, q3 = st.columns(3)
    with q1:
        qz = st.number_input("Depth (m)", 0.5, 55.0, 5.0, 0.5, key="pe_qz")
    with q2:
        qt = st.number_input("Time (days)", 0.0, 123.0, 96.0, 1.0, key="pe_qt")
    with q3:
        st.markdown("&nbsp;")  # spacer
        query_btn = st.button("🔍 Query", key="pe_query", width="stretch")

    if query_btn:
        psi_q = predict_psi(model, np.array([qz]), np.array([qt]), norm)[0]
        fs_q = compute_FS(np.array([psi_q]), np.array([qz]), geo)[0]

        from model_inference import vg_Se, vg_K, vg_theta
        Se_q = vg_Se(np.array([psi_q]), geo["alpha"], geo["n"])[0]
        K_q = vg_K(np.array([psi_q]), geo["Ks"], geo["alpha"], geo["n"], geo["l"])[0]
        theta_q = vg_theta(np.array([psi_q]), geo["theta_r"], geo["theta_s"],
                           geo["alpha"], geo["n"])[0]

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("ψ (m)", f"{psi_q:.2f}")
        r2.metric("FS", f"{fs_q:.3f}")
        r3.metric("Se", f"{Se_q:.4f}")
        r4.metric("θ", f"{theta_q:.4f}")
        r5.metric("K (m/s)", f"{K_q:.2e}")

        if fs_q >= 1.5:
            r6.metric("Status", "🟢 Safe")
        elif fs_q >= 1.0:
            r6.metric("Status", "🟡 Marginal")
        else:
            r6.metric("Status", "🔴 Unstable")
