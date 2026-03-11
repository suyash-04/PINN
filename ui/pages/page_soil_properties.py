"""
Soil Properties Page
=====================
Interactive Van Genuchten curve visualisation.
Shows θ(ψ), K(ψ), C(ψ), Se(ψ) as functions of suction.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _vg_Se(psi, alpha, n):
    """Effective saturation from Van Genuchten."""
    m = 1.0 - 1.0 / n
    h = np.abs(psi)
    return (1 + (alpha * h) ** n) ** (-m)


def _vg_theta(psi, alpha, n, theta_r, theta_s):
    Se = _vg_Se(psi, alpha, n)
    return theta_r + (theta_s - theta_r) * Se


def _vg_K(psi, alpha, n, Ks):
    m = 1.0 - 1.0 / n
    Se = _vg_Se(psi, alpha, n)
    Se = np.clip(Se, 1e-10, 1.0 - 1e-10)
    return Ks * Se ** 0.5 * (1 - (1 - Se ** (1 / m)) ** m) ** 2


def _vg_C(psi, alpha, n, theta_r, theta_s):
    """Specific moisture capacity dθ/dψ."""
    m = 1.0 - 1.0 / n
    h = np.abs(psi)
    ah_n = (alpha * h) ** n
    denom = (1 + ah_n) ** (m + 1)
    return (theta_s - theta_r) * alpha * n * m * (alpha * h) ** (n - 1) / denom


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🌍 Soil Hydraulic Properties")
    st.caption("Interactive Van Genuchten soil-water retention & hydraulic conductivity curves")

    # ── User controls ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Van Genuchten Fine-Tuning")
    alpha = st.sidebar.slider("α (1/m)", 0.01, 5.0, float(geo.get("alpha", 0.8)), 0.01,
                               key="soil_alpha")
    n_vg = st.sidebar.slider("n", 1.01, 5.0, float(geo.get("n", 1.4)), 0.01,
                              key="soil_n")
    theta_s = st.sidebar.slider("θ_s", 0.2, 0.6, float(geo.get("theta_s", 0.45)), 0.01,
                                 key="soil_ts")
    theta_r = st.sidebar.slider("θ_r", 0.0, 0.2, float(geo.get("theta_r", 0.05)), 0.01,
                                 key="soil_tr")
    Ks = st.sidebar.number_input("K_s (m/s)", value=float(geo.get("Ks", 1.5e-5)),
                                  format="%.2e", key="soil_Ks")

    psi_min = st.sidebar.slider("ψ range min (m)", -1000.0, -0.1, -500.0, 1.0, key="soil_pmin")
    n_pts = 500

    # ── Compute curves ───────────────────────────────────────────────
    psi = np.linspace(psi_min, -0.01, n_pts)
    Se = _vg_Se(psi, alpha, n_vg)
    theta = _vg_theta(psi, alpha, n_vg, theta_r, theta_s)
    K = _vg_K(psi, alpha, n_vg, Ks)
    C = _vg_C(psi, alpha, n_vg, theta_r, theta_s)

    common_layout = dict(
        template="plotly_white", height=420,
        margin=dict(l=60, r=20, t=40, b=60),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
    )

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_retention, tab_conduct, tab_capacity, tab_se, tab_combined = st.tabs([
        "💧 θ(ψ)", "🚿 K(ψ)", "📏 C(ψ)", "🔵 Se(ψ)", "📊 Combined"
    ])

    # ── Tab 1: Retention curve θ(ψ) ────────────────────────────────
    with tab_retention:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=psi, y=theta, mode="lines", name="θ(ψ)",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.15)",
        ))
        fig1.add_hline(y=theta_s, line_dash="dash", line_color="#64748b",
                       annotation_text=f"θ_s = {theta_s}")
        fig1.add_hline(y=theta_r, line_dash="dash", line_color="#64748b",
                       annotation_text=f"θ_r = {theta_r}")
        fig1.update_layout(
            xaxis_title="Pressure Head ψ (m)", yaxis_title="Volumetric Water Content θ",
            title="Soil-Water Retention Curve", **common_layout
        )
        st.plotly_chart(fig1, width="stretch")

        st.info(
            "The retention curve shows how much water the soil holds at a given suction. "
            "Near ψ = 0 (saturation), θ → θ_s. As suction increases (more negative ψ), "
            "θ decreases toward the residual θ_r."
        )

    # ── Tab 2: Hydraulic conductivity K(ψ) ──────────────────────────
    with tab_conduct:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=psi, y=K, mode="lines", name="K(ψ)",
            line=dict(color="#818cf8", width=3),
        ))
        fig2.update_layout(
            xaxis_title="Pressure Head ψ (m)",
            yaxis_title="Hydraulic Conductivity K (m/s)",
            yaxis_type="log",
            title="Unsaturated Hydraulic Conductivity", **common_layout
        )
        st.plotly_chart(fig2, width="stretch")

        st.info(
            "K drops dramatically as the soil dries. This plot uses a log scale "
            "to show the orders-of-magnitude variation. At saturation K → K_s."
        )

    # ── Tab 3: Specific moisture capacity C(ψ) ──────────────────────
    with tab_capacity:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=psi, y=C, mode="lines", name="C(ψ)",
            line=dict(color="#f472b6", width=3),
        ))
        fig3.update_layout(
            xaxis_title="Pressure Head ψ (m)",
            yaxis_title="Specific Moisture Capacity C (1/m)",
            title="Specific Moisture Capacity dθ/dψ", **common_layout
        )
        st.plotly_chart(fig3, width="stretch")

        st.info(
            "C(ψ) = dθ/dψ describes how rapidly moisture content changes "
            "with suction. It appears in the Richards' equation and peaks "
            "near the air-entry value."
        )

    # ── Tab 4: Effective saturation Se(ψ) ───────────────────────────
    with tab_se:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=psi, y=Se, mode="lines", name="Se(ψ)",
            line=dict(color="#22c55e", width=3),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.12)",
        ))
        fig4.add_hline(y=1.0, line_dash="dash", line_color="#64748b",
                       annotation_text="Se = 1 (saturated)")
        fig4.update_layout(
            xaxis_title="Pressure Head ψ (m)",
            yaxis_title="Effective Saturation Se",
            title="Effective Saturation", **common_layout
        )
        st.plotly_chart(fig4, width="stretch")

    # ── Tab 5: Combined 2×2 ─────────────────────────────────────────
    with tab_combined:
        fig5 = make_subplots(rows=2, cols=2,
                             subplot_titles=["θ(ψ)", "K(ψ) — log scale",
                                             "C(ψ)", "Se(ψ)"],
                             vertical_spacing=0.12, horizontal_spacing=0.1)
        fig5.add_trace(go.Scatter(x=psi, y=theta, line=dict(color="#38bdf8", width=2),
                                  showlegend=False), row=1, col=1)
        fig5.add_trace(go.Scatter(x=psi, y=K, line=dict(color="#818cf8", width=2),
                                  showlegend=False), row=1, col=2)
        fig5.add_trace(go.Scatter(x=psi, y=C, line=dict(color="#f472b6", width=2),
                                  showlegend=False), row=2, col=1)
        fig5.add_trace(go.Scatter(x=psi, y=Se, line=dict(color="#22c55e", width=2),
                                  showlegend=False), row=2, col=2)
        fig5.update_yaxes(type="log", row=1, col=2)
        fig5.update_layout(height=700, template="plotly_white",
                           margin=dict(l=60, r=20, t=50, b=60),
                           paper_bgcolor="rgba(255,255,255,1)",
                           plot_bgcolor="rgba(241,245,249,1)")
        st.plotly_chart(fig5, width="stretch")

    # ── Key Parameters Summary ───────────────────────────────────────
    st.divider()
    st.markdown("### 📐 Van Genuchten Parameter Summary")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("α (1/m)", f"{alpha:.3f}")
    p2.metric("n", f"{n_vg:.3f}")
    p3.metric("θ_s", f"{theta_s:.3f}")
    p4.metric("θ_r", f"{theta_r:.3f}")
    p5.metric("K_s (m/s)", f"{Ks:.2e}")

    st.markdown(r"""
    **Van Genuchten (1980) equations:**

    $$S_e = \left[1 + (\alpha |h|)^n\right]^{-m}, \quad m = 1 - 1/n$$

    $$\theta(h) = \theta_r + (\theta_s - \theta_r)\,S_e$$

    $$K(h) = K_s\,S_e^{1/2}\left[1 - (1 - S_e^{1/m})^m\right]^2$$
    """)
