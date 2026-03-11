"""
PDE Residual Visualisation Page
================================
Shows WHERE the PINN violates the Richards' equation physics.
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

from model_inference import compute_pde_residual, compute_pde_residual_grid, predict_psi


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🔬 PDE Residual Analysis")
    st.caption("Visualise where the PINN violates the Richards' equation: R(z,t) = C·∂ψ/∂t − ∂/∂z[K(∂ψ/∂z+1)]")

    tab_heatmap, tab_profile, tab_stats = st.tabs([
        "🗺️ Residual Heatmap", "📈 Residual Profiles", "📊 Statistics"
    ])

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        z_res = st.slider("Depth resolution", 10, 60, 30, 5, key="pde_zres")
    with c2:
        t_res = st.slider("Time resolution", 10, 60, 30, 5, key="pde_tres")
    with c3:
        z_max_plot = st.slider("Max depth (m)", 5.0, 55.0, 40.0, 5.0, key="pde_zmax")

    z_arr = np.linspace(0.5, z_max_plot, z_res)
    t_arr = np.linspace(0.5, norm["t_max"] - 0.5, t_res)

    # ── Compute residual grid ────────────────────────────────────────
    with st.spinner("Computing PDE residuals (requires autograd) …"):
        try:
            res_grid = compute_pde_residual_grid(model, z_arr, t_arr, norm, geo)
            abs_res = np.abs(res_grid)
            log_res = np.log10(abs_res + 1e-15)
        except Exception as e:
            st.error(f"Error computing PDE residual: {e}")
            st.info("This computation requires autograd. Ensure the model is in eval mode with gradients enabled.")
            return

    # ── Tab 1: Residual Heatmap ──────────────────────────────────────
    with tab_heatmap:
        scale_choice = st.radio("Colour scale", ["Log₁₀|R|", "Linear |R|"],
                                horizontal=True, key="pde_scale")

        z_data = log_res if "Log" in scale_choice else abs_res
        cb_title = "log₁₀|R|" if "Log" in scale_choice else "|R|"

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=t_arr, y=z_arr, z=z_data,
            colorscale="Inferno",
            colorbar=dict(title=cb_title),
            hovertemplate="Time: %{x:.1f} d<br>Depth: %{y:.1f} m<br>" + cb_title + ": %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Time (days)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),
            template="plotly_white", height=500,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig, width="stretch")

        # Interpretation
        st.info(
            "**Dark regions** = low residual (PINN satisfies physics well). "
            "**Bright regions** = high residual (physics violation). "
            "Large residuals near boundaries or at early times are common with PINNs."
        )

    # ── Tab 2: Residual Profiles ─────────────────────────────────────
    with tab_profile:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### |R| vs Depth (fixed time)")
            sel_times = st.multiselect("Select times (days)",
                                        [float(f"{t:.1f}") for t in t_arr],
                                        default=[float(f"{t_arr[0]:.1f}"),
                                                 float(f"{t_arr[len(t_arr)//2]:.1f}"),
                                                 float(f"{t_arr[-1]:.1f}")],
                                        key="pde_prof_t")

            fig_d = go.Figure()
            palette = ["#38bdf8", "#818cf8", "#f472b6", "#fb923c", "#22c55e"]
            for i, tv in enumerate(sel_times):
                t_idx = np.argmin(np.abs(t_arr - tv))
                fig_d.add_trace(go.Scatter(
                    x=abs_res[:, t_idx], y=z_arr,
                    mode="lines", name=f"t={t_arr[t_idx]:.0f}d",
                    line=dict(color=palette[i % len(palette)], width=2),
                ))
            fig_d.update_layout(
                xaxis_title="|R(z)|", xaxis_type="log",
                yaxis_title="Depth (m)", yaxis=dict(autorange="reversed"),
                template="plotly_white", height=400,
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig_d, width="stretch")

        with col_b:
            st.markdown("#### |R| vs Time (fixed depth)")
            sel_depths = st.multiselect("Select depths (m)",
                                         [float(f"{z:.1f}") for z in z_arr],
                                         default=[float(f"{z_arr[1]:.1f}"),
                                                  float(f"{z_arr[len(z_arr)//2]:.1f}"),
                                                  float(f"{z_arr[-2]:.1f}")],
                                         key="pde_prof_z")

            fig_t = go.Figure()
            for i, zv in enumerate(sel_depths):
                z_idx = np.argmin(np.abs(z_arr - zv))
                fig_t.add_trace(go.Scatter(
                    x=t_arr, y=abs_res[z_idx, :],
                    mode="lines", name=f"z={z_arr[z_idx]:.1f}m",
                    line=dict(color=palette[i % len(palette)], width=2),
                ))
            fig_t.update_layout(
                xaxis_title="Time (days)",
                yaxis_title="|R(t)|", yaxis_type="log",
                template="plotly_white", height=400,
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig_t, width="stretch")

    # ── Tab 3: Statistics ────────────────────────────────────────────
    with tab_stats:
        flat = abs_res.flatten()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean |R|", f"{np.mean(flat):.4e}")
        c2.metric("Median |R|", f"{np.median(flat):.4e}")
        c3.metric("Max |R|", f"{np.max(flat):.4e}")
        c4.metric("Std |R|", f"{np.std(flat):.4e}")
        c5.metric("% < 1e-3", f"{np.mean(flat < 1e-3)*100:.1f}%")

        # Histogram
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=np.log10(flat + 1e-15), nbinsx=50,
            marker_color="#818cf8", name="log₁₀|R|",
        ))
        fig_h.update_layout(
            xaxis_title="log₁₀|R|", yaxis_title="Count",
            template="plotly_white", height=350,
            margin=dict(l=60, r=20, t=30, b=60),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(241,245,249,1)",
        )
        st.plotly_chart(fig_h, width="stretch")

        # Worst-violation locations
        st.divider()
        st.markdown("#### 🎯 Top-10 Worst Violation Points")
        top_idx = np.argsort(flat)[-10:][::-1]
        Z_grid, T_grid = np.meshgrid(z_arr, t_arr, indexing="ij")
        import pandas as pd
        top_df = pd.DataFrame({
            "Depth (m)": Z_grid.flatten()[top_idx],
            "Time (days)": T_grid.flatten()[top_idx],
            "|R|": flat[top_idx],
        })
        st.dataframe(top_df, hide_index=True, width="stretch")
