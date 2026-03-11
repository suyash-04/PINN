"""
Animation Player Page
=====================
Animated ψ(z,t) and FS(z,t) evolution with play/pause.
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

from model_inference import predict_psi, compute_FS


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🎬 Animation Player")
    st.caption("Watch pressure head and slope stability evolve through time")

    available_times = sorted(df_hydrus["Time_days"].unique())

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        z_max_plot = st.slider("Max depth (m)", 5.0, 55.0, 40.0, 5.0, key="anim_zmax")
    with c2:
        z_res = st.slider("Depth points", 30, 200, 80, 10, key="anim_zres")
    with c3:
        mode = st.radio("View", ["ψ + FS Side-by-Side", "ψ Only", "FS Only"],
                        horizontal=True, key="anim_mode")

    z_arr = np.linspace(0.5, z_max_plot, z_res)

    # ── Pre-compute all frames ───────────────────────────────────────
    with st.spinner("Pre-computing frames …"):
        all_psi = {}
        all_fs = {}
        for t_val in available_times:
            t_full = np.full_like(z_arr, t_val)
            psi = predict_psi(model, z_arr, t_full, norm)
            all_psi[t_val] = psi
            all_fs[t_val] = compute_FS(psi, z_arr, geo)

    # Compute global axis ranges
    psi_all = np.concatenate(list(all_psi.values()))
    fs_all = np.concatenate(list(all_fs.values()))
    psi_range = [float(psi_all.min()) - 10, float(psi_all.max()) + 10]
    fs_range = [0, min(float(np.percentile(fs_all, 98)) + 1, 15)]

    # ── Build animation frames ───────────────────────────────────────
    if "Side-by-Side" in mode:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Pressure Head ψ(z)", "Factor of Safety FS(z)"],
                            horizontal_spacing=0.08)

        # Initial traces (t=0)
        t0 = available_times[0]
        fig.add_trace(go.Scatter(x=all_psi[t0], y=z_arr, mode="lines",
                                  line=dict(color="#38bdf8", width=3), name="ψ(z)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=all_fs[t0], y=z_arr, mode="lines",
                                  line=dict(color="#22c55e", width=3), name="FS(z)"), row=1, col=2)
        # FS=1 reference
        fig.add_trace(go.Scatter(x=[1, 1], y=[z_arr[0], z_arr[-1]], mode="lines",
                                  line=dict(color="#ef4444", dash="dash", width=1.5),
                                  showlegend=False), row=1, col=2)

        frames = []
        for t_val in available_times:
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=all_psi[t_val], y=z_arr),
                    go.Scatter(x=all_fs[t_val], y=z_arr),
                    go.Scatter(x=[1, 1], y=[z_arr[0], z_arr[-1]]),
                ],
                name=f"{t_val:.0f}",
                layout=go.Layout(title_text=f"Day {t_val:.0f}"),
            ))

        fig.update_xaxes(range=psi_range, title_text="ψ (m)", row=1, col=1)
        fig.update_xaxes(range=fs_range, title_text="FS", row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)

    elif mode == "ψ Only":
        fig = go.Figure()
        t0 = available_times[0]
        fig.add_trace(go.Scatter(x=all_psi[t0], y=z_arr, mode="lines",
                                  line=dict(color="#38bdf8", width=3), name="ψ(z)"))
        frames = []
        for t_val in available_times:
            frames.append(go.Frame(
                data=[go.Scatter(x=all_psi[t_val], y=z_arr)],
                name=f"{t_val:.0f}",
                layout=go.Layout(title_text=f"Pressure Head — Day {t_val:.0f}"),
            ))
        fig.update_xaxes(range=psi_range, title_text="ψ (m)")
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)")

    else:  # FS Only
        fig = go.Figure()
        t0 = available_times[0]
        fig.add_trace(go.Scatter(x=all_fs[t0], y=z_arr, mode="lines",
                                  line=dict(color="#22c55e", width=3), name="FS(z)"))
        fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444")
        frames = []
        for t_val in available_times:
            frames.append(go.Frame(
                data=[go.Scatter(x=all_fs[t_val], y=z_arr)],
                name=f"{t_val:.0f}",
                layout=go.Layout(title_text=f"Factor of Safety — Day {t_val:.0f}"),
            ))
        fig.update_xaxes(range=fs_range, title_text="FS")
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)")

    fig.frames = frames

    # Slider + play/pause buttons
    sliders = [dict(
        active=0,
        steps=[dict(args=[[f"{t:.0f}"], dict(frame=dict(duration=300, redraw=True), mode="immediate")],
                     method="animate", label=f"{t:.0f}")
               for t in available_times],
        x=0.05, y=0, len=0.9,
        currentvalue=dict(prefix="Day: ", font=dict(color="#1e293b")),
        transition=dict(duration=200),
    )]

    updatemenus = [dict(
        type="buttons", showactive=False,
        x=0.05, y=-0.05, xanchor="right", yanchor="top",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=400, redraw=True),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ],
    )]

    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus,
        template="plotly_white",
        height=550,
        margin=dict(l=60, r=20, t=50, b=80),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(241,245,249,1)",
    )
    st.plotly_chart(fig, width="stretch")

    # ── Snapshot table ───────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Frame Snapshot")
    snap_time = st.select_slider("Select time step (days)", options=available_times,
                                  value=available_times[len(available_times) // 2], key="anim_snap")
    snap_psi = all_psi[snap_time]
    snap_fs = all_fs[snap_time]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min ψ", f"{snap_psi.min():.1f} m")
    c2.metric("Max ψ", f"{snap_psi.max():.1f} m")
    c3.metric("Min FS", f"{snap_fs.min():.3f}")
    unstable_pct = np.mean(snap_fs < 1.0) * 100
    c4.metric("% Unstable (FS<1)", f"{unstable_pct:.1f}%")
