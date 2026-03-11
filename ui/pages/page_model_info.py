"""
Model Information Page
======================
Architecture details, training configuration, parameter counts,
network diagram, and training log viewer.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _count_params(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _layer_info(model):
    """Extract per-layer info (name, type, shape, #params)."""
    rows = []
    for name, param in model.named_parameters():
        rows.append({
            "Parameter": name,
            "Shape": str(list(param.shape)),
            "# Params": param.numel(),
            "Requires Grad": param.requires_grad,
        })
    return rows


def _draw_network_diagram(layer_sizes):
    """Draw a schematic neural-network diagram using Plotly."""
    fig = go.Figure()

    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    x_spacing = 1.0
    y_spacing = 0.6

    layer_positions = []

    for li, n_neurons in enumerate(layer_sizes):
        x = li * x_spacing
        display_n = min(n_neurons, 10)
        y_start = -(display_n - 1) * y_spacing / 2
        positions = [(x, y_start + j * y_spacing) for j in range(display_n)]
        layer_positions.append((positions, n_neurons, display_n))

    # Draw edges (connections) for adjacent layers
    for li in range(n_layers - 1):
        pos_a, _, disp_a = layer_positions[li]
        pos_b, _, disp_b = layer_positions[li + 1]
        for xa, ya in pos_a:
            for xb, yb in pos_b:
                fig.add_trace(go.Scatter(
                    x=[xa, xb], y=[ya, yb], mode="lines",
                    line=dict(color="rgba(148,163,184,0.15)", width=0.5),
                    showlegend=False, hoverinfo="skip",
                ))

    # Draw nodes
    colors = ["#38bdf8"] + ["#818cf8"] * (n_layers - 2) + ["#22c55e"]
    for li, (positions, n_actual, n_display) in enumerate(layer_positions):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=20, color=colors[li], line=dict(width=1.5, color="#334155")),
            showlegend=False, hoverinfo="text",
            hovertext=[f"Layer {li}: neuron {j}" for j in range(n_display)],
        ))
        # If truncated show "..."
        if n_actual > n_display:
            cx = positions[0][0]
            cy = positions[-1][1] - y_spacing * 0.6
            fig.add_annotation(x=cx, y=cy, text=f"…({n_actual})", showarrow=False,
                               font=dict(color="#64748b", size=10))

    # Layer labels
    labels = ["Input<br>(t, z)"] + [f"Hidden {i+1}<br>({layer_sizes[i+1]})" for i in range(n_layers - 2)] + ["Output<br>(ψ)"]
    for li, (positions, _, _) in enumerate(layer_positions):
        cx = positions[0][0]
        cy = max(p[1] for p in positions) + y_spacing * 1.2
        fig.add_annotation(x=cx, y=cy, text=labels[li], showarrow=False,
                           font=dict(color="#334155", size=11))

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        template="plotly_white", height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render(model, df_hydrus, geo, norm):
    st.markdown("## 🧠 Model Information")
    st.caption("Architecture details, parameter counts, and training configuration")

    # ── Metrics bar ──────────────────────────────────────────────────
    total, trainable = _count_params(model)
    n_layers = sum(1 for name, _ in model.named_parameters() if "weight" in name)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Parameters", f"{total:,}")
    m2.metric("Trainable Parameters", f"{trainable:,}")
    m3.metric("Weight Layers", n_layers)
    m4.metric("Model Size", f"{total * 4 / 1024:.1f} KB (fp32)")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_arch, tab_params, tab_norm, tab_config = st.tabs([
        "🏗️ Architecture", "📊 Parameters", "📐 Normalization", "⚙️ Configuration"
    ])

    # ── Tab 1: Architecture diagram ──────────────────────────────────
    with tab_arch:
        st.markdown("### Network Architecture Diagram")

        # Infer layer sizes from model
        layer_sizes = [2]  # input: (t, z)
        for name, param in model.named_parameters():
            if "weight" in name:
                layer_sizes.append(param.shape[0])

        fig = _draw_network_diagram(layer_sizes)
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Architecture Summary:**")
        arch_lines = []
        for i, size in enumerate(layer_sizes):
            if i == 0:
                arch_lines.append(f"- **Input Layer**: {size} neurons (t, z)")
            elif i == len(layer_sizes) - 1:
                arch_lines.append(f"- **Output Layer**: {size} neuron (ψ)")
            else:
                arch_lines.append(f"- **Hidden Layer {i}**: {size} neurons + tanh")
        st.markdown("\n".join(arch_lines))

        st.markdown("""
        **Key design choices:**
        - **Activation**: `tanh` — smooth, bounded, ideal for physics-informed learning
        - **Initialisation**: Xavier uniform — matched to tanh's linear region
        - **Optimiser**: Adam (exploration) → L-BFGS (fine-tuning)
        - **Architecture type**: Plain feedforward MLP (no skip connections)
        """)

    # ── Tab 2: Parameter table ───────────────────────────────────────
    with tab_params:
        st.markdown("### Per-Layer Parameter Breakdown")
        rows = _layer_info(model)
        st.dataframe(rows, hide_index=True, width="stretch")

        st.divider()
        st.markdown("### Weight Distribution")

        # Histograms for first, middle, last layers
        weight_params = [(name, param.detach().cpu().numpy().flatten())
                         for name, param in model.named_parameters()
                         if "weight" in name]

        if weight_params:
            n_w = len(weight_params)
            indices = [0, n_w // 2, n_w - 1]
            colors = ["#38bdf8", "#818cf8", "#22c55e"]

            fig_hist = go.Figure()
            for idx, color in zip(indices, colors):
                name, vals = weight_params[idx]
                fig_hist.add_trace(go.Histogram(
                    x=vals, name=name, marker_color=color,
                    opacity=0.7, nbinsx=50,
                ))

            fig_hist.update_layout(
                barmode="overlay",
                xaxis_title="Weight Value",
                yaxis_title="Count",
                template="plotly_white", height=350,
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig_hist, width="stretch")

    # ── Tab 3: Normalization info ────────────────────────────────────
    with tab_norm:
        st.markdown("### Normalization Parameters")
        st.markdown("""
        The PINN operates on **normalized inputs** ∈ [0, 1] and outputs de-normalised
        pressure head in metres.
        """)

        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**Input normalisation**")
            st.code(f"""
t_norm = t / t_max   where  t_max = {norm.get('t_max', '?')}
z_norm = z / z_max   where  z_max = {norm.get('z_max', '?')}
            """, language="python")

        with n2:
            st.markdown("**Output de-normalisation**")
            st.code(f"""
ψ = ψ_raw × (ψ_max − ψ_min) + ψ_min
  where  ψ_min = {norm.get('psi_min', '?')}
         ψ_max = {norm.get('psi_max', '?')}
            """, language="python")

        st.markdown("""
        | Variable | Physical Range | Normalised Range |
        |----------|---------------|------------------|
        | Time t | 0 → {t_max} days | 0 → 1 |
        | Depth z | 0 → {z_max} m | 0 → 1 |
        | Pressure ψ | {psi_min} → {psi_max} m | 0 → 1 (internal) |
        """.format(**{k: norm.get(k, '?') for k in ['t_max', 'z_max', 'psi_min', 'psi_max']}))

    # ── Tab 4: Training & loss configuration ─────────────────────────
    with tab_config:
        st.markdown("### Training Configuration")

        # Try to load params.yaml
        params_path = PROJECT_ROOT / "params.yaml"
        if params_path.exists():
            import yaml
            with open(params_path) as f:
                params = yaml.safe_load(f)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Training Hyper-parameters**")
                train_p = params.get("model_training", {})
                st.json({
                    "Adam learning rate": train_p.get("learning_rate", "?"),
                    "Adam epochs": train_p.get("adam_epochs", "?"),
                    "L-BFGS max iterations": train_p.get("lbfgs_epochs", "?"),
                    "L-BFGS line search": "strong_wolfe",
                })

            with c2:
                st.markdown("**Loss Weights (λ)**")
                loss_w = params.get("loss_weights", {})
                st.json({
                    "λ_physics (PDE)": loss_w.get("lambda_physics", "?"),
                    "λ_anchor (data)": loss_w.get("lambda_anchor", "?"),
                    "λ_boundary (BC)": loss_w.get("lambda_boundary", "?"),
                    "λ_initial (IC)": loss_w.get("lambda_initial", "?"),
                    "λ_failure (FS≤1)": loss_w.get("lambda_failure", "?"),
                })

            st.divider()
            st.markdown("**Architecture Config (params.yaml)**")
            arch_p = params.get("pinn_architecture", {})
            st.json(arch_p)

            st.divider()
            st.markdown("**Geo-mechanical Parameters**")
            geo_p = params.get("geo_params", {})
            st.json(geo_p)
        else:
            st.warning("params.yaml not found — cannot load configuration.")

        # Training log
        st.divider()
        st.markdown("### 📜 Loss Equation")
        st.latex(r"""
        \mathcal{L}_{\text{total}} =
            \lambda_{\text{phys}} \, \mathcal{L}_{\text{PDE}}
          + \lambda_{\text{anchor}} \, \mathcal{L}_{\text{data}}
          + \lambda_{\text{BC}} \, \mathcal{L}_{\text{BC}}
          + \lambda_{\text{IC}} \, \mathcal{L}_{\text{IC}}
          + \lambda_{\text{fail}} \, \mathcal{L}_{\text{FS}}
        """)
        st.markdown("""
        | Loss Term | Description |
        |-----------|-------------|
        | $\\mathcal{L}_{\\text{PDE}}$ | Richards' equation residual at collocation points |
        | $\\mathcal{L}_{\\text{data}}$ | MSE between PINN ψ and HYDRUS ψ at anchor points |
        | $\\mathcal{L}_{\\text{BC}}$ | Boundary condition enforcement *(currently placeholder)* |
        | $\\mathcal{L}_{\\text{IC}}$ | Initial condition enforcement *(currently placeholder)* |
        | $\\mathcal{L}_{\\text{FS}}$ | Penalty for FS ≤ 1 at failure points |
        """)
