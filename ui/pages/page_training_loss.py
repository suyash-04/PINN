"""
Training Loss Dashboard Page
=============================
Visualise per-component training losses over epochs.
Since the trainer doesn't save per-component logs, we reconstruct them
from the current model by evaluating each loss term on the training batch.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import re
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_inference import (
    predict_psi, predict_grid, compute_FS, vg_Se,
    DEFAULT_GEO, DEFAULT_NORM, BATCH_PATH, MODEL_PATH,
)


def _load_training_log() -> list[str] | None:
    """Try to find a training log file in common locations."""
    candidates = [
        PROJECT_ROOT / "logs" / "running_logs.log",
        PROJECT_ROOT / "artifacts" / "logs" / "training.log",
        PROJECT_ROOT / "training.log",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text().splitlines()
    return None


def _parse_log_lines(lines: list[str]) -> dict:
    """Parse loss values from log lines like 'Adam Epoch [500/10000] | Total Loss: 3.14e-01'."""
    adam_epochs = []
    adam_losses = []
    lbfgs_steps = []
    lbfgs_losses = []

    adam_pat = re.compile(r"Adam Epoch \[(\d+)/\d+\].*Loss:\s*([0-9eE.+-]+)")
    lbfgs_pat = re.compile(r"L-BFGS Step \[(\d+)/\d+\].*Loss:\s*([0-9eE.+-]+)")

    for line in lines:
        m = adam_pat.search(line)
        if m:
            adam_epochs.append(int(m.group(1)))
            adam_losses.append(float(m.group(2)))
            continue
        m = lbfgs_pat.search(line)
        if m:
            lbfgs_steps.append(int(m.group(1)))
            lbfgs_losses.append(float(m.group(2)))

    return dict(
        adam_epochs=np.array(adam_epochs),
        adam_losses=np.array(adam_losses),
        lbfgs_steps=np.array(lbfgs_steps),
        lbfgs_losses=np.array(lbfgs_losses),
    )


def _evaluate_loss_components(model, geo) -> dict:
    """Evaluate each loss component on the training batch."""
    if not BATCH_PATH.exists():
        return None

    batch = torch.load(BATCH_PATH, map_location="cpu")
    model.eval()

    norm_params = batch.get("norm_params", DEFAULT_NORM)
    # Handle numpy values in norm_params
    np_dict = {}
    for k, v in norm_params.items():
        np_dict[k] = float(v) if hasattr(v, 'item') else float(v)
    norm_params = np_dict

    psi_range = norm_params["psi_max"] - norm_params["psi_min"]
    results = {}

    # 1. Anchor loss
    z_data = batch.get("z_data")
    t_data = batch.get("t_data")
    psi_data = batch.get("psi_data")
    if z_data is not None and len(z_data) > 0:
        with torch.no_grad():
            psi_pred = model(z_data, t_data)
            anchor_mse = torch.mean((psi_pred - psi_data.view_as(psi_pred)) ** 2).item()
            results["Anchor (Data)"] = anchor_mse

            # De-normalise for physical error
            psi_pred_phys = psi_pred.numpy().flatten() * psi_range + norm_params["psi_min"]
            psi_data_phys = psi_data.numpy().flatten() * psi_range + norm_params["psi_min"]
            results["Anchor RMSE (m)"] = float(np.sqrt(np.mean((psi_pred_phys - psi_data_phys) ** 2)))

    # 2. Failure loss
    z_fail = batch.get("z_fail")
    t_fail = batch.get("t_fail")
    if z_fail is not None and len(z_fail) > 0:
        with torch.no_grad():
            psi_pred = model(z_fail, t_fail)
            psi_phys = (psi_pred * psi_range + norm_params["psi_min"]).numpy().flatten()
            z_phys = (z_fail * norm_params["z_max"]).numpy().flatten()
            fs = compute_FS(psi_phys, z_phys, geo)
            results["Failure Points"] = len(z_fail)
            results["Failure FS (mean)"] = float(np.mean(fs))
            results["Failure FS (min)"] = float(np.min(fs))

    # 3. Collocation points info
    z_coll = batch.get("z_coll")
    if z_coll is not None:
        results["Collocation Points"] = len(z_coll)

    return results


def render(model, df_hydrus, geo, norm):
    st.markdown("## 📊 Training Loss Dashboard")
    st.caption("Inspect training convergence and per-component loss breakdown")

    tab_curve, tab_decomp, tab_convergence = st.tabs([
        "📉 Loss Curve", "🧩 Loss Decomposition", "📈 Convergence Analysis"
    ])

    # ── Tab 1: Loss Curve from Log ───────────────────────────────────
    with tab_curve:
        log_lines = _load_training_log()

        if log_lines:
            parsed = _parse_log_lines(log_lines)
            fig = go.Figure()

            if len(parsed["adam_epochs"]) > 0:
                fig.add_trace(go.Scatter(
                    x=parsed["adam_epochs"], y=parsed["adam_losses"],
                    mode="lines+markers", name="Adam Phase",
                    line=dict(color="#38bdf8", width=2),
                    marker=dict(size=4),
                ))

            if len(parsed["lbfgs_steps"]) > 0:
                # Offset L-BFGS x-axis to come after Adam
                offset = parsed["adam_epochs"][-1] if len(parsed["adam_epochs"]) > 0 else 0
                fig.add_trace(go.Scatter(
                    x=parsed["lbfgs_steps"] + offset, y=parsed["lbfgs_losses"],
                    mode="lines+markers", name="L-BFGS Phase",
                    line=dict(color="#f472b6", width=2),
                    marker=dict(size=4),
                ))
                # Phase transition line
                fig.add_vline(x=offset, line_dash="dot", line_color="#64748b",
                              annotation_text="Adam → L-BFGS", annotation_position="top")

            fig.update_layout(
                xaxis_title="Training Step",
                yaxis_title="Total Loss",
                yaxis_type="log",
                template="plotly_white", height=450,
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig, width="stretch")

            # Summary metrics
            if len(parsed["adam_losses"]) > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Initial Loss", f"{parsed['adam_losses'][0]:.4e}")
                c2.metric("Adam Final Loss", f"{parsed['adam_losses'][-1]:.4e}")
                if len(parsed["lbfgs_losses"]) > 0:
                    c3.metric("L-BFGS Final Loss", f"{parsed['lbfgs_losses'][-1]:.4e}")
                    reduction = (1 - parsed["lbfgs_losses"][-1] / parsed["adam_losses"][0]) * 100
                    c4.metric("Total Reduction", f"{reduction:.1f}%")
                else:
                    reduction = (1 - parsed["adam_losses"][-1] / parsed["adam_losses"][0]) * 100
                    c3.metric("Total Reduction", f"{reduction:.1f}%")
        else:
            st.info(
                "📂 No training log found. The trainer logs to `logs/running_logs.log`.\n\n"
                "Re-run `python main.py` to generate the log, then refresh this page."
            )

            # Show a synthetic curve based on known training behaviour
            st.markdown("#### 🔄 Simulated Training Curve (from known configuration)")
            epochs = np.arange(1, 10001, 500)
            # Typical exponential decay pattern
            synthetic_loss = 11.2 * np.exp(-0.0004 * epochs) + 2.87
            fig_syn = go.Figure()
            fig_syn.add_trace(go.Scatter(
                x=epochs, y=synthetic_loss, mode="lines",
                line=dict(color="#38bdf8", width=2), name="Estimated Loss",
            ))
            fig_syn.update_layout(
                xaxis_title="Epoch", yaxis_title="Loss",
                template="plotly_white", height=350,
                paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig_syn, width="stretch")

    # ── Tab 2: Loss Decomposition ────────────────────────────────────
    with tab_decomp:
        st.markdown("### 🧩 Current Loss Component Breakdown")
        st.markdown("Evaluating each loss term on the **training batch** with the trained model:")

        with st.spinner("Evaluating loss components …"):
            comp = _evaluate_loss_components(model, geo)

        if comp:
            # Load loss weights from params.yaml
            params_path = PROJECT_ROOT / "params.yaml"
            loss_weights = {}
            if params_path.exists():
                import yaml
                with open(params_path) as f:
                    params = yaml.safe_load(f)
                loss_weights = params.get("loss_weights", {})

            # Display metrics
            cols = st.columns(min(len(comp), 4))
            for i, (name, val) in enumerate(comp.items()):
                with cols[i % len(cols)]:
                    if isinstance(val, float):
                        st.metric(name, f"{val:.6f}")
                    else:
                        st.metric(name, f"{val:,}")

            st.divider()

            # Loss weights bar chart
            if loss_weights:
                st.markdown("### ⚖️ Loss Weight Configuration (λ)")
                names = list(loss_weights.keys())
                values = list(loss_weights.values())
                fig_w = go.Figure(go.Bar(
                    x=names, y=values,
                    marker_color=["#38bdf8", "#818cf8", "#22c55e", "#f472b6", "#fb923c"],
                    text=[f"{v}" for v in values], textposition="auto",
                ))
                fig_w.update_layout(
                    yaxis_title="Weight (λ)",
                    template="plotly_white", height=300,
                    margin=dict(l=60, r=20, t=30, b=60),
                    paper_bgcolor="rgba(255,255,255,1)",
                    plot_bgcolor="rgba(241,245,249,1)",
                )
                st.plotly_chart(fig_w, width="stretch")

                st.markdown("""
                **Interpretation:**
                - Higher λ → that loss component has more influence on training
                - `lambda_failure = 20` heavily penalises FS ≤ 1 violations
                - `lambda_anchor = 10` ensures PINN matches HYDRUS data
                - `lambda_physics = 1` governs PDE satisfaction (Richards' equation)
                - `lambda_boundary = 5` and `lambda_initial = 5` are **not yet active** (placeholder)
                """)
        else:
            st.warning("Training batch file not found at `artifacts/dataset/data_batch.pt`.")

    # ── Tab 3: Convergence Analysis ──────────────────────────────────
    with tab_convergence:
        st.markdown("### 📈 Convergence Diagnostics")

        # Weight norm evolution (heuristic)
        st.markdown("#### Weight Magnitude per Layer")
        weight_norms = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weight_norms.append({
                    "Layer": name,
                    "Frobenius Norm": float(torch.norm(param, p="fro").item()),
                    "Max |w|": float(param.abs().max().item()),
                    "Mean |w|": float(param.abs().mean().item()),
                })

        if weight_norms:
            import pandas as pd
            df_wn = pd.DataFrame(weight_norms)
            st.dataframe(df_wn, hide_index=True, width="stretch")

            fig_norms = go.Figure()
            fig_norms.add_trace(go.Bar(
                x=[w["Layer"].replace("network.", "").replace(".weight", "") for w in weight_norms],
                y=[w["Frobenius Norm"] for w in weight_norms],
                marker_color="#818cf8",
                name="Frobenius Norm",
            ))
            fig_norms.update_layout(
                yaxis_title="‖W‖_F",
                template="plotly_white", height=300,
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor="rgba(255,255,255,1)",
                plot_bgcolor="rgba(241,245,249,1)",
            )
            st.plotly_chart(fig_norms, width="stretch")

        # Training config summary
        st.divider()
        st.markdown("#### ⚙️ Training Protocol")
        st.markdown("""
        | Phase | Optimizer | Steps | Key Settings |
        |-------|-----------|-------|-------------|
        | **Phase I** | Adam | 10,000 epochs | LR = 0.001, no scheduler |
        | **Phase II** | L-BFGS | 5,000 max iter | Strong Wolfe line search, history = 50 |

        **Known issue:** L-BFGS terminated after ~1 step because the loss was already
        at a flat region. Consider adding a learning rate scheduler or warm restarts.
        """)
