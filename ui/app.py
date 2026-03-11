"""
PINN Landslide Stability Dashboard — Main Application
======================================================
Launch:  cd ui && streamlit run app.py
"""

import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PINN Landslide Stability Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ──────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Import inference engine ──────────────────────────────────────────
from model_inference import (
    load_model, predict_psi, predict_grid, compute_FS, compute_FS_original,
    vg_Se, vg_theta, vg_K, vg_C,
    DEFAULT_GEO, DEFAULT_NORM, DEFAULT_ARCH,
    DATA_PATH, MODEL_PATH,
)

# ── Import pages ─────────────────────────────────────────────────────
from pages import (
    page_overview,
    page_psi_profiles,
    page_fs_contour,
    page_parameter_explorer,
    page_hydrus_comparison,
    page_soil_properties,
    page_model_info,
    # Tier 1
    page_animation,
    page_training_loss,
    page_pde_residual,
    page_export,
    page_error_analysis,
    # Tier 2
    page_uncertainty,
    page_critical_slip,
    page_rainfall_sim,
    page_validation,
    page_scenarios,
)


# ─────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PINN model …")
def get_model():
    return load_model()


@st.cache_data(show_spinner="Loading HYDRUS dataset …")
def get_hydrus_data():
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────
# Sidebar — Navigation + Global Parameters
# ─────────────────────────────────────────────────────────────────────
# ── Sidebar navigation helper ────────────────────────────────────────
NAV_SECTIONS = {
    "Core": [
        ("🏠 Overview", "overview"),
        ("📈 Pressure Head Profiles", "psi"),
        ("🗺️ Factor of Safety Map", "fs"),
        ("🎛️ Parameter Explorer", "params"),
        ("🔬 HYDRUS vs PINN", "hydrus"),
        ("🧪 Soil Property Curves", "soil"),
        ("🧠 Model Architecture", "arch"),
    ],
    "Advanced": [
        ("🎬 Animation Player", "animation"),
        ("📊 Training Loss", "loss"),
        ("🔬 PDE Residual", "pde"),
        ("📈 Error & Convergence", "error"),
        ("🧮 Validation Metrics", "validation"),
    ],
    "Research": [
        ("🎯 Uncertainty (MC)", "uncertainty"),
        ("🗺️ Critical Slip Surface", "slip"),
        ("🌧️ Rainfall Simulator", "rain"),
        ("🔄 Scenario Comparator", "scenario"),
        ("📥 Export & Report", "export"),
    ],
}

# Initialise session state for active page
if "active_page" not in st.session_state:
    st.session_state.active_page = "overview"

with st.sidebar:
    st.markdown("## 🏔️ PINN Dashboard")
    st.caption("Physics-Informed Neural Network for Landslide Stability Analysis")
    st.divider()

    # Render navigation buttons grouped by section
    for section_name, items in NAV_SECTIONS.items():
        st.markdown(f'<div class="sidebar-section">{section_name}</div>', unsafe_allow_html=True)
        for label, key in items:
            is_active = st.session_state.active_page == key
            css_class = "nav-btn-active" if is_active else "nav-btn"
            with st.container():
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.active_page = key
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Global Geotechnical Parameters ───────────────────────────────
    st.markdown("### ⚙️ Geotechnical Parameters")

    with st.expander("**Slope Properties**", expanded=False):
        beta = st.slider("Slope angle β (°)", 10.0, 60.0, DEFAULT_GEO["beta"], 0.5,
                         help="Inclination of the slope surface")
        c_prime = st.slider("Cohesion c′ (kPa)", 0.0, 50.0, DEFAULT_GEO["c_prime"], 0.5,
                            help="Effective cohesion of the soil")
        phi_prime = st.slider("Friction angle φ′ (°)", 10.0, 45.0, DEFAULT_GEO["phi_prime"], 0.5,
                              help="Effective angle of internal friction")
        gamma = st.slider("Unit weight γ (kN/m³)", 14.0, 26.0, DEFAULT_GEO["gamma"], 0.1,
                          help="Bulk unit weight of the soil mass")

    with st.expander("**Hydraulic Properties**", expanded=False):
        Ks = st.number_input("Ks (m/s)", value=DEFAULT_GEO["Ks"], format="%.2e",
                             help="Saturated hydraulic conductivity")
        theta_s = st.slider("θs (saturated)", 0.20, 0.60, DEFAULT_GEO["theta_s"], 0.01)
        theta_r = st.slider("θr (residual)", 0.01, 0.15, DEFAULT_GEO["theta_r"], 0.01)
        alpha_vg = st.slider("α van Genuchten (1/m)", 0.1, 5.0, DEFAULT_GEO["alpha"], 0.1)
        n_vg = st.slider("n van Genuchten", 1.05, 3.0, DEFAULT_GEO["n"], 0.05)

    # Build geo dict from sidebar values
    geo = dict(
        Ks=Ks, theta_s=theta_s, theta_r=theta_r,
        alpha=alpha_vg, n=n_vg, l=DEFAULT_GEO["l"],
        c_prime=c_prime, phi_prime=phi_prime,
        gamma=gamma, beta=beta, gamma_w=DEFAULT_GEO["gamma_w"],
    )

    st.divider()
    st.caption("Built for Minor Project — PINN Landslide Stability")


# ─────────────────────────────────────────────────────────────────────
# Load data & model
# ─────────────────────────────────────────────────────────────────────
model = get_model()
df_hydrus = get_hydrus_data()


# ─────────────────────────────────────────────────────────────────────
# Route to selected page
# ─────────────────────────────────────────────────────────────────────
PAGE_MAP = {
    "overview":    page_overview,
    "psi":         page_psi_profiles,
    "fs":          page_fs_contour,
    "params":      page_parameter_explorer,
    "hydrus":      page_hydrus_comparison,
    "soil":        page_soil_properties,
    "arch":        page_model_info,
    "animation":   page_animation,
    "loss":        page_training_loss,
    "pde":         page_pde_residual,
    "error":       page_error_analysis,
    "validation":  page_validation,
    "uncertainty": page_uncertainty,
    "slip":        page_critical_slip,
    "rain":        page_rainfall_sim,
    "scenario":    page_scenarios,
    "export":      page_export,
}

active = st.session_state.get("active_page", "overview")
if active in PAGE_MAP:
    PAGE_MAP[active].render(model, df_hydrus, geo, DEFAULT_NORM)
else:
    page_overview.render(model, df_hydrus, geo, DEFAULT_NORM)
