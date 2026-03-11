"""
Inference engine for the trained PINN model.
Provides functions to load the model, predict pressure head ψ(z,t),
compute Factor of Safety, and soil hydraulic properties.
"""

import sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# ── Add project root to path so we can import the PINN architecture ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pinn_landslide.components.pinn_architecture import PINN
from src.pinn_landslide.entity.config_entity import PINNArchConfig


# ─────────────────────────────────────────────────────────────────────
# Default constants (match params.yaml shipped with the repo)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_GEO = dict(
    Ks=1.5e-5, theta_s=0.45, theta_r=0.05,
    alpha=0.8, n=1.4, l=0.5,
    c_prime=5.0, phi_prime=28.0,
    gamma=20.0, beta=42.0, gamma_w=9.81,
)

DEFAULT_NORM = dict(t_max=123.0, z_max=55.0, psi_min=-500.0, psi_max=-46.907)

DEFAULT_ARCH = dict(
    n_hidden_layers=7, neurons_per_layer=64,
    activation="tanh", input_dim=2, output_dim=1,
)

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "pinn_model.pt"
DATA_PATH  = PROJECT_ROOT / "artifacts" / "dataset" / "final_data.csv"
BATCH_PATH = PROJECT_ROOT / "artifacts" / "dataset" / "data_batch.pt"


# ─────────────────────────────────────────────────────────────────────
# Van Genuchten helpers (numpy versions for fast UI plotting)
# ─────────────────────────────────────────────────────────────────────

def vg_Se(psi: np.ndarray, alpha: float, n: float) -> np.ndarray:
    """Effective saturation Se(ψ). ψ in metres (negative = unsaturated)."""
    m = 1.0 - 1.0 / n
    psi_safe = np.maximum(np.abs(psi), 1e-8)
    Se = np.where(psi < 0, (1.0 + (alpha * psi_safe) ** n) ** (-m), 1.0)
    return np.clip(Se, 0.0, 1.0)


def vg_theta(psi: np.ndarray, theta_r: float, theta_s: float,
             alpha: float, n: float) -> np.ndarray:
    """Volumetric water content θ(ψ)."""
    Se = vg_Se(psi, alpha, n)
    return theta_r + Se * (theta_s - theta_r)


def vg_K(psi: np.ndarray, Ks: float, alpha: float, n: float,
         l: float = 0.5) -> np.ndarray:
    """Hydraulic conductivity K(ψ) [m/s]."""
    m = 1.0 - 1.0 / n
    Se = np.clip(vg_Se(psi, alpha, n), 1e-8, 1.0 - 1e-6)
    inner_base = np.clip(1.0 - Se ** (1.0 / m), 1e-8, None)
    inner = (1.0 - inner_base ** m) ** 2
    return Ks * (Se ** l) * inner


def vg_C(psi: np.ndarray, theta_r: float, theta_s: float,
         alpha: float, n: float) -> np.ndarray:
    """Specific moisture capacity C(ψ) = dθ/dψ."""
    m = 1.0 - 1.0 / n
    psi_safe = np.maximum(np.abs(psi), 1e-8)
    C = np.where(
        psi < 0,
        (theta_s - theta_r) * m * n * alpha
        * (alpha * psi_safe) ** (n - 1)
        / (1.0 + (alpha * psi_safe) ** n) ** (m + 1),
        0.0,
    )
    return np.clip(C, 1e-8, None)


# ─────────────────────────────────────────────────────────────────────
# Factor of Safety (corrected for unsaturated soil with matric suction)
# ─────────────────────────────────────────────────────────────────────

def compute_FS(psi: np.ndarray, z: np.ndarray, geo: dict) -> np.ndarray:
    """
    Infinite slope FS accounting for matric suction in unsaturated soil.
    Uses the original code's approach: u = max(ψ, 0) * γ_w
    (pore water pressure is zero when unsaturated).
    """
    gamma_w = geo["gamma_w"]
    gamma   = geo["gamma"]
    c_prime = geo["c_prime"]
    phi_rad = math.radians(geo["phi_prime"])
    beta_rad = math.radians(geo["beta"])

    # Pore water pressure (positive only when ψ > 0)
    u = np.maximum(psi, 0.0) * gamma_w

    # Dynamic weight using moisture content
    Se = vg_Se(psi, geo["alpha"], geo["n"])
    theta = geo["theta_r"] + Se * (geo["theta_s"] - geo["theta_r"])
    dynamic_weight = gamma + theta * gamma_w

    sigma = dynamic_weight * z * (math.cos(beta_rad) ** 2)
    tau   = dynamic_weight * z * math.sin(beta_rad) * math.cos(beta_rad)
    effective_stress = np.maximum(sigma - u, 0.0)

    # Add matric suction contribution for unsaturated soil
    # φ^b ≈ φ'/2 is a common approximation (Fredlund & Rahardjo)
    phi_b_rad = phi_rad / 2.0
    matric_suction = np.maximum(-psi, 0.0) * gamma_w  # positive when ψ < 0
    suction_strength = matric_suction * math.tan(phi_b_rad)

    numerator = c_prime + effective_stress * math.tan(phi_rad) + suction_strength
    denominator = np.maximum(tau, 1e-9)

    fs = numerator / denominator
    return np.clip(fs, 0.0, 50.0)  # clip extreme values for display


def compute_FS_original(psi: np.ndarray, z: np.ndarray, geo: dict) -> np.ndarray:
    """Original FS formula from the codebase (without suction correction)."""
    gamma_w  = geo["gamma_w"]
    gamma    = geo["gamma"]
    c_prime  = geo["c_prime"]
    phi_rad  = math.radians(geo["phi_prime"])
    beta_rad = math.radians(geo["beta"])

    u = np.maximum(psi, 0.0) * gamma_w
    sigma = gamma * z * (math.cos(beta_rad) ** 2)
    tau   = gamma * z * math.sin(beta_rad) * math.cos(beta_rad)
    effective_stress = np.maximum(sigma - u, 0.0)
    numerator   = c_prime + effective_stress * math.tan(phi_rad)
    denominator = np.maximum(tau, 1e-9)
    return np.clip(numerator / denominator, 0.0, 50.0)


# ─────────────────────────────────────────────────────────────────────
# Model loader & predictor
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def load_model(model_path: str | Path | None = None,
               arch: dict | None = None) -> PINN:
    """Load trained PINN weights."""
    model_path = Path(model_path or MODEL_PATH)
    arch = arch or DEFAULT_ARCH
    cfg = PINNArchConfig(**arch)
    model = PINN(cfg)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_psi(model: PINN, z: np.ndarray, t: np.ndarray,
                norm: dict | None = None) -> np.ndarray:
    """
    Predict physical pressure head ψ(z, t) from the PINN.

    Parameters
    ----------
    z : depth array (metres, physical)
    t : time array  (days, physical)
    norm : normalisation params dict {z_max, t_max, psi_min, psi_max}

    Returns
    -------
    psi_phys : pressure head in metres (same shape as inputs)
    """
    norm = norm or DEFAULT_NORM
    z_norm = torch.tensor(z / norm["z_max"], dtype=torch.float32).view(-1, 1)
    t_norm = torch.tensor(t / norm["t_max"], dtype=torch.float32).view(-1, 1)
    psi_norm = model(z_norm, t_norm).numpy().flatten()
    psi_range = norm["psi_max"] - norm["psi_min"]
    psi_phys = psi_norm * psi_range + norm["psi_min"]
    return psi_phys


def predict_grid(model: PINN, z_arr: np.ndarray, t_arr: np.ndarray,
                 norm: dict | None = None) -> np.ndarray:
    """Predict ψ on a 2-D meshgrid (z × t). Returns shape (len(z), len(t))."""
    norm = norm or DEFAULT_NORM
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    z_flat = Z.flatten()
    t_flat = T.flatten()
    psi_flat = predict_psi(model, z_flat, t_flat, norm)
    return psi_flat.reshape(Z.shape)


# ─────────────────────────────────────────────────────────────────────
# PDE residual computation (Richards' equation)
# ─────────────────────────────────────────────────────────────────────

def compute_pde_residual(model: PINN, z_phys: np.ndarray, t_phys: np.ndarray,
                         norm: dict | None = None, geo: dict | None = None) -> np.ndarray:
    """
    Compute the Richards' equation residual at given (z, t) points.
    Returns |R(z,t)| = |C(ψ)·∂ψ/∂t − ∂/∂z[K(ψ)(∂ψ/∂z + 1)]|
    """
    norm = norm or DEFAULT_NORM
    geo = geo or DEFAULT_GEO

    z_t = torch.tensor(z_phys / norm["z_max"], dtype=torch.float32).view(-1, 1).requires_grad_(True)
    t_t = torch.tensor(t_phys / norm["t_max"], dtype=torch.float32).view(-1, 1).requires_grad_(True)

    psi_norm = model(z_t, t_t)

    # Gradients in normalised space
    dpsi_dz_n = torch.autograd.grad(psi_norm, z_t, grad_outputs=torch.ones_like(psi_norm),
                                     create_graph=True, retain_graph=True)[0]
    dpsi_dt_n = torch.autograd.grad(psi_norm, t_t, grad_outputs=torch.ones_like(psi_norm),
                                     create_graph=True, retain_graph=True)[0]
    d2psi_dz2_n = torch.autograd.grad(dpsi_dz_n, z_t, grad_outputs=torch.ones_like(dpsi_dz_n),
                                       create_graph=True, retain_graph=True)[0]

    psi_range = norm["psi_max"] - norm["psi_min"]
    psi_phys = (psi_norm * psi_range + norm["psi_min"])
    dpsi_dz = (dpsi_dz_n * psi_range) / norm["z_max"]
    dpsi_dt = (dpsi_dt_n * psi_range) / (norm["t_max"] * 24 * 3600.0)
    d2psi_dz2 = (d2psi_dz2_n * psi_range) / (norm["z_max"] ** 2)

    # Hydraulic properties (torch)
    from src.pinn_landslide.utils.utils import _vg_C, _vg_K
    C_val = _vg_C(psi_phys, geo["theta_r"], geo["theta_s"], geo["alpha"], geo["n"])
    K_val = _vg_K(psi_phys, geo["Ks"], geo["alpha"], geo["n"], geo.get("l", 0.5))

    dK_dz = torch.autograd.grad(K_val, z_t, grad_outputs=torch.ones_like(K_val),
                                 create_graph=False, retain_graph=False)[0] / norm["z_max"]

    LHS = C_val * dpsi_dt
    RHS = dK_dz * (dpsi_dz + 1.0) + K_val * d2psi_dz2
    residual = (LHS - RHS).detach().numpy().flatten()
    return residual


def compute_pde_residual_grid(model: PINN, z_arr: np.ndarray, t_arr: np.ndarray,
                              norm: dict | None = None, geo: dict | None = None) -> np.ndarray:
    """Compute PDE residual on a 2-D meshgrid. Returns shape (len(z), len(t))."""
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    z_flat = Z.flatten()
    t_flat = T.flatten()
    res_flat = compute_pde_residual(model, z_flat, t_flat, norm, geo)
    return res_flat.reshape(Z.shape)


# ─────────────────────────────────────────────────────────────────────
# Uncertainty quantification (parameter perturbation)
# ─────────────────────────────────────────────────────────────────────

def monte_carlo_fs(model: PINN, z_arr: np.ndarray, t_val: float,
                   geo_base: dict, norm: dict, n_samples: int = 100,
                   cov_frac: float = 0.10) -> dict:
    """
    Monte-Carlo uncertainty estimation for FS.
    Perturbs c', φ', β, α, n, Ks by ±cov_frac (coefficient of variation).
    Returns dict with mean, std, p5, p95 FS arrays.
    """
    rng = np.random.default_rng(42)
    perturb_keys = ["c_prime", "phi_prime", "beta", "alpha", "n", "Ks"]
    t_arr = np.full_like(z_arr, t_val)
    psi_pred = predict_psi(model, z_arr, t_arr, norm)

    fs_samples = np.zeros((n_samples, len(z_arr)))
    for i in range(n_samples):
        geo_p = dict(geo_base)
        for key in perturb_keys:
            scale = abs(geo_p[key]) * cov_frac if geo_p[key] != 0 else 0.01
            geo_p[key] = rng.normal(geo_p[key], scale)
        # Clamp to physical ranges
        geo_p["phi_prime"] = np.clip(geo_p["phi_prime"], 5.0, 50.0)
        geo_p["beta"] = np.clip(geo_p["beta"], 1.0, 80.0)
        geo_p["n"] = max(geo_p["n"], 1.01)
        geo_p["alpha"] = max(geo_p["alpha"], 0.01)
        geo_p["Ks"] = max(geo_p["Ks"], 1e-10)
        geo_p["c_prime"] = max(geo_p["c_prime"], 0.0)
        fs_samples[i] = compute_FS(psi_pred, z_arr, geo_p)

    return dict(
        mean=np.mean(fs_samples, axis=0),
        std=np.std(fs_samples, axis=0),
        p5=np.percentile(fs_samples, 5, axis=0),
        p95=np.percentile(fs_samples, 95, axis=0),
        samples=fs_samples,
    )


# ─────────────────────────────────────────────────────────────────────
# Critical slip surface finder
# ─────────────────────────────────────────────────────────────────────

def find_critical_depth(model: PINN, t_arr: np.ndarray,
                        geo: dict, norm: dict,
                        z_res: int = 200) -> dict:
    """
    For each time step, find the depth where FS is minimum.
    Returns dict with arrays: times, critical_depths, min_fs.
    """
    z_arr = np.linspace(0.5, norm["z_max"] * 0.9, z_res)
    critical_depths = np.zeros(len(t_arr))
    min_fs = np.zeros(len(t_arr))

    for i, t_val in enumerate(t_arr):
        t_full = np.full_like(z_arr, t_val)
        psi = predict_psi(model, z_arr, t_full, norm)
        fs = compute_FS(psi, z_arr, geo)
        idx = np.argmin(fs)
        critical_depths[i] = z_arr[idx]
        min_fs[i] = fs[idx]

    return dict(times=t_arr, critical_depths=critical_depths, min_fs=min_fs)


# ─────────────────────────────────────────────────────────────────────
# Hydrology validation metrics
# ─────────────────────────────────────────────────────────────────────

def compute_hydro_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Comprehensive hydrology validation metrics.
    Returns: R2, RMSE, MAE, NSE, KGE, PBIAS, nRMSE
    """
    obs_mean = np.mean(observed)
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - obs_mean) ** 2)

    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    mae = np.mean(np.abs(predicted - observed))

    # Nash-Sutcliffe Efficiency
    nse = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Kling-Gupta Efficiency
    r_corr = np.corrcoef(observed, predicted)[0, 1] if len(observed) > 1 else 0.0
    alpha_kge = np.std(predicted) / max(np.std(observed), 1e-10)
    beta_kge = np.mean(predicted) / max(np.mean(observed), 1e-10)
    kge = 1.0 - np.sqrt((r_corr - 1) ** 2 + (alpha_kge - 1) ** 2 + (beta_kge - 1) ** 2)

    # Percent Bias
    pbias = 100 * np.sum(predicted - observed) / max(np.abs(np.sum(observed)), 1e-10)

    # Normalised RMSE
    nrmse = rmse / max(np.std(observed), 1e-10) * 100

    return dict(R2=r2, RMSE=rmse, MAE=mae, NSE=nse, KGE=kge, PBIAS=pbias, nRMSE=nrmse)
