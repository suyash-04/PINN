"""
FastAPI backend for the PINN Landslide Dashboard.
Exposes all inference, VG hydraulic, and analytics endpoints.
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Also add the old Streamlit ui dir so we can reuse model_inference
UI_DIR = PROJECT_ROOT / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from model_inference import (
    load_model, predict_psi, predict_grid,
    compute_FS, compute_FS_original,
    compute_pde_residual_grid,
    compute_hydro_metrics, monte_carlo_fs, find_critical_depth,
    vg_Se, vg_theta, vg_K, vg_C,
    DEFAULT_GEO, DEFAULT_NORM, DEFAULT_ARCH,
    DATA_PATH, MODEL_PATH,
)

from schemas import (
    GeoParams, NormParams, PredictRequest, GridRequest,
    FSRequest, VGRequest, ComparisonRequest,
    PDEResidualRequest, MonteCarloRequest, CriticalSlipRequest,
    SensitivityRequest, ExportRequest,
)

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(title="PINN Landslide API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────────
MODEL = load_model()
DF_HYDRUS = pd.read_csv(DATA_PATH)


# ── Helper ───────────────────────────────────────────────────────────
def _geo(g: GeoParams | None) -> dict:
    if g is None:
        return dict(DEFAULT_GEO)
    return g.model_dump()

def _norm(n: NormParams | None) -> dict:
    if n is None:
        return dict(DEFAULT_NORM)
    return n.model_dump()


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


# ── Defaults / metadata ─────────────────────────────────────────────

@app.get("/api/defaults")
def get_defaults():
    times = sorted(DF_HYDRUS["Time_days"].unique().tolist())
    return {
        "geo": DEFAULT_GEO,
        "norm": DEFAULT_NORM,
        "arch": DEFAULT_ARCH,
        "available_times": times,
        "n_data_points": len(DF_HYDRUS),
        "depth_range": [float(DF_HYDRUS["Depth_m"].min()), float(DF_HYDRUS["Depth_m"].max())],
        "psi_range": [float(DF_HYDRUS["Pressure_Head"].min()), float(DF_HYDRUS["Pressure_Head"].max())],
    }


@app.get("/api/model-info")
def model_info():
    total = sum(p.numel() for p in MODEL.parameters())
    trainable = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    layers = []
    for name, param in MODEL.named_parameters():
        layers.append({
            "name": name,
            "shape": list(param.shape),
            "n_params": param.numel(),
            "requires_grad": param.requires_grad,
        })
    # Weight histograms for first, mid, last weight layers
    weight_layers = [(n, p.detach().cpu().numpy().flatten().tolist())
                     for n, p in MODEL.named_parameters() if "weight" in n]
    hist_data = {}
    if weight_layers:
        for idx in [0, len(weight_layers)//2, len(weight_layers)-1]:
            name, vals = weight_layers[idx]
            hist_data[name] = vals

    return {
        "total_params": total,
        "trainable_params": trainable,
        "n_weight_layers": sum(1 for n, _ in MODEL.named_parameters() if "weight" in n),
        "size_kb": round(total * 4 / 1024, 1),
        "layers": layers,
        "weight_histograms": hist_data,
        "arch": DEFAULT_ARCH,
        "norm": DEFAULT_NORM,
    }


# ── Prediction endpoints ────────────────────────────────────────────

@app.post("/api/predict")
def predict(req: PredictRequest):
    z = np.array(req.z)
    t = np.array(req.t)
    norm = _norm(req.norm)
    psi = predict_psi(MODEL, z, t, norm)
    return {"z": req.z, "t": req.t, "psi": psi.tolist()}


@app.post("/api/predict-grid")
def predict_grid_endpoint(req: GridRequest):
    z_arr = np.linspace(req.z_min, req.z_max, req.z_res)
    t_arr = np.linspace(req.t_min, req.t_max, req.t_res)
    norm = _norm(req.norm)
    psi_grid = predict_grid(MODEL, z_arr, t_arr, norm)
    return {
        "z": z_arr.tolist(),
        "t": t_arr.tolist(),
        "psi": psi_grid.tolist(),  # 2D array [z_res x t_res]
    }


# ── Factor of Safety ────────────────────────────────────────────────

@app.post("/api/factor-of-safety")
def factor_of_safety(req: FSRequest):
    z = np.array(req.z)
    t = np.array(req.t)
    geo = _geo(req.geo)
    norm = _norm(req.norm)
    psi = predict_psi(MODEL, z, t, norm)
    fs = compute_FS(psi, z, geo)
    fs_orig = compute_FS_original(psi, z, geo)
    return {
        "z": req.z, "t": req.t,
        "psi": psi.tolist(), "fs": fs.tolist(), "fs_original": fs_orig.tolist(),
    }


@app.post("/api/fs-grid")
def fs_grid(req: GridRequest):
    z_arr = np.linspace(req.z_min, req.z_max, req.z_res)
    t_arr = np.linspace(req.t_min, req.t_max, req.t_res)
    geo = _geo(req.geo)
    norm = _norm(req.norm)
    psi_grid = predict_grid(MODEL, z_arr, t_arr, norm)
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    fs_grid_val = compute_FS(psi_grid, Z, geo)
    total = fs_grid_val.size
    unstable = int(np.sum(fs_grid_val < 1.0))
    marginal = int(np.sum((fs_grid_val >= 1.0) & (fs_grid_val < 1.5)))
    safe = int(np.sum(fs_grid_val >= 1.5))
    return {
        "z": z_arr.tolist(), "t": t_arr.tolist(),
        "fs": fs_grid_val.tolist(),
        "psi": psi_grid.tolist(),
        "stats": {"total": total, "unstable": unstable, "marginal": marginal, "safe": safe},
    }


# ── Van Genuchten soil properties ────────────────────────────────────

@app.post("/api/soil-properties")
def soil_properties(req: VGRequest):
    psi = np.linspace(req.psi_min, req.psi_max, req.n_points)
    Se = vg_Se(psi, req.alpha, req.n).tolist()
    theta = vg_theta(psi, req.theta_r, req.theta_s, req.alpha, req.n).tolist()
    K = vg_K(psi, req.Ks, req.alpha, req.n, req.l).tolist()
    C_val = vg_C(psi, req.theta_r, req.theta_s, req.alpha, req.n).tolist()
    return {"psi": psi.tolist(), "Se": Se, "theta": theta, "K": K, "C": C_val}


# ── HYDRUS comparison ────────────────────────────────────────────────

@app.post("/api/hydrus-comparison")
def hydrus_comparison(req: ComparisonRequest):
    results = []
    for t_val in req.times:
        hyd = DF_HYDRUS[DF_HYDRUS["Time_days"] == t_val].sort_values("Depth_m")
        if len(hyd) == 0:
            continue
        z_hyd = hyd["Depth_m"].values
        psi_hyd = hyd["Pressure_Head"].values
        t_arr = np.full_like(z_hyd, t_val)
        norm = _norm(req.norm)
        psi_pinn = predict_psi(MODEL, z_hyd, t_arr, norm)
        err = psi_pinn - psi_hyd
        abs_err = np.abs(err)
        ss_res = np.sum(err**2)
        ss_tot = np.sum((psi_hyd - np.mean(psi_hyd))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        results.append({
            "time": t_val,
            "z": z_hyd.tolist(),
            "psi_hydrus": psi_hyd.tolist(),
            "psi_pinn": psi_pinn.tolist(),
            "abs_error": abs_err.tolist(),
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(abs_err)),
        })
    return {"comparisons": results}


# ── PDE residual ─────────────────────────────────────────────────────

@app.post("/api/pde-residual")
def pde_residual(req: PDEResidualRequest):
    z_arr = np.linspace(req.z_min, req.z_max, req.z_res)
    t_arr = np.linspace(req.t_min, req.t_max, req.t_res)
    geo = _geo(req.geo)
    norm = _norm(req.norm)
    res_grid = compute_pde_residual_grid(MODEL, z_arr, t_arr, norm, geo)
    abs_grid = np.abs(res_grid)
    return {
        "z": z_arr.tolist(), "t": t_arr.tolist(),
        "residual": res_grid.tolist(),
        "abs_residual": abs_grid.tolist(),
        "stats": {
            "mean": float(np.mean(abs_grid)),
            "max": float(np.max(abs_grid)),
            "median": float(np.median(abs_grid)),
        }
    }


# ── Monte Carlo uncertainty ──────────────────────────────────────────

@app.post("/api/uncertainty")
def uncertainty(req: MonteCarloRequest):
    z_arr = np.linspace(req.z_min, req.z_max, req.z_res)
    geo = _geo(req.geo)
    norm = _norm(req.norm)
    mc = monte_carlo_fs(MODEL, z_arr, req.time, geo, norm, req.n_samples, req.cov_frac)

    # Deterministic FS for comparison
    t_det = np.full_like(z_arr, req.time)
    psi_det = predict_psi(MODEL, z_arr, t_det, norm)
    fs_det = compute_FS(psi_det, z_arr, geo)

    return {
        "z": z_arr.tolist(),
        "mean": mc["mean"].tolist(),
        "std": mc["std"].tolist(),
        "p5": mc["p5"].tolist(),
        "p95": mc["p95"].tolist(),
        "deterministic": fs_det.tolist(),
    }


# ── Critical slip surface ────────────────────────────────────────────

@app.post("/api/critical-slip")
def critical_slip(req: CriticalSlipRequest):
    t_arr = np.array(req.times)
    geo = _geo(req.geo)
    norm = _norm(req.norm)
    result = find_critical_depth(MODEL, t_arr, geo, norm, z_res=req.z_res)
    return {
        "times": result["times"].tolist(),
        "critical_depths": result["critical_depths"].tolist(),
        "min_fs": result["min_fs"].tolist(),
    }


# ── Sensitivity analysis ─────────────────────────────────────────────

@app.post("/api/sensitivity")
def sensitivity(req: SensitivityRequest):
    z = np.array([req.z])
    t = np.array([req.t])
    norm = _norm(req.norm)
    geo = _geo(req.geo)
    psi = predict_psi(MODEL, z, t, norm)
    fs_base = compute_FS(psi, z, geo)[0]

    params_to_test = {
        "beta": ("β (slope)", 5.0),
        "c_prime": ("c′ (cohesion)", 2.0),
        "phi_prime": ("φ′ (friction)", 3.0),
        "gamma": ("γ (unit wt)", 2.0),
        "alpha": ("α (VG)", 0.3),
        "n": ("n (VG)", 0.2),
    }

    results = []
    for key, (label, delta) in params_to_test.items():
        low = max(geo[key] - delta, 0.01 if key in ("alpha", "n", "Ks") else 0)
        high = geo[key] + delta
        if key == "n":
            low = max(low, 1.01)
        geo_low = {**geo, key: low}
        geo_high = {**geo, key: high}
        fs_low = compute_FS(psi, z, geo_low)[0]
        fs_high = compute_FS(psi, z, geo_high)[0]
        results.append({
            "param": label, "key": key,
            "fs_low": float(fs_low - fs_base),
            "fs_high": float(fs_high - fs_base),
        })

    return {"fs_base": float(fs_base), "sensitivity": results}


# ── Validation metrics ────────────────────────────────────────────────

@app.get("/api/validation")
def validation():
    times = sorted(DF_HYDRUS["Time_days"].unique().tolist())
    norm = dict(DEFAULT_NORM)
    per_time = []
    all_obs, all_pred = [], []
    for t_val in times:
        hyd = DF_HYDRUS[DF_HYDRUS["Time_days"] == t_val].sort_values("Depth_m")
        z_hyd = hyd["Depth_m"].values
        psi_hyd = hyd["Pressure_Head"].values
        t_arr = np.full_like(z_hyd, t_val)
        psi_pinn = predict_psi(MODEL, z_hyd, t_arr, norm)
        metrics = compute_hydro_metrics(psi_hyd, psi_pinn)
        per_time.append({"time": t_val, **{k: round(float(v), 6) for k, v in metrics.items()}})
        all_obs.extend(psi_hyd.tolist())
        all_pred.extend(psi_pinn.tolist())

    overall = compute_hydro_metrics(np.array(all_obs), np.array(all_pred))
    return {
        "overall": {k: round(float(v), 6) for k, v in overall.items()},
        "per_time": per_time,
    }


# ── Export data ───────────────────────────────────────────────────────

@app.post("/api/export")
def export_data(req: ExportRequest):
    z_arr = np.linspace(req.z_min, req.z_max, req.z_res)
    t_arr = np.linspace(req.t_min, req.t_max, req.t_res)
    geo = _geo(req.geo)
    norm = _norm(req.norm)
    psi_grid = predict_grid(MODEL, z_arr, t_arr, norm)
    Z, T = np.meshgrid(z_arr, t_arr, indexing="ij")
    fs_grid_val = compute_FS(psi_grid, Z, geo)

    rows = []
    for i, z in enumerate(z_arr):
        for j, t in enumerate(t_arr):
            rows.append({
                "depth_m": round(float(z), 3),
                "time_days": round(float(t), 3),
                "psi_m": round(float(psi_grid[i, j]), 4),
                "fs": round(float(fs_grid_val[i, j]), 4),
            })
    return {"data": rows, "n_rows": len(rows)}


@app.get("/api/hydrus-data")
def hydrus_data():
    return {
        "columns": DF_HYDRUS.columns.tolist(),
        "data": DF_HYDRUS.to_dict(orient="records"),
    }
