"""
PINN Landslide Predictor
========================
Loads a trained PINN model and, given:
  - a soil column depth profile (z values in metres)
  - hourly time steps
  - hourly rainfall data (mm/hr)

predicts pressure head ψ(z, t), computes the Factor of Safety (FS)
at every (z, t) point, and reports when / where the slope first fails.

Usage
-----
    predictor = SlopeFailurePredictor(model_path="artifacts/model.pth")
    results   = predictor.predict(
                    z_values      = np.linspace(0.1, 3.0, 30),   # metres
                    rainfall_mm_hr= np.array([...]),              # length = n_hours
                )
    predictor.print_report(results)
    predictor.plot_results(results)
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from src.pinn_landslide.utils.utils import create_directories, read_yaml, _vg_Se, _vg_K
from src.pinn_landslide.entity.config_entity import PINNArchConfig as ArchConfig, GeoParamsConfig as GeoParams, TrainingConfig as TrainConfig
from src.pinn_landslide.components.pinn_architecture import PINN
from src.pinn_landslide.config.configuration import ConfigurationManager
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Minimal inline definitions so this file is self-contained.
# Replace the imports below with your real package imports if preferred.
# ---------------------------------------------------------------------------




# ── Minimal config dataclasses ───────────────────────────────────────────────



#

# ── Rainfall → boundary-flux conversion ──────────────────────────────────────
def rainfall_to_flux(rainfall_mm_hr: np.ndarray, Ks_m_s: float) -> np.ndarray:
    """
    Convert rainfall intensity (mm/hr) to infiltration flux (m/s).
    Infiltration is capped at Ks (excess becomes runoff — Green-Ampt style).
    Returns negative values (downward = positive z direction in Richards eq).
    """
    flux_m_s = rainfall_mm_hr / 1000.0 / 3600.0   # mm/hr → m/s
    return -np.minimum(flux_m_s, Ks_m_s)           # negative = infiltrating


# ============================================================================
# Main predictor class
# ============================================================================

class SlopeFailurePredictor:
    """
    End-to-end predictor for slope stability using a trained PINN.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved ``model.state_dict()`` file.
    arch_config : ArchConfig, optional
        Network architecture. Defaults match the training defaults.
    geo_params : GeoParams, optional
        Geotechnical parameters. Defaults match the training defaults.
    device : torch.device, optional
        CPU or CUDA device.
    """

    def __init__(
        self,
        model_path:  str | Path,
        arch_config: ArchConfig  = None,
        geo_params:  GeoParams   = None,
        device:      torch.device = None,
    ):
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.geo       =geo_params
        self.arch      = arch_config
        self.beta_rad  = math.radians(self.geo.beta)
        self.phi_rad   = math.radians(self.geo.phi_prime)

        # Load model
        self.model = PINN(self.arch).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[Predictor] Model loaded from {model_path}  |  device: {self.device}")

    # ------------------------------------------------------------------ #
    #  Core prediction                                                     #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        z_values:       np.ndarray,
        rainfall_mm_hr: np.ndarray,
        t_start_hr:     float = 0.0,
        norm_params:    Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Predict ψ, θ, K, FS for every (z, t) combination.

        Parameters
        ----------
        z_values : 1-D array of depths in metres (e.g. np.linspace(0.1, 3.0, 30))
        rainfall_mm_hr : 1-D array of hourly rainfall intensities (mm/hr)
        t_start_hr : starting hour offset (default 0)
        norm_params : dict with keys psi_min, psi_max, z_max, t_max.
                      If None, sensible defaults are inferred automatically.

        Returns
        -------
        dict with keys:
            times_hr, z_values, psi, K, theta, FS,
            rainfall_mm_hr, infiltration_flux,
            failure_detected, first_failure_time_hr, first_failure_depth_m,
            min_FS_series (minimum FS over depth at each hour)
        """
        n_hours = len(rainfall_mm_hr)
        times_hr = t_start_hr + np.arange(n_hours, dtype=np.float64)

        # ── Normalisation parameters ─────────────────────────────────────
        z_max = float(z_values.max())
        t_max = float(times_hr.max()) / 24.0        # keep in days internally

        if norm_params is None:
            # Heuristic defaults — ideally loaded from your saved batch dict
            psi_min = -10.0   # metres (negative = suction)
            psi_max =  2.0    # metres (positive = ponding)
        else:
            psi_min = norm_params["psi_min"]
            psi_max = norm_params["psi_max"]
            z_max   = norm_params.get("z_max", z_max)
            t_max   = norm_params.get("t_max", t_max)

        psi_range = psi_max - psi_min

        # ── Build mesh of all (z, t) pairs ───────────────────────────────
        ZZ, TT = np.meshgrid(z_values, times_hr / 24.0)   # TT in days
        Z_flat = ZZ.ravel()
        T_flat = TT.ravel()

        z_norm = torch.tensor(Z_flat / z_max, dtype=torch.float32, device=self.device).view(-1, 1)
        t_norm = torch.tensor(T_flat / t_max, dtype=torch.float32, device=self.device).view(-1, 1)

        # ── Forward pass (no gradient needed for FS calculation) ─────────
        with torch.no_grad():
            psi_norm = self.model(z_norm, t_norm)

        psi_phys = (psi_norm * psi_range + psi_min).cpu().numpy().reshape(n_hours, len(z_values))

        # ── Hydraulic properties ─────────────────────────────────────────
        psi_t = torch.tensor(psi_phys.ravel(), dtype=torch.float32, device=self.device)
        K_t   = _vg_K(psi_t, self.geo.Ks, self.geo.alpha, self.geo.n, self.geo.l)
        Se_t  = _vg_Se(psi_t, self.geo.alpha, self.geo.n)

        K_phys  = K_t.cpu().numpy().reshape(n_hours, len(z_values))
        Se_phys = Se_t.cpu().numpy().reshape(n_hours, len(z_values))
        theta   = self.geo.theta_r + Se_phys * (self.geo.theta_s - self.geo.theta_r)

        # ── Factor of Safety ─────────────────────────────────────────────
        z_grid = np.broadcast_to(z_values[np.newaxis, :], (n_hours, len(z_values)))
        u      = np.maximum(psi_phys, 0.0) * self.geo.gamma_w

        dynamic_weight = self.geo.gamma + theta * self.geo.gamma_w
        sigma          = dynamic_weight * z_grid * math.cos(self.beta_rad) ** 2
        tau            = dynamic_weight * z_grid * math.sin(self.beta_rad) * math.cos(self.beta_rad)
        eff_stress     = np.maximum(sigma - u, 0.0)

        FS = (self.geo.c_prime + eff_stress * math.tan(self.phi_rad)) / np.maximum(tau, 1e-9)

        # ── Rainfall → flux ──────────────────────────────────────────────
        infiltration_flux = rainfall_to_flux(rainfall_mm_hr, self.geo.Ks)

        # ── Failure detection ────────────────────────────────────────────
        min_FS_series = FS.min(axis=1)          # shape (n_hours,)
        failure_mask  = min_FS_series <= 1.0

        failure_detected       = bool(failure_mask.any())
        first_failure_time_hr  = float(times_hr[failure_mask][0])  if failure_detected else None
        first_failure_depth_m  = float(z_values[FS[failure_mask][0].argmin()]) if failure_detected else None

        return dict(
            times_hr               = times_hr,
            z_values               = z_values,
            psi                    = psi_phys,
            K                      = K_phys,
            theta                  = theta,
            FS                     = FS,
            rainfall_mm_hr         = rainfall_mm_hr,
            infiltration_flux      = infiltration_flux,
            failure_detected       = failure_detected,
            first_failure_time_hr  = first_failure_time_hr,
            first_failure_depth_m  = first_failure_depth_m,
            min_FS_series          = min_FS_series,
        )

    # ------------------------------------------------------------------ #
    #  Text report                                                         #
    # ------------------------------------------------------------------ #

    def print_report(self, results: Dict[str, Any]) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("  PINN SLOPE FAILURE PREDICTION — SUMMARY REPORT")
        print(sep)
        print(f"  Simulation duration : {results['times_hr'][-1]:.0f} hrs")
        print(f"  Depth range         : {results['z_values'][0]:.2f} – {results['z_values'][-1]:.2f} m")
        print(f"  Total rainfall      : {results['rainfall_mm_hr'].sum():.1f} mm")
        print(f"  Peak rainfall       : {results['rainfall_mm_hr'].max():.1f} mm/hr")
        print()

        if results["failure_detected"]:
            print(f"  ⚠️  FAILURE DETECTED")
            print(f"  First failure time  : {results['first_failure_time_hr']:.1f} hrs")
            print(f"  Critical depth      : {results['first_failure_depth_m']:.2f} m")
            print(f"  Min FS at failure   : {results['min_FS_series'].min():.4f}")
        else:
            print(f"  ✅  Slope remains STABLE throughout simulation")
            print(f"  Minimum FS observed : {results['min_FS_series'].min():.4f}")

        print()
        print("  Hourly FS Summary (minimum over all depths):")
        print(f"  {'Hour':>6}  {'Rain(mm/hr)':>12}  {'Min FS':>10}  {'Status':>10}")
        print("  " + "-" * 46)
        for i, (hr, rain, fs) in enumerate(
            zip(results["times_hr"], results["rainfall_mm_hr"], results["min_FS_series"])
        ):
            status = "⚠️ FAIL" if fs <= 1.0 else "OK"
            print(f"  {hr:>6.0f}  {rain:>12.2f}  {fs:>10.4f}  {status:>10}")
        print(sep + "\n")

    # ------------------------------------------------------------------ #
    #  Export to CSV                                                       #
    # ------------------------------------------------------------------ #

    def to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Flatten results into a tidy DataFrame with columns:
        time_hr, z_m, psi_m, K_m_s, theta, FS, rainfall_mm_hr
        """
        rows = []
        times = results["times_hr"]
        z     = results["z_values"]
        rain  = results["rainfall_mm_hr"]

        for i, t in enumerate(times):
            for j, depth in enumerate(z):
                rows.append({
                    "time_hr"       : t,
                    "z_m"           : depth,
                    "psi_m"         : results["psi"][i, j],
                    "K_m_s"         : results["K"][i, j],
                    "theta"         : results["theta"][i, j],
                    "FS"            : results["FS"][i, j],
                    "rainfall_mm_hr": rain[i],
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #

    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Four-panel summary figure:
          1. Hourly rainfall bar chart
          2. Minimum FS over time  (with failure threshold line)
          3. FS heat-map  (depth × time)
          4. Pressure-head heat-map  (depth × time)
        """
        times   = results["times_hr"]
        z       = results["z_values"]
        FS      = results["FS"]
        psi     = results["psi"]
        rain    = results["rainfall_mm_hr"]
        min_fs  = results["min_FS_series"]

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), constrained_layout=True)
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            ax.xaxis.label.set_color("#c9d1d9")
            ax.yaxis.label.set_color("#c9d1d9")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        title_kw = dict(color="#e6edf3", fontsize=11, fontweight="bold", pad=6)

        # ── 1. Rainfall ───────────────────────────────────────────────
        ax = axes[0]
        ax.bar(times, rain, width=0.8, color="#388bfd", alpha=0.85)
        ax.set_ylabel("Rainfall (mm/hr)")
        ax.set_title("Hourly Rainfall Input", **title_kw)
        if results["failure_detected"]:
            ax.axvline(results["first_failure_time_hr"], color="#f85149", ls="--", lw=1.4, label="Failure time")
            ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)

        # ── 2. Min FS series ──────────────────────────────────────────
        ax = axes[1]
        colors = np.where(min_fs <= 1.0, "#f85149", "#3fb950")
        ax.bar(times, min_fs, width=0.8, color=colors, alpha=0.85)
        ax.axhline(1.0, color="#e3b341", ls="--", lw=1.5, label="FS = 1.0 (failure)")
        ax.set_ylabel("Minimum FS (all depths)")
        ax.set_title("Factor of Safety — Minimum Over Depth Profile", **title_kw)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        if results["failure_detected"]:
            ax.axvline(results["first_failure_time_hr"], color="#f85149", ls=":", lw=1.2)

        # ── 3. FS heat-map ────────────────────────────────────────────
        ax = axes[2]
        FS_clipped = np.clip(FS, 0.0, 3.0)
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
        im = ax.pcolormesh(
            times, z, FS_clipped.T,
            cmap="RdYlGn", norm=norm, shading="auto"
        )
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(colors="#8b949e")
        cb.set_label("FS", color="#c9d1d9")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Factor of Safety  (depth × time heatmap)", **title_kw)
        if results["failure_detected"]:
            ax.axvline(results["first_failure_time_hr"], color="#f85149", ls="--", lw=1.4, label="First failure")
            ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)

        # ── 4. Pressure-head heat-map ─────────────────────────────────
        ax = axes[3]
        vm = max(abs(psi.min()), abs(psi.max()))
        im2 = ax.pcolormesh(
            times, z, psi.T,
            cmap="coolwarm", vmin=-vm, vmax=vm, shading="auto"
        )
        cb2 = fig.colorbar(im2, ax=ax, pad=0.01)
        cb2.ax.tick_params(colors="#8b949e")
        cb2.set_label("ψ (m)", color="#c9d1d9")
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Time (hr)")
        ax.set_title("Pressure Head  ψ(z, t)  (depth × time heatmap)", **title_kw)

        fig.suptitle(
            "PINN Slope Stability Prediction",
            color="#e6edf3", fontsize=14, fontweight="bold", y=1.01
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[Predictor] Figure saved → {save_path}")
        else:
            plt.show()


# ============================================================================
# Example / quick-start usage
# ============================================================================

def run_demo():
    """
    Demonstrates the predictor with synthetic rainfall input.
    Replace model_path and parameters with your actual values.
    """
    configuration_manager = ConfigurationManager()
    # ── Step 1: Initialise predictor ──────────────────────────────────────
    predictor = SlopeFailurePredictor(
        model_path  = "artifacts/model/pinn_model.pt",  # ← change this
        arch_config = configuration_manager.get_pinn_arch_config(),  # ← change this if you used non-default architecture
        geo_params  = configuration_manager.get_geo_params_config()
    )

    # ── Step 2: Define inputs ─────────────────────────────────────────────
    # Randomly sampled depth points in the soil column (metres)
    np.random.seed(42)
    z_values = np.sort(np.random.uniform(0.1, 3.0, 40))   # 40 random depths

    # Synthetic hourly rainfall — 72-hour event with a heavy burst at hour 36-48
    n_hours = 72
    rainfall = np.zeros(n_hours)
    rainfall[10:20]  = np.random.uniform(2,  8,  10)   # light rain
    rainfall[36:48]  = np.random.uniform(20, 40, 12)   # heavy burst
    rainfall[55:65]  = np.random.uniform(5,  15, 10)   # moderate rain
    # Add slight noise
    rainfall += np.random.uniform(0, 0.5, n_hours)

    # Optional: if you saved norm_params with your batch, load them here
    # norm_params = torch.load("artifacts/data_ingestion/batch.pt")["norm_params"]
    norm_params = None  # will use heuristic defaults

    # ── Step 3: Run prediction ────────────────────────────────────────────
    results = predictor.predict(
        z_values       = z_values,
        rainfall_mm_hr = rainfall,
        norm_params    = norm_params,
    )
    create_directories(["artifacts/results/"])

    # ── Step 4: Report ────────────────────────────────────────────────────
    predictor.print_report(results)

    # ── Step 5: Export tidy CSV ───────────────────────────────────────────
    df = predictor.to_dataframe(results)
    df.to_csv("artifacts/results/predictions.csv", index=False)
    print(f"[Predictor] Tidy results saved → artifacts/results/predictions.csv")

    # ── Step 6: Plot ──────────────────────────────────────────────────────
    predictor.plot_results(results, save_path="artifacts/results/prediction_plots.png")

    return results


if __name__ == "__main__":
    run_demo()