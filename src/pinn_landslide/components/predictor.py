"""
PINN Landslide Predictor
========================
Loads a trained PINN model and, given a CSV file containing hourly
forecasted rainfall, predicts when and where the Factor of Safety
first drops below 1.0 — providing an early warning lead time.

Expected rainfall CSV format
-----------------------------
    t,rainfall
    1155,0.0
    1156,2.3
    1157,5.1
    ...

  - Column `t`        : simulation hour (absolute, same units as training)
  - Column `rainfall` : hourly rainfall intensity in mm/hr

Usage
-----
    predictor = SlopeFailurePredictor()
    results   = predictor.predict_from_csv(
                    rainfall_csv  = "data/forecast_rainfall.csv",
                    z_values      = np.linspace(0.1, 55.0, 100),
                )
    predictor.print_report(results)
    predictor.plot_results(results)
    predictor.to_dataframe(results).to_csv("results/predictions.csv")
"""

import json
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from typing import Optional, Dict, Any

from src.pinn_landslide.utils.utils import create_directories, _vg_Se, _vg_K
from src.pinn_landslide.components.pinn_architecture import PINN
from src.pinn_landslide.config.configuration import ConfigurationManager


# ── Rainfall → boundary-flux conversion ──────────────────────────────────────
def rainfall_to_flux(rainfall_mm_hr: np.ndarray, Ks_m_s: float) -> np.ndarray:
    """
    Convert rainfall intensity (mm/hr) to infiltration flux (m/s).
    Infiltration is capped at Ks (excess becomes runoff).
    Returns negative values (downward infiltration).
    """
    flux_m_s = rainfall_mm_hr / 1000.0 / 3600.0
    return -np.minimum(flux_m_s, Ks_m_s)


# =============================================================================
# Main predictor class
# =============================================================================

class SlopeFailurePredictor:
    """
    End-to-end predictor for slope stability using a trained PINN.

    Loads norm_params.json saved by the trainer automatically so that
    normalisation exactly matches training.

    Parameters
    ----------
    model_path : str | Path, optional
        Path to pinn_model.pt.  Defaults to artifacts/model/pinn_model.pt.
    norm_params_path : str | Path, optional
        Path to norm_params.json.  Defaults to artifacts/model/norm_params.json.
    device : torch.device, optional
        CPU or CUDA.  Auto-detected if not specified.
    """

    DEFAULT_MODEL_PATH       = Path("artifacts/model/pinn_model.pt")
    DEFAULT_NORM_PARAMS_PATH = Path("artifacts/model/norm_params.json")

    def __init__(
        self,
        model_path:       str | Path = None,
        norm_params_path: str | Path = None,
        device:           torch.device = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Load configuration ────────────────────────────────────────────
        config_manager = ConfigurationManager()
        self.geo       = config_manager.get_geo_params_config()
        self.arch      = config_manager.get_pinn_arch_config()
        self.win       = config_manager.get_window_config()
        self.beta_rad  = math.radians(self.geo.beta)
        self.phi_rad   = math.radians(self.geo.phi_prime)

        # ── Load normalisation parameters from saved JSON ─────────────────
        norm_path = Path(norm_params_path or self.DEFAULT_NORM_PARAMS_PATH)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"norm_params.json not found at {norm_path}.\n"
                f"Run training first — trainer.py saves this file automatically."
            )
        with open(norm_path) as f:
            self.norm = json.load(f)

        self.T_START = float(self.norm.get('t_window_start', self.win.t_start_hr))
        self.T_DUR   = float(self.norm['t_max'])
        self.Z_MAX   = float(self.norm['z_max'])
        self.PSI_MIN = float(self.norm['psi_min'])
        self.PSI_MAX = float(self.norm['psi_max'])
        self.T_END   = self.T_START + self.T_DUR

        print(f"[Predictor] norm_params loaded  →  window {self.T_START:.0f}"
              f"–{self.T_END:.0f} hr  |  "
              f"psi {self.PSI_MIN:.3f} to {self.PSI_MAX:.3f} m")

        # ── Load trained model ────────────────────────────────────────────
        model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}.")
        self.model = PINN(self.arch).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[Predictor] Model loaded  →  {model_path}  |  device: {self.device}")

    # ── normalisation helpers (must match data_loader.py) ────────────────
    def _t_norm(self, t_hr: np.ndarray) -> np.ndarray:
        return (t_hr - self.T_START) / self.T_DUR

    def _z_norm(self, z_m: np.ndarray) -> np.ndarray:
        return z_m / self.Z_MAX

    def _psi_denorm(self, psi_norm: np.ndarray) -> np.ndarray:
        return psi_norm * (self.PSI_MAX - self.PSI_MIN) + self.PSI_MIN

    # =========================================================================
    # CSV loader
    # =========================================================================

    def load_rainfall_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """
        Load and validate a rainfall CSV file.

        Expected columns:
            t         — simulation hour (float or int)
            rainfall  — hourly rainfall intensity in mm/hr (float)

        Returns sorted DataFrame with exactly these two columns.

        Raises
        ------
        ValueError  if required columns are missing or values are invalid.
        FileNotFoundError  if the file does not exist.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Rainfall CSV not found: {path}")

        df = pd.read_csv(path)

        # Accept common column name variants
        col_map = {}
        for col in df.columns:
            lc = col.strip().lower()
            if lc in ('t', 'time', 'time_hr', 'time_hours', 'hour', 'hours'):
                col_map[col] = 't'
            elif lc in ('rainfall', 'rain', 'rainfall_mm_hr', 'rain_mm_hr',
                        'precip', 'precipitation', 'intensity'):
                col_map[col] = 'rainfall'

        df = df.rename(columns=col_map)

        missing = [c for c in ('t', 'rainfall') if c not in df.columns]
        if missing:
            raise ValueError(
                f"Rainfall CSV is missing columns: {missing}\n"
                f"Found columns: {list(df.columns)}\n"
                f"Required: 't' (simulation hour) and 'rainfall' (mm/hr)"
            )

        df = df[['t', 'rainfall']].copy()
        df['t']        = pd.to_numeric(df['t'],        errors='coerce')
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')

        if df.isnull().any().any():
            raise ValueError("Rainfall CSV contains non-numeric values.")
        if (df['rainfall'] < 0).any():
            raise ValueError("Rainfall values must be >= 0 mm/hr.")

        df = df.sort_values('t').reset_index(drop=True)

        print(f"[Predictor] Rainfall CSV loaded  →  {path.name}")
        print(f"            rows     : {len(df)}")
        print(f"            t range  : {df['t'].min():.1f} to {df['t'].max():.1f} hr")
        print(f"            rainfall : {df['rainfall'].min():.2f} to "
              f"{df['rainfall'].max():.2f} mm/hr  "
              f"(total {df['rainfall'].sum():.1f} mm)")

        return df

    # =========================================================================
    # Core prediction from CSV
    # =========================================================================

    def predict_from_csv(
        self,
        rainfall_csv: str | Path,
        z_values:     np.ndarray,
    ) -> Dict[str, Any]:
        """
        Predict slope stability from a rainfall CSV file.

        Parameters
        ----------
        rainfall_csv : str | Path
            Path to CSV file with columns `t` (simulation hour) and
            `rainfall` (mm/hr).
        z_values : 1-D np.ndarray
            Depth values in metres to query (e.g. np.linspace(0.1, 55.0, 100)).

        Returns
        -------
        dict  — full results including FS, psi, failure time, lead time.
                Pass directly to print_report(), plot_results(), to_dataframe().
        """
        df = self.load_rainfall_csv(rainfall_csv)
        return self._run_prediction(
            abs_times_hr   = df['t'].values,
            rainfall_mm_hr = df['rainfall'].values,
            z_values       = z_values,
        )

    # =========================================================================
    # Core prediction (internal)
    # =========================================================================

    def _run_prediction(
        self,
        abs_times_hr:   np.ndarray,
        rainfall_mm_hr: np.ndarray,
        z_values:       np.ndarray,
    ) -> Dict[str, Any]:

        forecast_start_hr = float(abs_times_hr[0])

        # ── Clip to training window ───────────────────────────────────────
        valid_mask = (abs_times_hr >= self.T_START) & (abs_times_hr <= self.T_END)
        n_valid    = valid_mask.sum()

        if n_valid == 0:
            raise ValueError(
                f"No forecast hours fall within the training window "
                f"[{self.T_START:.0f}, {self.T_END:.0f}] hr.\n"
                f"Your CSV covers {abs_times_hr[0]:.0f}–{abs_times_hr[-1]:.0f} hr."
            )

        if not valid_mask.all():
            n_clipped = int((~valid_mask).sum())
            print(f"[Predictor] Warning: {n_clipped} rows outside training window "
                  f"[{self.T_START:.0f}, {self.T_END:.0f}] hr — clipped.")

        valid_times_hr   = abs_times_hr[valid_mask]
        valid_rainfall   = rainfall_mm_hr[valid_mask]
        forecast_offsets = valid_times_hr - forecast_start_hr

        # ── Build (z, t) mesh ─────────────────────────────────────────────
        ZZ, TT = np.meshgrid(z_values, valid_times_hr)
        z_flat = ZZ.ravel()
        t_flat = TT.ravel()

        z_tensor = torch.tensor(
            self._z_norm(z_flat), dtype=torch.float32, device=self.device
        ).view(-1, 1)
        t_tensor = torch.tensor(
            self._t_norm(t_flat), dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # ── PINN forward pass ─────────────────────────────────────────────
        with torch.no_grad():
            psi_norm_pred = self.model(z_tensor, t_tensor).cpu().numpy()

        n_z   = len(z_values)
        psi_phys = self._psi_denorm(psi_norm_pred).reshape(n_valid, n_z)

        # ── Hydraulic properties ──────────────────────────────────────────
        psi_t = torch.tensor(
            psi_phys.ravel(), dtype=torch.float32, device=self.device
        )
        K_phys  = _vg_K(psi_t, self.geo.Ks, self.geo.alpha,
                        self.geo.n, self.geo.l).cpu().numpy().reshape(n_valid, n_z)
        Se_phys = _vg_Se(psi_t, self.geo.alpha,
                         self.geo.n).cpu().numpy().reshape(n_valid, n_z)
        theta   = self.geo.theta_r + Se_phys * (self.geo.theta_s - self.geo.theta_r)

        # ── Factor of Safety ──────────────────────────────────────────────
        z_grid         = np.broadcast_to(z_values[np.newaxis, :], (n_valid, n_z))
        u              = np.maximum(psi_phys, 0.0) * self.geo.gamma_w
        dynamic_weight = self.geo.gamma + theta * self.geo.gamma_w
        sigma          = dynamic_weight * z_grid * math.cos(self.beta_rad) ** 2
        tau            = (dynamic_weight * z_grid
                          * math.sin(self.beta_rad) * math.cos(self.beta_rad))
        eff_stress     = np.maximum(sigma - u, 0.0)
        FS             = ((self.geo.c_prime
                           + eff_stress * math.tan(self.phi_rad))
                          / np.maximum(tau, 1e-9))

        # ── Failure detection ─────────────────────────────────────────────
        min_FS_series    = FS.min(axis=1)
        failure_mask     = min_FS_series <= 1.0
        failure_detected = bool(failure_mask.any())

        first_failure_abs_hr    = None
        first_failure_offset_hr = None
        first_failure_depth_m   = None
        warning_lead_time_hr    = None

        if failure_detected:
            idx = int(np.argmax(failure_mask))
            first_failure_abs_hr    = float(valid_times_hr[idx])
            first_failure_offset_hr = float(forecast_offsets[idx])
            first_failure_depth_m   = float(z_values[FS[idx].argmin()])
            warning_lead_time_hr    = first_failure_offset_hr

        return dict(
            abs_times_hr            = valid_times_hr,
            forecast_offsets_hr     = forecast_offsets,
            forecast_start_hr       = forecast_start_hr,
            t_window_start          = self.T_START,
            t_window_end            = self.T_END,
            z_values                = z_values,
            psi                     = psi_phys,
            K                       = K_phys,
            theta                   = theta,
            FS                      = FS,
            min_FS_series           = min_FS_series,
            rainfall_mm_hr          = valid_rainfall,
            infiltration_flux       = rainfall_to_flux(valid_rainfall, self.geo.Ks),
            failure_detected        = failure_detected,
            first_failure_abs_hr    = first_failure_abs_hr,
            first_failure_offset_hr = first_failure_offset_hr,
            first_failure_depth_m   = first_failure_depth_m,
            warning_lead_time_hr    = warning_lead_time_hr,
            min_fs_overall          = float(min_FS_series.min()),
        )

    # =========================================================================
    # Text report
    # =========================================================================

    def print_report(self, results: Dict[str, Any]) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  PINN SLOPE FAILURE PREDICTION — EARLY WARNING REPORT")
        print(sep)
        print(f"  Forecast start      : hr {results['forecast_start_hr']:.0f}  "
              f"(day {results['forecast_start_hr']/24:.1f})")
        print(f"  Training window     : hr {results['t_window_start']:.0f} – "
              f"hr {results['t_window_end']:.0f}")
        print(f"  Valid forecast hrs  : {len(results['abs_times_hr'])}")
        print(f"  Depth range queried : {results['z_values'][0]:.2f} – "
              f"{results['z_values'][-1]:.2f} m")
        print(f"  Total rainfall      : {results['rainfall_mm_hr'].sum():.1f} mm")
        print(f"  Peak rainfall       : {results['rainfall_mm_hr'].max():.1f} mm/hr")
        print()

        if results["failure_detected"]:
            print("  ⚠️  SLOPE FAILURE PREDICTED")
            print(f"  Failure at sim hour : {results['first_failure_abs_hr']:.1f} hr  "
                  f"(day {results['first_failure_abs_hr']/24:.2f})")
            print(f"  Forecast offset     : hour {results['first_failure_offset_hr']:.0f} "
                  f"of forecast")
            print(f"  Critical depth      : {results['first_failure_depth_m']:.2f} m")
            print(f"  Warning lead time   : {results['warning_lead_time_hr']:.1f} hr "
                  f"from forecast start")
            print(f"  Minimum FS          : {results['min_fs_overall']:.4f}")
        else:
            print("  ✅  SLOPE REMAINS STABLE throughout forecast window")
            print(f"  Minimum FS observed : {results['min_fs_overall']:.4f}")

        print()
        print(f"  {'Sim hr':>8}  {'Offset hr':>10}  {'Rain mm/hr':>12}  "
              f"{'Min FS':>10}  {'Status':>8}")
        print("  " + "-" * 56)
        for abs_t, off_t, rain, fs in zip(
            results["abs_times_hr"],
            results["forecast_offsets_hr"],
            results["rainfall_mm_hr"],
            results["min_FS_series"],
        ):
            status = "FAIL" if fs <= 1.0 else "OK"
            print(f"  {abs_t:>8.0f}  {off_t:>10.0f}  {rain:>12.2f}  "
                  f"{fs:>10.4f}  {status:>8}")
        print(sep + "\n")

    # =========================================================================
    # Export to CSV
    # =========================================================================

    def to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        for i, (abs_t, off_t, rain) in enumerate(zip(
            results["abs_times_hr"],
            results["forecast_offsets_hr"],
            results["rainfall_mm_hr"],
        )):
            for j, depth in enumerate(results["z_values"]):
                rows.append({
                    "sim_time_hr"    : abs_t,
                    "forecast_hr"    : off_t,
                    "z_m"           : depth,
                    "psi_m"         : results["psi"][i, j],
                    "K_m_s"         : results["K"][i, j],
                    "theta"         : results["theta"][i, j],
                    "FS"            : results["FS"][i, j],
                    "rainfall_mm_hr": rain,
                    "status"        : "FAIL" if results["FS"][i, j] <= 1.0
                                      else "STABLE",
                })
        return pd.DataFrame(rows)

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_results(self, results: Dict[str, Any],
                     save_path: Optional[str] = None) -> None:
        times  = results["forecast_offsets_hr"]
        z      = results["z_values"]
        FS     = results["FS"]
        psi    = results["psi"]
        rain   = results["rainfall_mm_hr"]
        min_fs = results["min_FS_series"]
        fail_hr = results.get("first_failure_offset_hr")

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

        # ── 1. Rainfall ───────────────────────────────────────────────────
        ax = axes[0]
        ax.bar(times, rain, width=0.8, color="#388bfd", alpha=0.85)
        ax.set_ylabel("Rainfall (mm/hr)")
        ax.set_title("Forecasted Hourly Rainfall", **title_kw)
        if fail_hr is not None:
            ax.axvline(fail_hr, color="#f85149", ls="--", lw=1.4,
                       label=f"Failure at forecast hr {fail_hr:.0f}")
            ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)

        # ── 2. Min FS series ──────────────────────────────────────────────
        ax = axes[1]
        bar_colors = np.where(min_fs <= 1.0, "#f85149", "#3fb950")
        ax.bar(times, min_fs, width=0.8, color=bar_colors, alpha=0.85)
        ax.axhline(1.0, color="#e3b341", ls="--", lw=1.5,
                   label="FS = 1.0 (failure threshold)")
        ax.set_ylabel("Minimum FS (all depths)")
        ax.set_title("Factor of Safety — Minimum Over Depth Profile", **title_kw)
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        if fail_hr is not None:
            ax.axvline(fail_hr, color="#f85149", ls=":", lw=1.2)

        # ── 3. FS heatmap ─────────────────────────────────────────────────
        ax = axes[2]
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
        im = ax.pcolormesh(times, z, np.clip(FS, 0.0, 3.0).T,
                           cmap="RdYlGn", norm=norm, shading="auto")
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(colors="#8b949e")
        cb.set_label("FS", color="#c9d1d9")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Factor of Safety  (depth × forecast hour)", **title_kw)
        if fail_hr is not None:
            ax.axvline(fail_hr, color="#f85149", ls="--", lw=1.4,
                       label=f"First failure hr {fail_hr:.0f}")
            ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)

        # ── 4. Pressure head heatmap ──────────────────────────────────────
        ax = axes[3]
        vm = max(abs(psi.min()), abs(psi.max()))
        im2 = ax.pcolormesh(times, z, psi.T,
                            cmap="coolwarm", vmin=-vm, vmax=vm, shading="auto")
        cb2 = fig.colorbar(im2, ax=ax, pad=0.01)
        cb2.ax.tick_params(colors="#8b949e")
        cb2.set_label("ψ (m)", color="#c9d1d9")
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Forecast hour (offset from start)")
        ax.set_title("Pressure Head  ψ(z, t)  (depth × forecast hour)", **title_kw)

        title = "PINN Slope Stability — Early Warning Prediction"
        if results["failure_detected"]:
            title += f"\n⚠  Failure predicted at forecast hour {fail_hr:.0f}"
        fig.suptitle(title, color="#e6edf3", fontsize=13,
                     fontweight="bold", y=1.01)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[Predictor] Figure saved → {save_path}")
        else:
            plt.show()


def convert_rainfall_csv(
        rainfall_m_hr_csv: str | Path,
        t_min: float = 1000.0,
        t_max: float = 1500.0,
        output_path: str | Path = "artifacts/dataset/rainfall_mm_hr.csv",
) -> str:
    """
    Convert a rainfall CSV from m/hr to mm/hr and filter to a time window.

    Parameters
    ----------
    rainfall_m_hr_csv : str | Path
        Input CSV with columns t_hours and Rain_m_hr (units: m/hr).
    t_min : float
        Start of time window in simulation hours (default 1000).
    t_max : float
        End of time window in simulation hours (default 1500).
    output_path : str | Path
        Where to save the converted CSV.

    Returns
    -------
    str — path to the saved CSV (pass directly to predict_from_csv).
    """
    df = pd.read_csv(rainfall_m_hr_csv)

    # Rename columns to standard names
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ('t', 'time', 't_hours', 'time_hr', 'time_hours', 'hour'):
            col_map[col] = 't'
        elif 'm_hr' in lc or 'rain' in lc or 'precip' in lc:
            col_map[col] = 'rainfall_m_hr'
    df = df.rename(columns=col_map)

    if 't' not in df.columns or 'rainfall_m_hr' not in df.columns:
        raise ValueError(
            f"Could not find time and rainfall columns.\n"
            f"Found: {list(df.columns)}"
        )

    # Filter to time window
    df = df[(df['t'] >= t_min) & (df['t'] <= t_max)].copy()

    if len(df) == 0:
        raise ValueError(
            f"No rows found between t={t_min} and t={t_max}."
        )

    # Convert m/hr -> mm/hr
    df['rainfall'] = df['rainfall_m_hr'] * 1000.0
    df = df[['t', 'rainfall']].reset_index(drop=True)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[Predictor] Rainfall converted  m/hr \u2192 mm/hr")
    print(f"            window   : t={df['t'].min():.0f} to t={df['t'].max():.0f} hr  "
          f"({len(df)} rows)")
    print(f"            range    : {df['rainfall'].min():.3f} to "
          f"{df['rainfall'].max():.3f} mm/hr")
    print(f"            saved \u2192  {output_path}")

    return str(output_path)


def run(rainfall_csv: str, z_min: float = 0.1, z_max: float = 55.0,
        n_depths: int = 100, output_dir: str = "artifacts/results"):
    """
    Run the predictor from a rainfall CSV file (m/hr or mm/hr).

    Parameters
    ----------
    rainfall_csv : str
        Path to CSV with columns t_hours and Rain_m_hr (m/hr).
        Automatically converted to mm/hr and filtered to t=1000-1500.
    z_min : float
        Shallowest depth to query in metres.
    z_max : float
        Deepest depth to query in metres.
    n_depths : int
        Number of depth points.
    output_dir : str
        Directory where predictions.csv and prediction_plots.png are saved.
    """
    # Convert m/hr -> mm/hr, filter to t=1000-1500
    converted_csv = convert_rainfall_csv(
        rainfall_m_hr_csv = rainfall_csv,
        t_min             = 1000.0,
        t_max             = 1500.0,
        output_path       = f"{output_dir}/rainfall_mm_hr.csv",
    )

    predictor = SlopeFailurePredictor()
    z_values  = np.linspace(z_min, z_max, n_depths)

    results = predictor.predict_from_csv(
        rainfall_csv = converted_csv,
        z_values     = z_values,
    )

    create_directories([output_dir])
    predictor.print_report(results)

    df = predictor.to_dataframe(results)
    csv_out = f"{output_dir}/predictions.csv"
    df.to_csv(csv_out, index=False)
    print(f"[Predictor] Results saved \u2192 {csv_out}")

    plot_out = f"{output_dir}/prediction_plots.png"
    predictor.plot_results(results, save_path=plot_out)

    return results

if __name__ == "__main__":
    import sys
    

    csv_path  = ('dataset/hourly_rainfall.csv')   
    z_min_arg =  0.1
    z_max_arg =  55.0
    n_dep_arg =  100

    run(csv_path, z_min_arg, z_max_arg, n_dep_arg)