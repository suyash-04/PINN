import pandas as pd
import torch
import numpy as np
from pathlib import Path
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.constants import *


class Dataloader():
    def __init__(self, device: torch.device, config: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
        self.device = device

        win = ConfigurationManager().get_window_config()
        self.T_START = win.t_start_hr
        self.T_DUR   = win.t_dur_hr
        self.Z_MAX   = win.z_max_m
        self.PSI_MIN = win.psi_min_m
        self.PSI_MAX = win.psi_max_m

    # ── normalisation helpers ─────────────────────────────────────────────────

    def _t_norm(self, t_hr: np.ndarray) -> np.ndarray:
        return (t_hr - self.T_START) / self.T_DUR

    def _z_norm(self, z_m: np.ndarray) -> np.ndarray:
        return z_m / self.Z_MAX

    def _psi_norm(self, psi_m: np.ndarray) -> np.ndarray:
        return (psi_m - self.PSI_MIN) / (self.PSI_MAX - self.PSI_MIN)

    def _tensor(self, arr: np.ndarray, grad: bool = False) -> torch.Tensor:
        t = torch.tensor(
            np.asarray(arr, dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        ).view(-1, 1)
        return t.requires_grad_(True) if grad else t

    # ── public interface ──────────────────────────────────────────────────────

    def get_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.config.data_ingestion.ingested_data)

    # ── point loaders ─────────────────────────────────────────────────────────

    def _load_anchor(self) -> dict:
        """
        Load pre-built anchor points CSV.
        Recomputes t_norm and psi_norm from physical values — the stored
        normalised columns in the file used a different reference.
        """
        df = pd.read_csv(self.config.data_ingestion.anchor_csv)

        return {
            'z_data'  : self._tensor(self._z_norm(df['z_meters'].values)),
            't_data'  : self._tensor(self._t_norm(df['t_hours'].values)),
            'psi_data': self._tensor(self._psi_norm(df['psi_meters'].values)),
        }

    def _load_collocation(self) -> dict:
        """
        Load pre-built zone-biased collocation CSV.
        Recomputes t_norm from t_hr — the stored t_norm used a wrong reference.
        z_norm is recomputed from z_m for consistency.
        Collocation points require grad=True for autograd in physics_loss.
        """
        df = pd.read_csv(self.config.data_ingestion.collocation_csv)

        return {
            'z_coll': self._tensor(self._z_norm(df['z_m'].values),  grad=True),
            't_coll': self._tensor(self._t_norm(df['t_hr'].values),  grad=True),
        }

    def _load_initial_condition(self) -> dict:
        """
        Load the pressure head profile at the window start (t = T_START).
        Depth values in the IC file are negative (HYDRUS convention) —
        abs() is applied before normalisation.
        t_norm = 0 for all IC points (they are at the start of the domain).
        """
        df  = pd.read_csv(self.config.data_ingestion.ic_csv)
        n   = len(df)
        z_m = np.abs(df['z_m'].values)

        return {
            'z_ic'  : self._tensor(self._z_norm(z_m)),
            't_ic'  : torch.zeros(n, 1, dtype=torch.float32, device=self.device),
            'psi_ic': self._tensor(self._psi_norm(df['h_m'].values)),
        }

    def _build_boundary(self) -> dict:
        """
        Surface boundary condition: z = 0 at each timestep in the window.
        Uses the unique timesteps from the anchor CSV as the time grid.
        target_flux = -Ks (placeholder uniform infiltration).
        """
        df           = pd.read_csv(self.config.data_ingestion.anchor_csv)
        t_hrs_unique = np.unique(df['t_hours'].values)
        n_bc         = len(t_hrs_unique)

        geo         = ConfigurationManager().get_geo_params_config()
        target_flux = torch.full(
            (n_bc, 1), float(-geo.Ks),
            dtype=torch.float32, device=self.device
        )

        return {
            'z_bc'        : self._tensor(np.zeros(n_bc)),
            't_bc'        : self._tensor(self._t_norm(t_hrs_unique)),
            'target_flux' : target_flux,
        }

    def _extract_failure_points(self, df: pd.DataFrame) -> dict:
        """
        Extract failure points from the processed ingested CSV.
        Keeps only points inside the training window (t_norm in [0, 1]).
        Supports both Time_hours and Time_days column names.
        """
        if 'Time_hours' in df.columns:
            t_hr = df['Time_hours'].values
        elif 'Time_days' in df.columns:
            t_hr = df['Time_days'].values * 24.0
        else:
            raise ValueError("Processed CSV must have 'Time_hours' or 'Time_days'")

        failing = df[(df['FS'] <= 1.0) & (df['Depth_m'] > 0.1)].copy()
        failing_t = t_hr[(df['FS'] <= 1.0) & (df['Depth_m'] > 0.1)]

        if len(failing) == 0:
            print("  Warning: no failure points found (FS <= 1.0).")
            return {'z_fail': None, 't_fail': None}

        t_norm = self._t_norm(failing_t)
        z_norm = self._z_norm(failing['Depth_m'].values)

        mask   = (t_norm >= 0.0) & (t_norm <= 1.0)
        t_norm = t_norm[mask]
        z_norm = z_norm[mask]

        if len(t_norm) == 0:
            print("  Warning: no failure points fall within the training window.")
            return {'z_fail': None, 't_fail': None}

        print(f"  Failure points in window : {len(t_norm):,}")
        return {
            'z_fail': self._tensor(z_norm),
            't_fail': self._tensor(t_norm),
        }

    # ── master batch builder ──────────────────────────────────────────────────

    def get_real_batch(self, df: pd.DataFrame) -> dict:
        """
        Assemble the full PINN training batch.

        Point types and their sources:
          Collocation (10,000) : collocation_points_10000.csv  — physics enforcement
          Anchor      (1,000)  : PINN_anchor_points_FINAL.csv  — labelled HYDRUS data
          IC          (999)    : PINN_initial_condition_hr1323.csv — profile at t_start
          Boundary    (~n_ts)  : built from anchor timestep grid — surface flux
          Failure     (varies) : subset of ingested CSV where FS <= 1.0

        All tensors share the same normalisation:
          t_norm   = (t_hours - 1323) / 166
          z_norm   = z_m / 55
          psi_norm = (psi - (-2.457)) / 4.252
        """
        print("Building training batch...")
        print(f"  Window : {self.T_START:.0f} to {self.T_START + self.T_DUR:.0f} hr"
              f"  ({self.T_DUR/24:.1f} days)")
        print(f"  psi    : {self.PSI_MIN:.3f} to {self.PSI_MAX:.3f} m")

        anchor      = self._load_anchor()
        collocation = self._load_collocation()
        ic          = self._load_initial_condition()
        boundary    = self._build_boundary()
        failure     = self._extract_failure_points(df)

        print(f"  Anchor pts      : {len(anchor['z_data']):,}")
        print(f"  Collocation pts : {len(collocation['z_coll']):,}")
        print(f"  IC pts          : {len(ic['z_ic']):,}")
        print(f"  BC pts          : {len(boundary['z_bc']):,}")
        if failure['z_fail'] is not None:
            print(f"  Failure pts     : {len(failure['z_fail']):,}")

        batch = {
            'z_coll'      : collocation['z_coll'],
            't_coll'      : collocation['t_coll'],
            'z_data'      : anchor['z_data'],
            't_data'      : anchor['t_data'],
            'psi_data'    : anchor['psi_data'],
            'z_fail'      : failure['z_fail'],
            't_fail'      : failure['t_fail'],
            'z_ic'        : ic['z_ic'],
            't_ic'        : ic['t_ic'],
            'psi_ic'      : ic['psi_ic'],
            'z_bc'        : boundary['z_bc'],
            't_bc'        : boundary['t_bc'],
            'target_flux' : boundary['target_flux'],
            'norm_params' : {
                't_max'          : self.T_DUR,
                't_window_start' : self.T_START,
                'z_max'          : self.Z_MAX,
                'psi_min'        : self.PSI_MIN,
                'psi_max'        : self.PSI_MAX,
                'time_unit'      : 'hours',
            },
        }

        self.save_batch(batch)
        return batch

    def save_batch(self, batch: dict):
        path = self.config.data_ingestion.batch_path
        torch.save(batch, path)
        print(f"  Batch saved → {path}")