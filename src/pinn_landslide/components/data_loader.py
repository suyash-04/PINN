import pandas as pd
import torch
import numpy as np
from pathlib import Path
from scipy.stats.qmc import LatinHypercube          # GAP 2: LHS import
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.constants import *


class Dataloader():
    def __init__(self, device: torch.device, config: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
        self.device = device

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_ingestion.ingested_data)
        return df

    def extract_failure_points(self, df: pd.DataFrame, z_max: float, t_max: float):
        failing_rows = df[(df['FS'] <= 1.0) & (df['Depth_m'] > 0.1)]

        if len(failing_rows) == 0:
            print("Warning: HYDRUS simulation never generated enough pore pressure to fail the slope!")
            return None, None

        z_fail_norm = torch.tensor(
            failing_rows['Depth_m'].values / z_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)
        t_fail_norm = torch.tensor(
            failing_rows['Time_days'].values / t_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)

        return z_fail_norm, t_fail_norm

    def get_real_batch(self, df: pd.DataFrame):

        # ── 1. Normalisation boundaries ───────────────────────────────────
        t_max   = float(df['Time_days'].max())
        z_max   = float(df['Depth_m'].max())
        psi_min = float(df['Pressure_Head'].min())
        psi_max = float(df['Pressure_Head'].max())

        # ── 2. Anchor data tensors (normalised) ───────────────────────────
        t_data = torch.tensor(
            df['Time_days'].values / t_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)
        z_data = torch.tensor(
            df['Depth_m'].values / z_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)
        psi_data = torch.tensor(
            (df['Pressure_Head'].values - psi_min) / (psi_max - psi_min),
            dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # ── 3. GAP 2: Latin Hypercube collocation points ──────────────────
        # Replaces torch.rand — guarantees uniform stratified domain coverage.
        # Proposal §5.3.2 specifies LHS explicitly. seed=42 = reproducible.
        N_coll  = 10000
        sampler = LatinHypercube(d=2, seed=42)
        lhs_pts = sampler.random(n=N_coll)          # (10000, 2) in [0, 1]

        z_coll = torch.tensor(
            lhs_pts[:, 0:1], dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        t_coll = torch.tensor(
            lhs_pts[:, 1:2], dtype=torch.float32, device=self.device
        ).requires_grad_(True)

        # ── 4. GAP 1: Initial condition points (t = t_min, all depths) ────
        # Enforces ψ(z,0) = ψ_HYDRUS(z,0) — the pre-monsoon suction profile.
        # Anchors the PINN to the realistic antecedent moisture state so the
        # temporal evolution of Richards equation starts from a physical state.
        t_min_val = df['Time_days'].min()
        ic_mask   = df['Time_days'] == t_min_val

        z_ic = torch.tensor(
            df.loc[ic_mask, 'Depth_m'].values / z_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)
        t_ic = torch.zeros(
            int(ic_mask.sum()), 1,
            dtype=torch.float32, device=self.device
        )
        psi_ic = torch.tensor(
            (df.loc[ic_mask, 'Pressure_Head'].values - psi_min) / (psi_max - psi_min),
            dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # ── 5. GAP 1: Surface boundary points (z = z_min, all time steps) ─
        # Enforces the Darcy flux at the surface to match the rainfall input.
        # target_flux is negative (infiltrating downward, m/s).
        #
        # Currently uses −Ks as a placeholder (fully infiltrating at every
        # time step). If you have DHM hourly rainfall as a column in df,
        # replace the torch.full line with that array converted to m/s:
        #   flux_vals = -np.minimum(df.loc[bc_mask,'Rainfall_mm_hr'].values
        #                           / 1000 / 3600, geo.Ks)
        #   target_flux = torch.tensor(flux_vals, ...).view(-1,1)
        z_min_val = df['Depth_m'].min()
        bc_mask   = df['Depth_m'] == z_min_val

        z_bc = torch.tensor(
            df.loc[bc_mask, 'Depth_m'].values / z_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)
        t_bc = torch.tensor(
            df.loc[bc_mask, 'Time_days'].values / t_max,
            dtype=torch.float32, device=self.device
        ).view(-1, 1)

        geo         = ConfigurationManager().get_geo_params_config()
        target_flux = torch.full(
            (int(bc_mask.sum()), 1), -geo.Ks,
            dtype=torch.float32, device=self.device
        )

        # ── 6. Failure points ─────────────────────────────────────────────
        z_fail, t_fail = self.extract_failure_points(df, z_max, t_max)

        # ── 7. Assemble batch ─────────────────────────────────────────────
        batch = {
            # Collocation — physics loss
            'z_coll'      : z_coll,
            't_coll'      : t_coll,
            # Anchor data — anchor loss
            'z_data'      : z_data,
            't_data'      : t_data,
            'psi_data'    : psi_data,
            # Failure points — failure loss
            'z_fail'      : z_fail,
            't_fail'      : t_fail,
            # Initial condition — ic loss  ← GAP 1 NEW
            'z_ic'        : z_ic,
            't_ic'        : t_ic,
            'psi_ic'      : psi_ic,
            # Surface boundary — bc loss   ← GAP 1 NEW
            'z_bc'        : z_bc,
            't_bc'        : t_bc,
            'target_flux' : target_flux,
            # Normalisation params
            'norm_params' : {
                't_max'   : t_max,
                'z_max'   : z_max,
                'psi_min' : psi_min,
                'psi_max' : psi_max,
            }
        }
        self.save_batch(batch)
        return batch

    def save_batch(self, batch: dict):
        path = self.config.data_ingestion.batch_path
        torch.save(batch, path)
        print(f"Batch saved at {path}")