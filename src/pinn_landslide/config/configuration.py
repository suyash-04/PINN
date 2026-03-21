from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.pinn_landslide.constants import *
from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.entity.config_entity import (
    PINNArchConfig,
    PINNLossConfig,
    GeoParamsConfig,
    TrainingConfig,
    WindowConfig,
)
from src.pinn_landslide.logger.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WINDOW_KEYS = (
    't_start_hr',
    't_end_hr',
    't_dur_hr',
    'z_max_m',
    'psi_min_m',
    'psi_max_m',
    'phys_scale',
)


class ConfigurationManager():
    def __init__(
        self,
        CONFIG_FILE_PATH=CONFIG_FILE_PATH,
        PARAMS_FILE_PATH=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        create_directories([self.config.artifacts_root])

    def get_pinn_arch_config(self) -> PINNArchConfig:
        a = self.params['architecture']
        logger.info(f"PINN Architecture Config: {a}")
        return PINNArchConfig(
            n_hidden_layers  = a['n_hidden_layers'],
            neurons_per_layer= a['neurons_per_layer'],
            activation       = a['activation'],
            input_dim        = a['input_dim'],
            output_dim       = a['output_dim'],
        )

    def get_geo_params_config(self) -> GeoParamsConfig:
        g = self.params['geo_params']
        logger.info(f"Geo Params Config: {g}")
        return GeoParamsConfig(
            Ks       = g['Ks'],
            theta_s  = g['theta_s'],
            theta_r  = g['theta_r'],
            alpha    = g['alpha'],
            n        = g['n'],
            l        = g['l'],
            c_prime  = g['c_prime'],
            phi_prime= g['phi_prime'],
            gamma    = g['gamma'],
            beta     = g['beta'],
            gamma_w  = g['gamma_w'],
        )

    def get_loss_weights_config(self) -> PINNLossConfig:
        w = self.params['loss_weights']
        logger.info(f"Loss Weights Config: {w}")
        return PINNLossConfig(
            lambda_physics = w['lambda_physics'],
            lambda_anchor  = w['lambda_anchor'],
            lambda_initial = w['lambda_initial'],
            lambda_boundary= w['lambda_boundary'],
            lambda_failure = w['lambda_failure'],
        )

    def get_training_config(self) -> TrainingConfig:
        t = self.params['training']
        logger.info(f"Training Config: {t}")
        return TrainingConfig(
            adam_epochs  = t['adam_epochs'],
            lbfgs_epochs = t['lbfgs_epochs'],
            learning_rate= t['learning_rate'],
            device       = t['device'],
        )

    def get_window_config(self) -> WindowConfig:
        """
        Returns the pre-failure window normalisation constants.
        Used by Dataloader to normalise t, z, psi consistently,
        and by loss.py to de-normalise for PDE evaluation.
        """
        w = self.params.get('window')
        parsed = self._parse_window_values(w)

        if all(value is not None for value in parsed.values()):
            logger.info(f"Window Config: {parsed}")
            return WindowConfig(**parsed)

        derived = self._derive_window_config()
        logger.warning(
            "Window config missing or incomplete in params.yaml; "
            f"derived values from dataset files instead: {derived}"
        )
        return WindowConfig(**derived)

    def _parse_window_values(self, window_block) -> dict[str, float | None]:
        parsed = {}
        for key in WINDOW_KEYS:
            value = None if window_block is None else window_block.get(key)
            parsed[key] = self._coerce_float(value)
        return parsed

    def _coerce_float(self, value) -> float | None:
        if value is None or isinstance(value, Mapping):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_path(self, path_like) -> Path:
        path = Path(str(path_like)).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _derive_window_config(self) -> dict[str, float]:
        anchor_path = self._resolve_path(self.config.data_ingestion.anchor_csv)
        ic_path = self._resolve_path(self.config.data_ingestion.ic_csv)

        anchor_df = pd.read_csv(anchor_path)
        ic_df = pd.read_csv(ic_path)

        if 't_hr' in ic_df.columns and ic_df['t_hr'].notna().any():
            t_start_hr = float(ic_df['t_hr'].dropna().iloc[0])
        else:
            t_start_hr = float(anchor_df['t_hours'].min())

        duration_candidates = None
        if {'t_hours', 't_norm'}.issubset(anchor_df.columns):
            valid = anchor_df['t_norm'].notna() & (anchor_df['t_norm'] > 0)
            if valid.any():
                duration_candidates = (
                    (anchor_df.loc[valid, 't_hours'] - t_start_hr)
                    / anchor_df.loc[valid, 't_norm']
                )
                duration_candidates = duration_candidates[
                    duration_candidates > 0
                ]

        if duration_candidates is not None and not duration_candidates.empty:
            t_dur_hr = float(duration_candidates.median())
        else:
            t_dur_hr = float(anchor_df['t_hours'].max() - t_start_hr)

        z_candidates = None
        if {'z_meters', 'z_norm'}.issubset(anchor_df.columns):
            valid = anchor_df['z_norm'].notna() & (anchor_df['z_norm'] > 0)
            if valid.any():
                z_candidates = (
                    anchor_df.loc[valid, 'z_meters'].abs()
                    / anchor_df.loc[valid, 'z_norm']
                )
                z_candidates = z_candidates[z_candidates > 0]

        if z_candidates is not None and not z_candidates.empty:
            z_max_m = float(z_candidates.median())
        else:
            z_max_m = float(
                max(anchor_df['z_meters'].abs().max(), ic_df['z_m'].abs().max())
            )
        psi_min_m = float(
            min(anchor_df['psi_meters'].min(), ic_df['h_m'].min())
        )
        psi_max_m = float(
            max(anchor_df['psi_meters'].max(), ic_df['h_m'].max())
        )

        return {
            't_start_hr': t_start_hr,
            't_end_hr': t_start_hr + t_dur_hr,
            't_dur_hr': t_dur_hr,
            'z_max_m': z_max_m,
            'psi_min_m': psi_min_m,
            'psi_max_m': psi_max_m,
            'phys_scale': 9.818e-10,
        }
