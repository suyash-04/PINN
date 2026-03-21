import os
import sys
from src.pinn_landslide.constants import *
from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.entity.config_entity import (
    PINNArchConfig,
    PINNLossConfig,
    GeoParamsConfig,
    TrainingConfig,
    WindowConfig,
)
from src.pinn_landslide.exception.exception import customexception
from src.pinn_landslide.logger.logger import logger


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
        w = self.params['window']
        logger.info(f"Window Config: {w}")
        return WindowConfig(
            t_start_hr= float(w['t_start_hr']),
            t_end_hr  = float(w['t_end_hr']),
            t_dur_hr  = float(w['t_dur_hr']),
            z_max_m   = float(w['z_max_m']),
            psi_min_m = float(w['psi_min_m']),
            psi_max_m = float(w['psi_max_m']),
            phys_scale= float(w['phys_scale']),
        )