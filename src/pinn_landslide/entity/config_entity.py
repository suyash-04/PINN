from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class PINNArchConfig:
    n_hidden_layers  : int
    neurons_per_layer: int
    activation       : str
    input_dim        : int
    output_dim       : int


@dataclass(frozen=True)
class PINNLossConfig:
    lambda_physics : float
    lambda_anchor  : float
    lambda_initial : float
    lambda_boundary: float
    lambda_failure : float


@dataclass(frozen=True)
class GeoParamsConfig:
    Ks       : float
    theta_s  : float
    theta_r  : float
    alpha    : float
    n        : float
    l        : float
    c_prime  : float
    phi_prime: float
    gamma    : float
    beta     : float
    gamma_w  : float


@dataclass(frozen=True)
class TrainingConfig:
    adam_epochs  : int
    lbfgs_epochs : int
    learning_rate: float
    device       : str


@dataclass(frozen=True)
class WindowConfig:
    """
    Normalisation constants for the pre-failure training window.
    All time values are in hours.  Loaded from params.yaml [window] block.
    """
    t_start_hr: float   # window start — treated as t_norm = 0
    t_end_hr  : float   # failure time — treated as t_norm = 1
    t_dur_hr  : float   # window duration = t_end - t_start
    z_max_m   : float   # full column depth
    psi_min_m : float   # minimum psi across anchor + IC data
    psi_max_m : float   # maximum psi across anchor + IC data
    phys_scale: float   # Richards residual char. scale for loss.py