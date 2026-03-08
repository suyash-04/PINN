from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen= True)
class PINNArchConfig:
    n_hidden_layers: int
    neurons_per_layer: int
    activation: str 
    input_dim : int
    output_dim : int


@dataclass(frozen=True)
class PINNLossConfig:
    lambda_physics: float
    lambda_anchor: float
    lambda_initial: float
    lambda_boundary: float
    lambda_failure: float

@dataclass(frozen=True)
class GeoParamsConfig:
    Ks: float          # Saturated hydraulic conductivity (m/s) — phyllite-schist
    theta_s: float      # Saturated water content
    theta_r: float      # Residual water content
    alpha: float       # van Genuchten alpha (1/m)
    n: float           # Pore-size distribution index
    l: float           # Tortuosity / connectivity
    c_prime: float     # Effective cohesion (kPa)
    phi_prime: float   # Effective friction angle (degrees)
    gamma: float       # Bulk unit weight (kN/m3)
    beta: float        # Slope angle (degrees)
    gamma_w:float
@dataclass(frozen=True)

class TrainingConfig:
    adam_epochs: int
    lbfgs_epochs: int
    learning_rate: float
    device: str

