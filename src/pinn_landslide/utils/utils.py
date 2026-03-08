import os, sys
import yaml
from src.pinn_landslide.logger.logger import logger
from src.pinn_landslide.exception.exception import customexception
import json, joblib
from ensure import ensure_annotations
from box import Box  
from pathlib import Path
from typing import Any
import base64
import torch 

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Box:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} read successfully")
            # Convert dictionary to Box for attribute-style access
            return Box(content, default_box=True, frozen_box=False)
    except Exception as e:
        raise customexception(e, sys)

@ensure_annotations
def create_directories(path_to_direcories: list, verbose=True):
    try:
        for path in path_to_direcories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"directory : {path} created successfully")
    except Exception as e:
        raise customexception(e, sys)


import torch

def _vg_Se(psi: torch.Tensor, alpha: float, n: float) -> torch.Tensor:
    """Effective saturation Se(psi). psi in metres."""
    m = 1.0 - 1.0 / n
    
    # FIX: Clamp psi to avoid exactly 0 in the unsaturated branch
    psi_safe = torch.clamp(torch.abs(psi), min=1e-8)
    
    return torch.where(
        psi < 0,
        (1.0 + (alpha * psi_safe) ** n) ** (-m),
        torch.ones_like(psi),
    )


def _vg_K(psi: torch.Tensor, Ks: float, alpha: float, n: float, l: float = 0.5) -> torch.Tensor:
    m = 1.0 - 1.0 / n
    # Se is clamped safely below 1.0 to prevent (1.0 - Se^(1/m)) from becoming exactly 0
    Se = _vg_Se(psi, alpha, n).clamp(1e-8, 1.0 - 1e-6)
    
    # FIX: Clamp the inner term before raising to power m to prevent NaN gradients
    inner_base = torch.clamp(1.0 - Se ** (1.0 / m), min=1e-8)
    inner = (1.0 - inner_base ** m) ** 2
    
    return Ks * (Se ** l) * inner


def _vg_C(psi: torch.Tensor, theta_r: float, theta_s: float, alpha: float, n: float) -> torch.Tensor:
    """Specific moisture capacity C = dtheta/dpsi."""
    m = 1.0 - 1.0 / n
    
    # FIX: Clamp psi to prevent 0^(n-1) which causes NaN gradients
    psi_safe = torch.clamp(torch.abs(psi), min=1e-8)
    
    C = torch.where(
        psi < 0,
        (theta_s - theta_r) * m * n * alpha
        * (alpha * psi_safe) ** (n - 1)
        / (1.0 + (alpha * psi_safe) ** n) ** (m + 1),
        torch.zeros_like(psi),
    )
    # Ensure capacity doesn't drop to absolute zero, which can stall Richards' equation
    return C.clamp(min=1e-8)
