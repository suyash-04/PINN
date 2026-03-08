"""
components/loss_functions.py

Multi-component loss function for the PINN as defined in Section 5.4 of the proposal.

L_total = lambda_physics * L_physics
        + lambda_anchor  * L_anchor
        + lambda_initial * L_initial
        + lambda_boundary* L_boundary
        + lambda_failure * L_failure
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict

# Note: Ensure these entities exist in your src/pinn_landslide/entity/__init__.py
from pinn_landslide.entity import GeotechnicalParams, TrainingConfig
from pinn_landslide.logger import logger


# ── Van Genuchten helpers (PyTorch — differentiable) ─────────────────────────

def _vg_Se(psi: torch.Tensor, alpha: float, n: float) -> torch.Tensor:
    """Effective saturation Se(psi). psi in metres."""
    m = 1.0 - 1.0 / n
    return torch.where(
        psi < 0,
        (1.0 + (alpha * torch.abs(psi)) ** n) ** (-m),
        torch.ones_like(psi),
    )

def _vg_K(psi: torch.Tensor, Ks: float, alpha: float, n: float, l: float = 0.5) -> torch.Tensor:
    """Unsaturated hydraulic conductivity K(psi)."""
    m = 1.0 - 1.0 / n
    Se = _vg_Se(psi, alpha, n).clamp(1e-8, 1.0)
    inner = (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
    return Ks * Se ** l * inner

def _vg_C(psi: torch.Tensor, theta_r: float, theta_s: float, alpha: float, n: float) -> torch.Tensor:
    """Specific moisture capacity C = dtheta/dpsi."""
    m = 1.0 - 1.0 / n
    C = torch.where(
        psi < 0,
        (theta_s - theta_r) * m * n * alpha
        * (alpha * torch.abs(psi)) ** (n - 1)
        / (1.0 + (alpha * torch.abs(psi)) ** n) ** (m + 1),
        torch.zeros_like(psi),
    )
    return C.clamp(min=1e-8)


# ── Loss components ───────────────────────────────────────────────────────────

@dataclass
class LossBreakdown:
    """Data class to track individual loss components during training."""
    physics: torch.Tensor
    anchor: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor
    failure: torch.Tensor
    total: torch.Tensor

    def as_dict(self) -> Dict[str, float]:
        return {k: v.item() for k, v in self.__dict__.items()}


class PINNLossFunction:
    """
    Computes the weighted multi-component PINN loss.

    Parameters
    ----------
    geo    : GeotechnicalParams — van Genuchten & stability parameters
    config : TrainingConfig     — lambda weights
    depth_m: float              — physical domain depth (for de-normalisation)
    t_hrs_window: float         — duration of PINN training window (hours)
    gamma_w: float              — unit weight of water (kN/m^3)
    """

    def __init__(
        self,
        geo: GeotechnicalParams,
        config: TrainingConfig,
        depth_m: float,
        t_hrs_window: float,
        gamma_w: float = 9.81,
    ):
        self.geo = geo
        self.cfg = config
        self.depth_m = depth_m
        self.t_win = t_hrs_window
        self.gamma_w = gamma_w
        self._beta = math.radians(geo.beta)
        self._phi = math.radians(geo.phi_prime)

    # ── Main entry point ──────────────────────────────────────────────────────

    def compute(
        self,
        model: nn.Module,
        coll_z: torch.Tensor,  coll_t: torch.Tensor,     # collocation
        anch_z: torch.Tensor,  anch_t: torch.Tensor,     # anchor points
        anch_psi: torch.Tensor,                          # HYDRUS target
        ic_z: torch.Tensor,    ic_psi: torch.Tensor,     # initial condition
        bc_t: torch.Tensor,    bc_flux: torch.Tensor,    # boundary
        fail_z: torch.Tensor,  t_fail_norm: float,       # failure event
    ) -> LossBreakdown:

        L_phy = self._physics_loss(model, coll_z, coll_t)
        L_anc = self._anchor_loss(model, anch_z, anch_t, anch_psi)
        L_ini = self._initial_loss(model, ic_z, ic_psi)
        L_bnd = self._boundary_loss(model, bc_t, bc_flux)
        L_fai = self._failure_loss(model, fail_z, t_fail_norm)

        total = (
            self.cfg.lambda_physics  * L_phy
            + self.cfg.lambda_anchor   * L_anc
            + self.cfg.lambda_initial  * L_ini
            + self.cfg.lambda_boundary * L_bnd
            + self.cfg.lambda_failure  * L_fai
        )
        return LossBreakdown(L_phy, L_anc, L_ini, L_bnd, L_fai, total)

    # ── L_physics (Richards Equation Residual) ────────────────────────────────

    def _physics_loss(
        self, model: nn.Module,
        z_n: torch.Tensor, t_n: torch.Tensor,
    ) -> torch.Tensor:
        """Richards equation residual."""
        psi, dpsi_dz_n, dpsi_dt_n, d2psi_dz2_n = model.predict_with_gradients(z_n, t_n)

        # De-normalise derivatives using the chain rule
        # Assumes z is positive downwards from the surface (0 to 30m)
        dpsi_dz   = dpsi_dz_n   / self.depth_m               # m/m
        dpsi_dt   = dpsi_dt_n   / (self.t_win * 3600.0)      # m/s
        d2psi_dz2 = d2psi_dz2_n / (self.depth_m ** 2)        # m/m^2

        g = self.geo
        C   = _vg_C(psi, g.theta_r, g.theta_s, g.alpha, g.n)
        K   = _vg_K(psi, g.Ks, g.alpha, g.n, g.l)
        
        # Calculate dK/dpsi for the expanded spatial derivative
        dK_dpsi = torch.autograd.grad(
            K, psi,
            grad_outputs=torch.ones_like(K),
            create_graph=True, retain_graph=True,
        )[0] if psi.requires_grad else torch.zeros_like(psi)

        # RHS of Richards: d/dz [K*(dpsi/dz + 1)] = dK/dpsi * dpsi/dz * (dpsi/dz + 1) + K * d2psi/dz2
        rhs = dK_dpsi * dpsi_dz * (dpsi_dz + 1.0) + K * d2psi_dz2

        residual = C * dpsi_dt - rhs
        return torch.mean(residual ** 2)

    # ── L_anchor (Data Alignment) ─────────────────────────────────────────────

    def _anchor_loss(
        self, model: nn.Module,
        z_n: torch.Tensor, t_n: torch.Tensor, psi_target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between PINN predictions and HYDRUS-1D anchor points."""
        psi_pred = model(z_n, t_n)
        
        # Critical safety check to prevent (N, 1) and (N,) broadcasting bugs
        psi_target = psi_target.view_as(psi_pred)
        
        return torch.mean((psi_pred - psi_target) ** 2)

    # ── L_initial (Initial State Match) ───────────────────────────────────────

    def _initial_loss(
        self, model: nn.Module,
        z_n: torch.Tensor, psi_ic: torch.Tensor,
    ) -> torch.Tensor:
        """Anchors the pressure head at t=0 to the initial condition profile."""
        t_zero = torch.zeros_like(z_n)
        psi_pred = model(z_n, t_zero)
        
        psi_ic = psi_ic.view_as(psi_pred)
        
        return torch.mean((psi_pred - psi_ic) ** 2)

    # ── L_boundary (Surface Flux Match) ───────────────────────────────────────

    def _boundary_loss(
        self, model: nn.Module,
        t_n: torch.Tensor,          # normalised time at surface
        I_ms: torch.Tensor,         # rainfall intensity (m/s) at those times
    ) -> torch.Tensor:
        """Surface flux: q_pred = -K(psi) * (dpsi/dz + 1) matched to rainfall I(t)."""
        z_surf = torch.zeros_like(t_n)
        
        # Extract gradients right at the surface
        psi_surf, dpsi_dz_n, _, _ = model.predict_with_gradients(z_surf, t_n)
        dpsi_dz = dpsi_dz_n / self.depth_m
        
        K_surf = _vg_K(psi_surf, self.geo.Ks, self.geo.alpha, self.geo.n, self.geo.l)
        
        # Calculate Darcy flux (m/s). Downward flow is positive.
        q_pred = -K_surf * (dpsi_dz + 1.0)                
        
        I_ms = I_ms.view_as(q_pred)
        
        return torch.mean((q_pred + I_ms) ** 2)            

    # ── L_failure (Event-Based Stability) ─────────────────────────────────────

    def _failure_loss(
        self, model: nn.Module,
        fail_z_n: torch.Tensor,     # normalised depths near failure plane
        t_fail_n: float,            # normalised time of known failure event
    ) -> torch.Tensor:
        """Soft hinge: penalise FS > 1 at the documented failure time."""
        t_f = torch.full_like(fail_z_n, t_fail_n)
        psi_pred = model(fail_z_n, t_f)                    # Shape: (Nf, 1)

        # De-normalise depth back to physical meters
        z_phys = fail_z_n * self.depth_m                   # Shape: (Nf, 1)

        # Pore water pressure (kPa) 
        u = torch.clamp(psi_pred, min=0.0) * self.gamma_w  

        # Normal stress (kPa): sigma = gamma * z * cos^2(beta)
        sigma = self.geo.gamma * z_phys * (math.cos(self._beta) ** 2)

        # Factor of Safety (infinite-slope model)
        numerator   = self.geo.c_prime + (sigma - u) * math.tan(self._phi)
        denominator = sigma * math.sin(self._beta) * math.cos(self._beta) + 1e-8
        FS = numerator / denominator                       

        # Soft hinge penalty: max(0, 1 - FS)^2 
        # Only penalizes if the model thinks the slope is safe (FS > 1) at failure time
        penalty = torch.clamp(1.0 - FS, min=0.0) ** 2
        
        return torch.mean(penalty)