import torch
import torch.nn as nn
import math
from src.pinn_landslide.config.configuration import PINNLossConfig, GeoParamsConfig
from src.pinn_landslide.utils.utils import _vg_C, _vg_K, _vg_Se
class CustomLoss(nn.Module):
    def __init__(self, loss_config: 'PINNLossConfig', geo_params: 'GeoParamsConfig', model: nn.Module):
        super(CustomLoss, self).__init__()
        self.loss_config = loss_config
        self.geo_params = geo_params
        self.model = model

        # Constants for geotechnical math
        self.gamma_w = 9.81
        self.beta_rad = math.radians(self.geo_params.beta)
        self.phi_rad = math.radians(self.geo_params.phi_prime)

    # ── 1. PDE Physics Loss (Richards' Equation) ─────────────────────────────
    def physics_loss(self, z_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        psi, dpsi_dz, dpsi_dt, d2psi_dz2 = self.model.predict_with_gradients(z_norm, t_norm)

        C = _vg_C(psi, self.geo_params.theta_r, self.geo_params.theta_s, self.geo_params.alpha, self.geo_params.n)
        K = _vg_K(psi, self.geo_params.Ks, self.geo_params.alpha, self.geo_params.n, self.geo_params.l)

        dK_dz = torch.autograd.grad(
            outputs=K, inputs=z_norm,
            grad_outputs=torch.ones_like(K),
            create_graph=True, retain_graph=True
        )[0]

        LHS = C * dpsi_dt
        RHS = dK_dz * (dpsi_dz - math.cos(self.beta_rad)) + K * d2psi_dz2
        
        return torch.mean((LHS - RHS) ** 2)

    # ── 2. Anchor Loss (Data Loss) ───────────────────────────────────────────
    def anchor_loss(self, z_data: torch.Tensor, t_data: torch.Tensor, psi_true: torch.Tensor) -> torch.Tensor:
        """Penalizes mismatch between predicted head and known sensor/HYDRUS data."""
        if z_data is None or len(z_data) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        psi_pred = self.model(z_data, t_data)
        return torch.mean((psi_pred - psi_true) ** 2)

    # ── 3. Initial Condition Loss ────────────────────────────────────────────
    def initial_loss(self, z_ic: torch.Tensor, t_ic: torch.Tensor, psi_ic_true: torch.Tensor) -> torch.Tensor:
        """Ensures the network starts at the correct dry/wet state at t=0."""
        if z_ic is None or len(z_ic) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        psi_pred = self.model(z_ic, t_ic)
        return torch.mean((psi_pred - psi_ic_true) ** 2)

    # ── 4. Boundary Condition Loss ───────────────────────────────────────────
    def boundary_loss(self, z_bc: torch.Tensor, t_bc: torch.Tensor, psi_bc_true: torch.Tensor) -> torch.Tensor:
        """Enforces surface rainfall infiltration or bottom drainage boundaries."""
        if z_bc is None or len(z_bc) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        psi_pred = self.model(z_bc, t_bc)
        return torch.mean((psi_pred - psi_bc_true) ** 2)

    # ── 5. Failure Loss (Geotechnical Factor of Safety) ──────────────────────
    def _calculate_fs(self, z: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Helper to calculate FS tensors purely in PyTorch."""
        # Absolute depth
        z_abs = torch.abs(z)

        # Pore pressure (u): torch.relu automatically does max(psi, 0)
        u = torch.relu(psi) * self.gamma_w

        # Dynamic moisture (theta) using effective saturation (Se)
        Se = _vg_Se(psi, self.geo_params.alpha, self.geo_params.n)
        theta = self.geo_params.theta_r + Se * (self.geo_params.theta_s - self.geo_params.theta_r)
        
        # Dynamic unit weight (assuming geo_params.gamma is the dry unit weight)
        dynamic_weight = self.geo_params.gamma + (theta * self.gamma_w)

        # Stresses
        sigma = dynamic_weight * z_abs * (math.cos(self.beta_rad) ** 2)
        tau = dynamic_weight * z_abs * math.sin(self.beta_rad) * math.cos(self.beta_rad)

        # Effective stress (clamped to prevent liquefaction bug)
        effective_stress = torch.relu(sigma - u)

        # Resisting and Driving forces
        numerator = self.geo_params.c_prime + (effective_stress * math.tan(self.phi_rad))
        denominator = torch.clamp(tau, min=1e-9) # Prevent division by zero at surface

        # Mask surface node to NaN, else calculate FS
        fs = torch.where(z_abs < 1e-5, torch.tensor(float('nan')), numerator / denominator)
        return fs

    def failure_loss(self, z_fail: torch.Tensor, t_fail: torch.Tensor, fs_true: torch.Tensor) -> torch.Tensor:
        """Penalizes the model if its predicted FS doesn't match the known failure state."""
        if z_fail is None or len(z_fail) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        psi_pred = self.model(z_fail, t_fail)
        fs_pred = self._calculate_fs(z_fail, psi_pred)
        
        # Filter out NaN values (surface nodes) before calculating MSE
        valid_mask = ~torch.isnan(fs_pred)
        if not valid_mask.any():
            return torch.tensor(0.0, device=fs_pred.device)
            
        return torch.mean((fs_pred[valid_mask] - fs_true[valid_mask]) ** 2)

    # ── Master Forward Pass ──────────────────────────────────────────────────
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Combines all losses. Expects a dictionary of tensors to keep inputs clean.
        """
        loss_pde = self.physics_loss(batch['z_coll'], batch['t_coll'])
        
        loss_data = self.anchor_loss(batch.get('z_data'), batch.get('t_data'), batch.get('psi_data'))
        loss_ic = self.initial_loss(batch.get('z_ic'), batch.get('t_ic'), batch.get('psi_ic'))
        loss_bc = self.boundary_loss(batch.get('z_bc'), batch.get('t_bc'), batch.get('psi_bc'))
        loss_fs = self.failure_loss(batch.get('z_fail'), batch.get('t_fail'), batch.get('fs_fail'))

        # Weighted sum
        total_loss = (
            self.loss_config.lambda_physics * loss_pde +
            self.loss_config.lambda_anchor * loss_data +
            self.loss_config.lambda_initial * loss_ic +
            self.loss_config.lambda_boundary * loss_bc +
            self.loss_config.lambda_failure * loss_fs
        )

        return total_loss