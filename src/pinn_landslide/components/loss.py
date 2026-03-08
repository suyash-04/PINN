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

        self.gamma_w = 9.81
        self.beta_rad = math.radians(self.geo_params.beta)
        self.phi_rad = math.radians(self.geo_params.phi_prime)

    # ── 1. PDE Physics Loss ─────────────────────────────
    def physics_loss(self, z_norm: torch.Tensor, t_norm: torch.Tensor, norm_params: dict) -> torch.Tensor:
        psi_norm, dpsi_dz_n, dpsi_dt_n, d2psi_dz2_n = self.model.predict_with_gradients(z_norm, t_norm)

        # Extract limits
        z_max, t_max = norm_params['z_max'], norm_params['t_max']
        psi_min, psi_max = norm_params['psi_min'], norm_params['psi_max']
        psi_range = psi_max - psi_min

        # DE-NORMALIZE for physical equations
        psi_phys = (psi_norm * psi_range) + psi_min
        dpsi_dz = (dpsi_dz_n * psi_range) / z_max
        dpsi_dt = (dpsi_dt_n * psi_range) / (t_max * 24 * 3600.0) # assuming t_max is in days, convert to seconds
        d2psi_dz2 = (d2psi_dz2_n * psi_range) / (z_max ** 2)

        # Calculate hydraulic properties using PHYSICAL pressure head
        C = _vg_C(psi_phys, self.geo_params.theta_r, self.geo_params.theta_s, self.geo_params.alpha, self.geo_params.n)
        K = _vg_K(psi_phys, self.geo_params.Ks, self.geo_params.alpha, self.geo_params.n, self.geo_params.l)

        dK_dz = torch.autograd.grad(
            outputs=K, inputs=z_norm,
            grad_outputs=torch.ones_like(K),
            create_graph=True, retain_graph=True
        )[0] / z_max  # De-normalize gradient

        # Richards Equation (1D Vertical): d/dz [K * (dpsi/dz + 1)]
        LHS = C * dpsi_dt
        RHS = dK_dz * (dpsi_dz + 1.0) + K * d2psi_dz2
        
        return torch.mean((LHS - RHS) ** 2)

    # ── 2. Anchor Loss ───────────────────────────────────────────
    def anchor_loss(self, z_data: torch.Tensor, t_data: torch.Tensor, psi_data_norm: torch.Tensor) -> torch.Tensor:
        if z_data is None or len(z_data) == 0: return torch.tensor(0.0, device=next(self.model.parameters()).device)
        psi_pred_norm = self.model(z_data, t_data)
        return torch.mean((psi_pred_norm - psi_data_norm.view_as(psi_pred_norm)) ** 2)

    # ── 3. Boundary Loss (Flux Matching) ──────────────────────────────────
    def boundary_loss(self, z_bc: torch.Tensor, t_bc: torch.Tensor, target_flux: torch.Tensor, norm_params: dict) -> torch.Tensor:
        if z_bc is None or len(z_bc) == 0 or target_flux is None: return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        psi_norm, dpsi_dz_n, _, _ = self.model.predict_with_gradients(z_bc, t_bc)
        psi_phys = (psi_norm * (norm_params['psi_max'] - norm_params['psi_min'])) + norm_params['psi_min']
        dpsi_dz = (dpsi_dz_n * (norm_params['psi_max'] - norm_params['psi_min'])) / norm_params['z_max']
        
        K_surf = _vg_K(psi_phys, self.geo_params.Ks, self.geo_params.alpha, self.geo_params.n, self.geo_params.l)
        
        # Darcy Flux: q = -K * (dpsi/dz + 1)
        q_pred = -K_surf * (dpsi_dz + 1.0)
        return torch.mean((q_pred + target_flux.view_as(q_pred)) ** 2)

    # ── 4. Failure Loss (Soft Hinge Penalty) ──────────────────────
    def failure_loss(self, z_fail_n: torch.Tensor, t_fail_n: torch.Tensor, norm_params: dict) -> torch.Tensor:
        if z_fail_n is None or len(z_fail_n) == 0: return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        psi_norm = self.model(z_fail_n, t_fail_n)
        psi_phys = (psi_norm * (norm_params['psi_max'] - norm_params['psi_min'])) + norm_params['psi_min']
        z_phys = z_fail_n * norm_params['z_max']
        
        # Pore pressure (u)
        u = torch.relu(psi_phys) * self.gamma_w

        # Effective Saturation and dynamic unit weight
        Se = _vg_Se(psi_phys, self.geo_params.alpha, self.geo_params.n)
        theta = self.geo_params.theta_r + Se * (self.geo_params.theta_s - self.geo_params.theta_r)
        dynamic_weight = self.geo_params.gamma + (theta * self.gamma_w)

        sigma = dynamic_weight * z_phys * (math.cos(self.beta_rad) ** 2)
        tau = dynamic_weight * z_phys * math.sin(self.beta_rad) * math.cos(self.beta_rad)
        effective_stress = torch.relu(sigma - u)

        numerator = self.geo_params.c_prime + (effective_stress * math.tan(self.phi_rad))
        denominator = torch.clamp(tau, min=1e-9) 
        fs_pred = numerator / denominator

        # Soft hinge penalty: penalizes if FS > 1.0 at failure time
        penalty = torch.clamp(1.0 - fs_pred, min=0.0) ** 2
        return torch.mean(penalty)

    # ── Master Forward Pass ──────────────────────────────────────────────────
    def forward(self, batch: dict) -> torch.Tensor:
        loss_pde = self.physics_loss(batch['z_coll'], batch['t_coll'], batch['norm_params'])
        loss_data = self.anchor_loss(batch.get('z_data'), batch.get('t_data'), batch.get('psi_data'))
        loss_fs = self.failure_loss(batch.get('z_fail'), batch.get('t_fail'), batch['norm_params'])
        
        # Add loss_bc and loss_ic conditionally if you add them to your batch later
        loss_bc = torch.tensor(0.0, device=loss_pde.device) 
        loss_ic = torch.tensor(0.0, device=loss_pde.device)

        total_loss = (
            self.loss_config.lambda_physics * loss_pde +
            self.loss_config.lambda_anchor * loss_data +
            self.loss_config.lambda_failure * loss_fs +
            self.loss_config.lambda_boundary * loss_bc +
            self.loss_config.lambda_initial * loss_ic 
        )
        return total_loss