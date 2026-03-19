import torch
import torch.nn as nn
import math
from src.pinn_landslide.config.configuration import PINNLossConfig, GeoParamsConfig
from src.pinn_landslide.utils.utils import _vg_C, _vg_K, _vg_Se


class CustomLoss(nn.Module):
    def __init__(self, loss_config: 'PINNLossConfig', geo_params: 'GeoParamsConfig', model: nn.Module):
        super(CustomLoss, self).__init__()
        self.loss_config = loss_config
        self.geo_params  = geo_params
        self.model       = model

        self.gamma_w  = 9.81
        self.beta_rad = math.radians(self.geo_params.beta)
        self.phi_rad  = math.radians(self.geo_params.phi_prime)

        # ── Physics residual characteristic scale ─────────────────────────
        # For highly unsaturated soil (ψ ~ −100 m), C(ψ) ~ 2.76e-4 m⁻¹ and
        # the time span is 123 days = 1.06e7 s.  The Richards residual in
        # physical units is therefore O(C · Δψ / Δt) ~ 1e-8 s⁻¹.
        # Dividing by this scale before squaring brings L_physics to O(1),
        # making λ_physics = 1e6 actually meaningful in the total loss.
        # Formula:  char_scale = C_characteristic * psi_range / t_max_seconds
        #   C_char   ≈ 2.76e-4  (Van Genuchten C at ψ = −100 m, Jure params)
        #   psi_range ≈ 453 m   (from HYDRUS training data)
        #   t_max_s  ≈ 1.06e7 s (123 days)
        # Result    ≈ 1.18e-8   → stored as self._phys_scale
        #
        # If you change soil parameters significantly, recompute C_char and
        # update _phys_scale accordingly, or set it to None to skip scaling.
        self._phys_scale: float = 1.18e-8   # [s⁻¹] characteristic residual

    # ── helper ───────────────────────────────────────────────────────────
    @property
    def _device(self):
        return next(self.model.parameters()).device

    # ── 1. PDE Physics Loss ───────────────────────────────────────────────
    def physics_loss(self, z_norm: torch.Tensor, t_norm: torch.Tensor,
                     norm_params: dict) -> torch.Tensor:
        psi_norm, dpsi_dz_n, dpsi_dt_n, d2psi_dz2_n = \
            self.model.predict_with_gradients(z_norm, t_norm)

        z_max, t_max     = norm_params['z_max'], norm_params['t_max']
        psi_min, psi_max = norm_params['psi_min'], norm_params['psi_max']
        psi_range        = psi_max - psi_min

        # De-normalise to physical units
        psi_phys  = (psi_norm  * psi_range) + psi_min
        dpsi_dz   = (dpsi_dz_n * psi_range) / z_max
        dpsi_dt   = (dpsi_dt_n * psi_range) / (t_max * 24 * 3600.0)
        d2psi_dz2 = (d2psi_dz2_n * psi_range) / (z_max ** 2)

        C = _vg_C(psi_phys, self.geo_params.theta_r, self.geo_params.theta_s,
                  self.geo_params.alpha, self.geo_params.n)
        K = _vg_K(psi_phys, self.geo_params.Ks, self.geo_params.alpha,
                  self.geo_params.n, self.geo_params.l)

        dK_dz = torch.autograd.grad(
            outputs=K, inputs=z_norm,
            grad_outputs=torch.ones_like(K),
            create_graph=True, retain_graph=True
        )[0] / z_max

        # Richards equation residual: C·∂ψ/∂t  −  ∂/∂z[K·(∂ψ/∂z + 1)]
        LHS = C * dpsi_dt
        RHS = dK_dz * (dpsi_dz + 1.0) + K * d2psi_dz2
        residual = LHS - RHS

        # ── Residual normalisation (key fix) ──────────────────────────────
        # For dry unsaturated soil, C ~ 1e-4 and K ~ 1e-12, making the raw
        # residual O(1e-13).  Without normalisation λ_physics = 1e6 is still
        # far too small to influence gradients.  Dividing by _phys_scale
        # brings the normalised residual to O(1), so λ_physics = 1e6 gives
        # a weighted loss of O(1e6 * 1²) = O(1e6) — clearly visible to Adam.
        # This is consistent with Depina et al. (2022) and Wang et al. (2022).
        if self._phys_scale is not None and self._phys_scale > 0:
            residual = residual / self._phys_scale

        return torch.mean(residual ** 2)

    # ── 2. Anchor Loss ────────────────────────────────────────────────────
    def anchor_loss(self, z_data: torch.Tensor, t_data: torch.Tensor,
                    psi_data_norm: torch.Tensor) -> torch.Tensor:
        if z_data is None or len(z_data) == 0:
            return torch.tensor(0.0, device=self._device)
        psi_pred_norm = self.model(z_data, t_data)
        return torch.mean((psi_pred_norm - psi_data_norm.view_as(psi_pred_norm)) ** 2)

    # ── 3. Boundary Loss (Surface Flux Matching) ──────────────────────────
    def boundary_loss(self, z_bc: torch.Tensor, t_bc: torch.Tensor,
                      target_flux: torch.Tensor, norm_params: dict) -> torch.Tensor:
        if z_bc is None or len(z_bc) == 0 or target_flux is None:
            return torch.tensor(0.0, device=self._device)

        psi_norm, dpsi_dz_n, _, _ = self.model.predict_with_gradients(z_bc, t_bc)
        psi_range = norm_params['psi_max'] - norm_params['psi_min']
        psi_phys  = (psi_norm  * psi_range) + norm_params['psi_min']
        dpsi_dz   = (dpsi_dz_n * psi_range) / norm_params['z_max']

        K_surf = _vg_K(psi_phys, self.geo_params.Ks, self.geo_params.alpha,
                       self.geo_params.n, self.geo_params.l)

        # Darcy flux: q = −K·(∂ψ/∂z + 1)
        q_pred = -K_surf * (dpsi_dz + 1.0)
        return torch.mean((q_pred + target_flux.view_as(q_pred)) ** 2)

    # ── 4. Initial Condition Loss ─────────────────────────────────────────
    def initial_condition_loss(self, z_ic: torch.Tensor, t_ic: torch.Tensor,
                               psi_ic_target: torch.Tensor) -> torch.Tensor:
        """
        Enforces ψ(z, 0) = ψ_HYDRUS(z, 0) — the pre-monsoon suction profile.
        Without this the network is free to predict any initial pressure field,
        which breaks the temporal evolution of the Richards equation.
        """
        if z_ic is None or len(z_ic) == 0:
            return torch.tensor(0.0, device=self._device)
        psi_pred_norm = self.model(z_ic, t_ic)
        return torch.mean((psi_pred_norm - psi_ic_target.view_as(psi_pred_norm)) ** 2)

    # ── 5. Failure Loss (Soft Hinge Penalty) ─────────────────────────────
    def failure_loss(self, z_fail_n: torch.Tensor, t_fail_n: torch.Tensor,
                     norm_params: dict) -> torch.Tensor:
        if z_fail_n is None or len(z_fail_n) == 0:
            return torch.tensor(0.0, device=self._device)

        psi_norm  = self.model(z_fail_n, t_fail_n)
        psi_range = norm_params['psi_max'] - norm_params['psi_min']
        psi_phys  = (psi_norm * psi_range) + norm_params['psi_min']
        z_phys    = z_fail_n * norm_params['z_max']

        u              = torch.relu(psi_phys) * self.gamma_w
        Se             = _vg_Se(psi_phys, self.geo_params.alpha, self.geo_params.n)
        theta          = self.geo_params.theta_r + Se * (self.geo_params.theta_s - self.geo_params.theta_r)
        dynamic_weight = self.geo_params.gamma + (theta * self.gamma_w)

        sigma            = dynamic_weight * z_phys * (math.cos(self.beta_rad) ** 2)
        tau              = dynamic_weight * z_phys * math.sin(self.beta_rad) * math.cos(self.beta_rad)
        effective_stress = torch.relu(sigma - u)

        numerator   = self.geo_params.c_prime + (effective_stress * math.tan(self.phi_rad))
        denominator = torch.clamp(tau, min=1e-9)
        fs_pred     = numerator / denominator

        # Soft hinge: penalise if FS > 1.0 at known failure points
        penalty = torch.clamp(1.0 - fs_pred, min=0.0) ** 2
        return torch.mean(penalty)

    # ── Master Forward Pass ───────────────────────────────────────────────
    def forward(self, batch: dict) -> tuple:
        loss_pde  = self.physics_loss(
            batch['z_coll'], batch['t_coll'], batch['norm_params'])

        loss_data = self.anchor_loss(
            batch.get('z_data'), batch.get('t_data'), batch.get('psi_data'))

        loss_fs   = self.failure_loss(
            batch.get('z_fail'), batch.get('t_fail'), batch['norm_params'])

        loss_bc   = self.boundary_loss(
            batch.get('z_bc'), batch.get('t_bc'),
            batch.get('target_flux'), batch['norm_params'])

        loss_ic   = self.initial_condition_loss(
            batch.get('z_ic'), batch.get('t_ic'), batch.get('psi_ic'))

        total_loss = (
            self.loss_config.lambda_physics  * loss_pde  +
            self.loss_config.lambda_anchor   * loss_data +
            self.loss_config.lambda_failure  * loss_fs   +
            self.loss_config.lambda_boundary * loss_bc   +
            self.loss_config.lambda_initial  * loss_ic
        )

        components = {
            'total'   : total_loss.item(),
            'physics' : loss_pde.item(),
            'anchor'  : loss_data.item(),
            'failure' : loss_fs.item(),
            'boundary': loss_bc.item(),
            'initial' : loss_ic.item(),
        }

        return total_loss, components