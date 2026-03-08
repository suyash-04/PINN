import torch
import torch.nn as nn
from typing import Tuple
from src.pinn_landslide.entity.config_entity import PINNArchConfig

class PINN(nn.Module):
    def __init__(self, arch_config: PINNArchConfig, input_dim: int = 2, output_dim: int = 1):
        super().__init__()
        self.arch_config = arch_config
        self.input_dim = input_dim
        self.output_dim = output_dim
       
        self.network = self.build_network()
        self._init_weights()

    def build_network(self) -> nn.Sequential:  
        act_class = self._get_activation() 
        layers = []
        
        layers.append(nn.Linear(self.input_dim, self.arch_config.neurons_per_layer))
        layers.append(act_class())

        for _ in range(self.arch_config.n_hidden_layers):
            layers.append(nn.Linear(self.arch_config.neurons_per_layer, self.arch_config.neurons_per_layer))
            layers.append(act_class())

        layers.append(nn.Linear(self.arch_config.neurons_per_layer, self.output_dim))
        return nn.Sequential(*layers)
    
    def _get_activation(self):
        mapping = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
        act = self.arch_config.activation.lower()
        if act not in mapping:
            raise ValueError(f"Unsupported activation: {act}. Choose from {list(mapping.keys())}")
        return mapping[act]
        
    def _init_weights(self) -> None:
        """Xavier (Glorot) initialisation — standard for PINNs with tanh."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, z_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_norm, t_norm], dim=1)
        return self.network(x)

    def predict_with_gradients(self, z_norm: torch.Tensor, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_norm = z_norm.requires_grad_(True)
        t_norm = t_norm.requires_grad_(True)

        psi = self.forward(z_norm, t_norm)

        dpsi_dz = torch.autograd.grad(psi, z_norm, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        dpsi_dt = torch.autograd.grad(psi, t_norm, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        d2psi_dz2 = torch.autograd.grad(dpsi_dz, z_norm, grad_outputs=torch.ones_like(dpsi_dz), create_graph=True, retain_graph=True)[0]

        return psi, dpsi_dz, dpsi_dt, d2psi_dz2