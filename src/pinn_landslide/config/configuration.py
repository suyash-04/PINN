import os, sys
from src.pinn_landslide.constants import *
from src.pinn_landslide.utils.utils import read_yaml,create_directories
from src.pinn_landslide.entity.config_entity import PINNArchConfig, PINNLossConfig, GeoParamsConfig, TrainingConfig
from src.pinn_landslide.exception.exception import customexception
from src.pinn_landslide.logger.logger import logger

class ConfigurationManager():
    def __init__(self, CONFIG_FILE_PATH =CONFIG_FILE_PATH , PARAMS_FILE_PATH = PARAMS_FILE_PATH):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        create_directories([self.config.artifacts_root])
    

    def get_pinn_arch_config(self) -> PINNArchConfig:
    
        print(f"Loading parameters from {self.params}")
        a = self.params['architecture']
        logger.info(f"PINN Architecture Config: {a}")
        return PINNArchConfig(
                n_hidden_layers=a["n_hidden_layers"],
                neurons_per_layer=a["neurons_per_layer"],
                activation=a["activation"],
                input_dim=a["input_dim"],
                output_dim=a["output_dim"]
            )
    
    def get_geo_params_config(self) -> GeoParamsConfig:
        print(f"Loading parameters from {self.params}")
        g = self.params['geo_params']
        logger.info(f"Geotechnical Parameters Config: {g}")
        return GeoParamsConfig(
                Ks=g["Ks"],
                theta_s=g["theta_s"],
                theta_r=g["theta_r"],
                alpha=g["alpha"],
                n=g["n"],
                l=g["l"],
                c_prime=g["c_prime"],
                phi_prime=g["phi_prime"],
                gamma=g["gamma"],
                beta=g["beta"],
                gamma_w=g["gamma_w"]
            )
    def get_loss_weights_config(self) -> PINNLossConfig:
        print(f"Loading parameters from {self.params}")
        w = self.params['loss_weights']
        logger.info(f"Loss Weights Config: {w}")
        return PINNLossConfig(
                lambda_physics=w["lambda_physics"],
                lambda_anchor=w["lambda_anchor"],
                lambda_initial=w["lambda_initial"],
                lambda_boundary=w["lambda_boundary"],
                lambda_failure=w["lambda_failure"]
            )

    def get_training_config(self) -> TrainingConfig:
        print(f"Loading parameters from {self.params}") 
        t = self.params['training']
        logger.info(f"Training Config loaded: {t}")
        return TrainingConfig(
            epochs=t['epochs'],
            learning_rate=t['learning_rate'],
            device=t['device']
        )