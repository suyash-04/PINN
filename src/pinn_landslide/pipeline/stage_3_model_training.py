import torch
import os,sys
from src.pinn_landslide.utils.utils import read_yaml,create_directories
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.components.trainer import PINNTrainer
from src.pinn_landslide.constants import *
from src.pinn_landslide.exception.exception import customexception


class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.device = self.config.get_training_config().device

    def run(self):
        try:
            trainer = PINNTrainer(device=self.device)
            trainer.train()

        except Exception as e:
            raise customexception(e, sys)
