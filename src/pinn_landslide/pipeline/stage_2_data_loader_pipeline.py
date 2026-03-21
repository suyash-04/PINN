import torch
import os,sys
from src.pinn_landslide.utils.utils import read_yaml,create_directories
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.components.data_loader import Dataloader
from src.pinn_landslide.constants import *
from src.pinn_landslide.exception.exception import customexception



class DataLoaderPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.device = self.config.get_training_config().device
  
    
    def run(self):

        try:
            dataloader = Dataloader(device=self.device)
            df = dataloader.get_dataframe()
            dataloader.get_real_batch(df)

        except Exception as e:
            raise customexception(e, sys)
    