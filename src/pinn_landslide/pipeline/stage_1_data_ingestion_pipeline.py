import torch
import os,sys
from src.pinn_landslide.utils.utils import read_yaml,create_directories
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.components.data_ingestion import DataIngestion
from src.pinn_landslide.constants import *
from src.pinn_landslide.exception.exception import customexception


class DataIngestionPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.device = self.config.get_training_config().device
   
    def run(self):
        try:
            geo_config = self.config.get_geo_params_config()
            data_ingestion = DataIngestion(device=self.device, geo_config=geo_config)
            df = data_ingestion.load_hydrus_data()
            df = data_ingestion.add_FS_feature(df)

        except Exception as e:
            raise customexception(e, sys)