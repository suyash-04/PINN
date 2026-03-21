import pandas as pd
import torch
import numpy as np
from pathlib import Path
from src.pinn_landslide.constants import *
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.entity.config_entity import GeoParamsConfig
from src.pinn_landslide.utils.utils import create_directories, read_yaml
import math


class DataIngestion():
    def __init__(self, device: torch.device, geo_config: GeoParamsConfig,
                 config: Path = CONFIG_FILE_PATH):
        self.config     = read_yaml(config)
        self.device     = device
        self.geo_config = geo_config
        create_directories([f"{self.config.data_ingestion.root}"])

    def load_hydrus_data(self) -> pd.DataFrame:
        """
        Parse HYDRUS-1D Nod_Inf output file.
        Time units in data.out are HOURS (T = hours in file header).
        Column written: Time_hours  (not Time_days).
        """
        parsed_data  = []
        current_time = None
        self.file_path = self.config.data_ingestion.raw_data_file

        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Time:"):
                    current_time = float(line.split()[1])
                    continue
                if line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 4 and current_time is not None:
                        depth = float(parts[1])
                        head  = float(parts[2])
                        parsed_data.append([current_time, depth, head])

        df = pd.DataFrame(
            parsed_data,
            columns=['Time_hours', 'Depth_m', 'Pressure_Head']
        )
        df['Depth_m'] = np.abs(df['Depth_m'])
        return df

    def add_FS_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Factor of Safety using the infinite-slope model and
        save the processed CSV.  Column Time_hours is preserved.
        """
        gamma_w  = self.geo_config.gamma_w
        c_prime  = self.geo_config.c_prime
        phi_rad  = math.radians(self.geo_config.phi_prime)
        beta_rad = math.radians(self.geo_config.beta)

        positive_head  = np.maximum(df['Pressure_Head'], 0)
        u              = positive_head * gamma_w

        if 'Moisture' in df.columns:
            dynamic_weight = self.geo_config.gamma + (df['Moisture'] * gamma_w)
        else:
            dynamic_weight = self.geo_config.gamma

        abs_depth        = df['Depth_m']
        sigma            = dynamic_weight * abs_depth * (np.cos(beta_rad) ** 2)
        tau              = dynamic_weight * abs_depth * np.sin(beta_rad) * np.cos(beta_rad)
        effective_stress = np.maximum(sigma - u, 0)

        df['FS'] = (c_prime + effective_stress * np.tan(phi_rad)) / np.maximum(tau, 1e-9)

        df.to_csv(self.config.data_ingestion.ingested_data, index=False)
        return df