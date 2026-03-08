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
    def __init__(self, device: torch.device, geo_config: GeoParamsConfig ,config: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
        self.device = device
        self.geo_config = geo_config
        
        create_directories([f"{self.config.data_ingestion.root}"])

    def load_hydrus_data(self) -> pd.DataFrame:
        """Parses a HYDRUS-1D Nod_Inf.out file into a clean Pandas DataFrame."""
        parsed_data = []
        current_time = None
        self.file_path = self.config.data_ingestion.raw_data_file

        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # 1. Catch the Time step
                if line.startswith("Time:"):
                    # Extracts the number after "Time:"
                    current_time = float(line.split()[1])
                    continue
                    
                # 2. Catch the Data rows (they always start with a Node number)
                if line and line[0].isdigit():
                    parts = line.split()
                    # Ensure we have a valid data row (at least Node, Depth, Head, Moisture)
                    if len(parts) >= 4 and current_time is not None:
                        depth = float(parts[1])
                        head = float(parts[2])
                        
                        # Store the row
                        parsed_data.append([current_time, depth, head])
                        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_data, columns=['Time_days', 'Depth_m', 'Pressure_Head'])
        
        # CRITICAL: Convert negative HYDRUS depths to absolute positive thickness for PINN math
        df['Depth_m'] = np.abs(df['Depth_m'])
        return df
    
    def add_FS_feature(self, df: pd.DataFrame):
        # 1. Use your geotechnical parameters
        gamma_w = self.geo_config.gamma_w
        gamma_dry = self.geo_config.gamma - (self.geo_config.theta_s * gamma_w) 
        c_prime = self.geo_config.c_prime
        phi_rad = math.radians(self.geo_config.phi_prime)
        beta_rad = math.radians(self.geo_config.beta)
        
        # 2. Calculate simplified FS for the dataframe
        positive_head = np.maximum(df["Pressure_Head"], 0)
        u = positive_head * gamma_w
        
        dynamic_weight = gamma_dry + (df["Moisture"] * gamma_w) if "Moisture" in df else self.geo_config.gamma
        abs_depth = df['Depth_m']
        
        sigma = dynamic_weight * abs_depth * (np.cos(beta_rad) ** 2)
        tau = dynamic_weight * abs_depth * np.sin(beta_rad) * np.cos(beta_rad)
        
        effective_stress = np.maximum(sigma - u, 0)
        
        # Add tiny epsilon to tau to prevent division by zero
        df['FS'] = (c_prime + effective_stress * np.tan(phi_rad)) / np.maximum(tau, 1e-9)
        
        df.to_csv(self.config.data_ingestion.ingested_data, index=False)
        return df
    
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_manager = ConfigurationManager()
    geo_config = config_manager.get_geo_params_config()

    data_ingestion = DataIngestion(device=device, geo_config=geo_config)
    df = data_ingestion.load_hydrus_data()
    df = data_ingestion.add_FS_feature(df)

    print(df.head())