import pandas as pd
import torch
import numpy as np
from pathlib import Path
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.constants import *

class Dataloader():
    def __init__(self, device: torch.device, config: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
        self.device = device

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_ingestion.ingested_data)
        return df

    def extract_failure_points(self, df: pd.DataFrame, z_max: float, t_max: float):
        failing_rows = df[(df['FS'] <= 1.0) & (df['Depth_m'] > 0.1)]
        
        if len(failing_rows) == 0:
            print("Warning: HYDRUS simulation never generated enough pore pressure to fail the slope!")
            return None, None
            
        # Normalize failure points before sending to PINN
        z_fail_norm = torch.tensor(failing_rows['Depth_m'].values / z_max, dtype=torch.float32, device=self.device).view(-1, 1)
        t_fail_norm = torch.tensor(failing_rows['Time_days'].values / t_max, dtype=torch.float32, device=self.device).view(-1, 1)
        
        return z_fail_norm, t_fail_norm
    
    def get_real_batch(self, df: pd.DataFrame):
        # 1. Find Normalization Boundaries
        t_max = float(df['Time_days'].max())
        z_max = float(df['Depth_m'].max())
        psi_min = float(df['Pressure_Head'].min())
        psi_max = float(df['Pressure_Head'].max())

        # 2. Convert to NORMALIZED PyTorch Tensors [0, 1]
        t_data = torch.tensor(df['Time_days'].values / t_max, dtype=torch.float32, device=self.device).view(-1, 1)
        z_data = torch.tensor(df['Depth_m'].values / z_max, dtype=torch.float32, device=self.device).view(-1, 1)
        psi_data = torch.tensor((df['Pressure_Head'].values - psi_min) / (psi_max - psi_min), dtype=torch.float32, device=self.device).view(-1, 1)

        # 3. Generate Random Collocation Points in [0, 1] range
        N_coll = 10000 
        t_coll = torch.rand(N_coll, 1, device=self.device, requires_grad=True)
        z_coll = torch.rand(N_coll, 1, device=self.device, requires_grad=True)

        z_fail, t_fail = self.extract_failure_points(df, z_max, t_max)
        
        # 4. Build the batch dictionary including normalization parameters
        batch = {
            'z_coll': z_coll,
            't_coll': t_coll,
            'z_data': z_data,
            't_data': t_data,
            'psi_data': psi_data,
            'z_fail': z_fail,
            't_fail': t_fail,
            'norm_params': {
                't_max': t_max,
                'z_max': z_max,
                'psi_min': psi_min,
                'psi_max': psi_max
            }
        }
        self.save_batch(batch)
        return batch

    def save_batch(self, batch: dict):
        path = self.config.data_ingestion.batch_path
        torch.save(batch, path)
        print(f"Batch saved at {path}")