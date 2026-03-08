import pandas as pd
import torch
import numpy as np
from pathlib import Path
from src.pinn_landslide.config.configuration import ConfigurationManager

from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.constants import *
import math


class Dataloader():
    def __init__(self, device: torch.device ,config: Path = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
        self.device = device
    

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_ingestion.ingested_data)
        return df

    
    
    def extract_failure_points(self, df: pd.DataFrame):
        
        failing_rows = df[(df['FS'] <= 1.0) & (df['Depth_m'] > 0.1)]
        
        if len(failing_rows) == 0:
            print("Warning: Your HYDRUS simulation never generated enough pore pressure to fail the slope!")
            return None, None, None
            
        # 4. Convert failing rows to Tensors
        z_fail = torch.tensor(failing_rows['Depth_m'].values, dtype=torch.float32, device=self.device).view(-1, 1)
        t_fail = torch.tensor(failing_rows['Time_days'].values, dtype=torch.float32, device=self.device).view(-1, 1)
        fs_fail = torch.tensor(failing_rows['FS'].values, dtype=torch.float32, device=self.device).view(-1, 1)
        
        return z_fail, t_fail, fs_fail
    
    def get_real_batch(self, df: pd.DataFrame):
        """
        Converts the HYDRUS DataFrame into the PINN dictionary batch.
        """
        # 1. Convert DataFrame columns to PyTorch Tensors
        # .view(-1, 1) ensures the shape is (N, 1) which the neural network requires
        t_data = torch.tensor(df['Time_days'].values, dtype=torch.float32, device=self.device).view(-1, 1)
        z_data = torch.tensor(df['Depth_m'].values, dtype=torch.float32, device=self.device).view(-1, 1)
        psi_data = torch.tensor(df['Pressure_Head'].values, dtype=torch.float32, device=self.device).view(-1, 1)

        # 2. Generate Random Collocation Points (Physics points)
        # We find the min/max of your actual data to bound the physics engine
        t_min, t_max = df['Time_days'].min(), df['Time_days'].max()
        z_min, z_max = df['Depth_m'].min(), df['Depth_m'].max()
        
        N_coll = 5000 # Number of random physics points you want to evaluate
        
        # torch.rand generates between 0 and 1, so we scale it to your min/max boundaries
        t_coll = (t_max - t_min) * torch.rand(N_coll, 1, device=self.device) + t_min
        z_coll = (z_max - z_min) * torch.rand(N_coll, 1, device=self.device) + z_min
        
        # REQUIRED: Set requires_grad=True so torch.autograd can calculate dpsi_dz and dpsi_dt
        t_coll.requires_grad_(True)
        z_coll.requires_grad_(True)

        z_fail, t_fail, fs_fail = self.extract_failure_points(df)
        # 3. Build the batch dictionary
        batch = {
            'z_coll': z_coll,
            't_coll': t_coll,
            
            'z_data': z_data,
            't_data': t_data,
            'psi_data': psi_data,
            
            # (We will leave the failure points empty for now until we identify them)
            'z_fail': z_fail,
            't_fail': t_fail,
            'fs_fail': fs_fail
        }
        self.save_batch(batch)
        return batch
    

    def save_batch(self, batch: dict):
        path = self.config.data_ingestion.batch_path
        torch.save(batch, path)
        print(f"Batch saved at {path}")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = Dataloader(device=device)
    df = dataloader.get_dataframe()
    batch = dataloader.get_real_batch(df)       
