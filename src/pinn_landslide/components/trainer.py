import torch
import torch.optim as optim
from pathlib import Path

from src.pinn_landslide.utils.utils import read_yaml, create_directories
from src.pinn_landslide.config.configuration import ConfigurationManager
from src.pinn_landslide.components.pinn_architecture import PINN
from src.pinn_landslide.constants import *
from src.pinn_landslide.components.loss import CustomLoss
from src.pinn_landslide.logger.logger import logger

class PINNTrainer:
    def __init__(self, device, config_path: Path = CONFIG_FILE_PATH):
    
        # 1. Load Configurations
        self.config = read_yaml(config_path)
        config_manager = ConfigurationManager()
        self.arch_config = config_manager.get_pinn_arch_config()
        self.geo_config = config_manager.get_geo_params_config()
        self.loss_config = config_manager.get_loss_weights_config()
        self.train_config = config_manager.get_training_config()
    
        # 2. Set Device
        self.device = device
        logger.info(f"Using device: {self.device}")

        # 3. Initialize Model and Loss Function
        self.model = PINN(self.arch_config).to(self.device)
        self.criterion = CustomLoss(self.loss_config, self.geo_config, self.model).to(self.device)
        
        # 4. Initialize Adam Optimizer (Phase I)
        self.adam_optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        
        # 5. Initialize L-BFGS Optimizer (Phase II)
        # Note: Ensure your train_config has an 'lbfgs_epochs' or 'lbfgs_steps' attribute
        self.lbfgs_optimizer = optim.LBFGS(
            self.model.parameters(),
            max_iter=self.train_config.lbfgs_epochs, 
            history_size=50,
            line_search_fn="strong_wolfe" # Highly recommended for PINNs
        )
        
        create_directories([self.config.model_training.root])

    def load_batch(self, device):
            """
            Loads the PINN batch dictionary from disk.
            """
            path = self.config.data_ingestion.batch_path
            
            # FIX: Add weights_only=False to allow loading the numpy scalars in norm_params
            batch = torch.load(path, map_location=device)
            
            return batch
    def train(self):
        logger.info("Starting PINN Training...")
        self.model.train()

        # Load the batch ONCE into memory to prevent I/O bottlenecks during L-BFGS
        logger.info("Loading data batch into memory...")
        batch = self.load_batch(device=self.device)

        # ==========================================
        # PHASE I: Adam Optimizer (Global Search)
        # ==========================================
        logger.info("--- Phase I: Adam Optimization ---")
        # Ensure your config differentiates between adam_epochs (e.g., 10000) and lbfgs_epochs (e.g., 5000)
        for epoch in range(1, self.train_config.adam_epochs + 1):
            self.adam_optimizer.zero_grad()
            
            loss = self.criterion(batch)
            loss.backward()
            self.adam_optimizer.step()

            if epoch % 500 == 0 or epoch == 1:
                logger.info(f"Adam Epoch [{epoch}/{self.train_config.adam_epochs}] | Total Loss: {loss.item():.6e}")

        # ==========================================
        # PHASE II: L-BFGS Optimizer (Fine-Tuning)
        # ==========================================
        logger.info("--- Phase II: L-BFGS Optimization ---")
        
        # We use a mutable list to track the step count and loss inside the closure
        lbfgs_step = [0]

        def closure():
            self.lbfgs_optimizer.zero_grad()
            loss = self.criterion(batch)
            loss.backward()
            
            lbfgs_step[0] += 1
            if lbfgs_step[0] % 200 == 0 or lbfgs_step[0] == 1:
                logger.info(f"L-BFGS Step [{lbfgs_step[0]}/{self.train_config.lbfgs_epochs}] | Total Loss: {loss.item():.6e}")
            
            return loss

        # The step function handles the entire L-BFGS loop internally up to max_iter
        self.lbfgs_optimizer.step(closure)
                
        # ==========================================
        # SAVE MODEL
        # ==========================================
        logger.info("Training Complete. Saving model...")
        torch.save(self.model.state_dict(), self.config.model_training.model_path)


if __name__ == "__main__":
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PINNTrainer(device=device)
    trainer.train()