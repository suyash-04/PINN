import torch
import torch.optim as optim
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
        
        # 4. Initialize Optimizer (Adam is standard for initial PINN training)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        create_directories([self.config.model_training.root])

    
    def load_batch(self, device):
        """
        Loads the PINN batch dictionary from disk.
        """
        path = self.config.data_ingestion.batch_path
        batch = torch.load(path, map_location=device)
        return batch

    def train(self):
        logger.info("Starting PINN Training...")
        self.model.train()

        for epoch in range(1, self.train_config.epochs + 1):
            self.optimizer.zero_grad()

            # 1. Fetch data batch (Replace with next(iter(dataloader)))
            batch = self.load_batch(device=self.device)

            # 2. Forward pass & Compute combined loss
            loss = self.criterion(batch)

            # 3. Backward pass
            loss.backward()
            
            # 4. Optimizer step
            self.optimizer.step()

            # 5. Logging
            if epoch % 500 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch}/{self.train_config.epochs}] | Total Loss: {loss.item():.6f}")
                
        logger.info("Training Complete. Saving model...")
        # Save model weights
        torch.save(self.model.state_dict(), self.config.model_training.model_path)

if __name__ == "__main__":
    trainer = PINNTrainer()
    trainer.train()