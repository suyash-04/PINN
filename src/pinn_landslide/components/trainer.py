import json
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
        self.config       = read_yaml(config_path)
        config_manager    = ConfigurationManager()
        self.arch_config  = config_manager.get_pinn_arch_config()
        self.geo_config   = config_manager.get_geo_params_config()
        self.loss_config  = config_manager.get_loss_weights_config()
        self.train_config = config_manager.get_training_config()

        # 2. Set Device
        self.device = device
        logger.info(f"Using device: {self.device}")

        # 3. Initialize Model and Loss Function
        self.model     = PINN(self.arch_config).to(self.device)
        self.criterion = CustomLoss(self.loss_config, self.geo_config, self.model).to(self.device)

        # 4. Adam Optimizer (Phase I)
        self.adam_optimizer = optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate
        )

        # FIX 1 — StepLR scheduler ─────────────────────────────────────────
        # Adam oscillated at anchor~0.07 from epoch 1000-10000 because lr
        # was still too large for the flat loss landscape after physics was
        # solved. Halving lr every 2000 epochs allows gradual convergence:
        #   epoch    0-2000 : lr = 0.0005
        #   epoch 2000-4000 : lr = 0.00025
        #   epoch 4000-6000 : lr = 0.000125
        #   epoch 6000-8000 : lr = 0.0000625
        #   epoch 8000+     : lr = 0.00003125
        # This pushes anchor/IC below 0.02 before L-BFGS starts.
        self.adam_scheduler = optim.lr_scheduler.StepLR(
            self.adam_optimizer, step_size=2000, gamma=0.5
        )

        # 5. L-BFGS Optimizer (Phase II) ───────────────────────────────────
        # FIX 2 — max_iter=100 per outer step instead of max_iter=5000 once.
        # Previously: one .step(closure) call ran 5000 iterations silently.
        #   - Only 1 log line appeared (closure logged at step 1)
        #   - L-BFGS converged immediately with no visible progress
        #   - Could not detect early convergence or divergence
        # Now: 50 outer loops × max_iter=100 = same 5000 total iterations,
        #   but with a log entry every outer loop = 50 checkpoints visible.
        #   Early stopping exits the loop when total loss stops improving.
        self.lbfgs_optimizer = optim.LBFGS(
            self.model.parameters(),
            max_iter=100,           # iterations per outer step (was 5000)
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        self._lbfgs_outer_steps = 50   # 50 × 100 = 5000 total iterations

        # Loss history — one entry per logged step, written to JSON after training
        self.loss_history: list[dict] = []

        create_directories([self.config.model_training.root])

    # ── helpers ───────────────────────────────────────────────────────────
    def load_batch(self, device):
        """Load the pre-built PINN batch dictionary from disk."""
        path  = self.config.data_ingestion.batch_path
        batch = torch.load(path, map_location=device, weights_only=False)
        return batch

    def _log_components(self, phase: str, step: int, components: dict):
        """Log a one-line component breakdown and append to loss_history."""
        logger.info(
            f"{phase} [{step}] "
            f"total={components['total']:.3e}  "
            f"pde={components['physics']:.3e}  "
            f"anchor={components['anchor']:.3e}  "
            f"fail={components['failure']:.3e}  "
            f"bc={components['boundary']:.3e}  "
            f"ic={components['initial']:.3e}"
        )
        self.loss_history.append({'phase': phase, 'step': step, **components})

    # ── main training loop ─────────────────────────────────────────────────
    def train(self):
        logger.info("Starting PINN Training...")
        self.model.train()

        logger.info("Loading data batch into memory...")
        batch = self.load_batch(device=self.device)

        # ══════════════════════════════════════════════════════════════════
        # PHASE I: Adam + StepLR (Global Search with decaying lr)
        # ══════════════════════════════════════════════════════════════════
        logger.info("--- Phase I: Adam Optimization (with StepLR scheduler) ---")

        for epoch in range(1, self.train_config.adam_epochs + 1):
            self.adam_optimizer.zero_grad()

            loss, components = self.criterion(batch)
            loss.backward()
            self.adam_optimizer.step()

            # FIX 1: step the scheduler every epoch
            self.adam_scheduler.step()

            if epoch % 500 == 0 or epoch == 1:
                current_lr = self.adam_optimizer.param_groups[0]['lr']
                logger.info(
                    f"Adam [{epoch}/{self.train_config.adam_epochs}] "
                    f"lr={current_lr:.2e}  "
                    f"total={components['total']:.3e}  "
                    f"pde={components['physics']:.3e}  "
                    f"anchor={components['anchor']:.3e}  "
                    f"fail={components['failure']:.3e}  "
                    f"bc={components['boundary']:.3e}  "
                    f"ic={components['initial']:.3e}"
                )
                self.loss_history.append({
                    'phase': 'Adam', 'step': epoch,
                    'lr': current_lr, **components
                })

        # ══════════════════════════════════════════════════════════════════
        # PHASE II: L-BFGS (Fine-Tuning with outer loop)
        # ══════════════════════════════════════════════════════════════════
        logger.info("--- Phase II: L-BFGS Optimization ---")

        # FIX 2: outer loop gives visible progress + early stopping
        prev_total   = float('inf')
        patience     = 5          # stop if no improvement for 5 outer steps
        no_improve   = 0
        total_iters  = [0]        # mutable counter shared with closure

        for outer in range(1, self._lbfgs_outer_steps + 1):

            # closure: called multiple times per .step() by line search
            def closure():
                self.lbfgs_optimizer.zero_grad()
                loss, components = self.criterion(batch)
                loss.backward()
                total_iters[0] += 1
                return loss

            self.lbfgs_optimizer.step(closure)

            # Evaluate once after the step for clean logging
            with torch.no_grad():
                _, components = self.criterion(batch)

            self._log_components("LBFGS", outer, components)

            # Early stopping check
            current_total = components['total']
            improvement   = prev_total - current_total

            if improvement < 1e-6:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        f"L-BFGS early stop at outer step {outer} "
                        f"— no improvement for {patience} consecutive steps "
                        f"(delta={improvement:.2e})"
                    )
                    break
            else:
                no_improve = 0

            prev_total = current_total

        logger.info(f"L-BFGS complete — {total_iters[0]} total closure calls")

        # ══════════════════════════════════════════════════════════════════
        # SAVE
        # ══════════════════════════════════════════════════════════════════
        logger.info("Training Complete. Saving model...")
        torch.save(self.model.state_dict(), self.config.model_training.model_path)

        history_path = Path(self.config.model_training.root) / "loss_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        logger.info(f"Loss history saved → {history_path}")

        norm_params      = batch['norm_params']
        norm_params_path = Path(self.config.model_training.root) / "norm_params.json"
        with open(norm_params_path, 'w') as f:
            json.dump({k: float(v) for k, v in norm_params.items()}, f, indent=2)
        logger.info(f"norm_params saved → {norm_params_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PINNTrainer(device=device)
    trainer.train()