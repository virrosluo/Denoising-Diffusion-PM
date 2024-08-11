import lightning
import torch
from torch import optim

from noise_scheduler import LinearNoiseScheduler

from dataclasses import dataclass, field
from typing import *

@dataclass
class TrainingConfig():
    training_process_log: str = field(
        default="./training_process"
    )

    precision: str = field(
        default="32-true",
        metadata={
            "help": "Choosing the precision for training model",
            "choices": "['16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16']"
        }
    )

    lr: float = field(
        default=1e-3
    )

    num_epochs: float = field(
        default=100
    )

    run_valid_step_after: float = field(
        default=0.1
    )

    use_deepspeed: bool = field(
        default=True
    )

class Diffusion_LightningModel(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        noise_scheduler: LinearNoiseScheduler, 
        config: TrainingConfig
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.noise_scheduler = noise_scheduler
        self.criterion = torch.nn.MSELoss()

    def similar_run(self, batch, batch_idx):
        img, _ = batch

        noise = torch.randn_like(img).to(img.device)
        timestep = torch.randint(
            low=0, 
            high=self.noise_scheduler.num_timesteps, 
            size=(img.shape[0], )
        ).to(img.device)

        # Add noise to origin image according to timestep ith
        noisy_img = self.noise_scheduler.add_noise(img, noise, timestep)

        # Predict noise from the model
        noise_prediction = self.model(img, timestep)
        loss = self.criterion(noise_prediction, noise)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.similar_run(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.similar_run(batch, batch_idx)
        self.log("valid_loss", loss)

        return {"valid_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.similar_run(batch, batch_idx)
        self.log("test_loss", loss)

        return {"test_loss": loss}

    def configure_optimizers(self):
        return optim.AdamW(
            params=self.model.parameters(), 
            lr=self.config.lr, 
        )

