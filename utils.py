from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer

from lightning_model import Diffusion_LightningModel

import torch
import torchvision

class ImageLoggingCallback(Callback):
    def __init__(self, image_shape, num_samples, num_timestep):
        super().__init__()
        self.batch_shape = tuple([num_samples] + list(image_shape))
        self.log_samples = num_samples
        self.num_timestep = num_timestep

    @torch.inference_mode
    def on_train_epoch_start(self, trainer: Trainer, pl_module: Diffusion_LightningModel) -> None:
        xt = torch.randn(
            size=self.batch_shape,
            dtype=pl_module._dtype,
            device=pl_module.device,
            requires_grad=False
        )

        timestep = torch.as_tensor(self.num_timestep - 1).unsqueeze(dim=0).to(pl_module.device)
        
        noise_pred = pl_module.model(
            xt,
            timestep
        )

        xt, x0_pred = pl_module.noise_scheduler.sample_prev_timestep(
            xt=xt,
            noise_pred=noise_pred,
            t=timestep
        )

        imgs = torch.clamp(torch.cat([xt, x0_pred], dim=0), -1., 1.).detach().cpu()
        imgs = (imgs + 1) / 2
        grid = torchvision.utils.make_grid(
            tensor=imgs,
            nrow=2
        )
        pl_module.logger.experiment.add_image("image revision", grid, trainer.global_step)