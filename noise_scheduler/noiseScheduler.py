import torch
from .config import SchedulerConfig

class LinearNoiseScheduler:
    def __init__(self, config: SchedulerConfig, dtype):
        self.num_timesteps = config.num_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end

        self.dtype = dtype

        self.betas = torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.num_timesteps, dtype=dtype)
        self.alphas = 1. - self.betas

        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    def add_noise(
        self, 
        origin: torch.Tensor, 
        noise: torch.Tensor, 
        t: torch.Tensor
    ):
        original_shape = origin.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].to(origin.device)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].to(origin.device)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod * origin + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(
        self, 
        xt: torch.Tensor, 
        noise_pred: torch.Tensor, 
        t: torch.Tensor
    ):
        '''Get image x0 from the noise predicted and noisy image xt at timestep t'''
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].to(xt.device)
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].to(xt.device)

        x0 = (xt - sqrt_one_minus_alpha_cum_prod * noise_pred) / sqrt_alpha_cum_prod
        x0 = torch.clamp(x0, min=-1., max=1.0)

        alphas = self.alphas[t].to(xt.device)
        betas = self.betas[t].to(xt.device)

        mean_sampling = xt - ((betas * noise_pred) / sqrt_one_minus_alpha_cum_prod)
        mean_sampling = mean_sampling / torch.sqrt(alphas)

        if t == 0:
            return mean_sampling, x0
        else:
            variance = sqrt_alpha_cum_prod * (1. - self.alpha_cum_prod[t - 1].to(xt.device)) / (1. - self.alpha_cum_prod[t].to(xt.device))
            sigma = variance ** 0.5
            z = torch.randn_like(xt, device=xt.device, dtype=xt.dtype)

            return mean_sampling + sigma * z, x0

if __name__ == "__main__":
    noise_scheduler = LinearNoiseScheduler(
        SchedulerConfig(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02
        )
    )

    img = torch.randn(size=(3, 3, 1280, 720))
    t = torch.randint(0, 1000, size=(3, ))

    noise_scheduler.add_noise(img, None, t)