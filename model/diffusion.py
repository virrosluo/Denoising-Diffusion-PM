import torch
import torch.nn as nn

from .down_block import DownSamplingBlock
from .mid_block import MiddleSamplingBlock
from .up_block import UpSamplingBlock
from .time_embedder import CustomRoFormerSinusoidalPositionalEmbedding
from .config import ModelConfig

class Unet(nn.Module):
    '''
    The output of the Unet after running down_block -> mid_block -> up_block will output a tensor with shape = img_shape
    The output image will represent the noise which is added to each pixel in the original image to create noisy image
    '''
    def __init__(self, in_channels, config: ModelConfig):
        super().__init__()
        self.config = config

        self.t_project = CustomRoFormerSinusoidalPositionalEmbedding(
            num_positions=1000,
            embedding_dim=self.config.time_embedding_dim,
        )

        self.up_sample = list(reversed(self.config.down_sampling_options))
        self.conv2_in = nn.Conv2d(in_channels, self.config.down_channels[0], kernel_size=3, padding=1)

        self.down_components = nn.ModuleList([])
        for i in range(len(self.config.down_channels) - 1):
            self.down_components.append(
                DownSamplingBlock(
                    in_channels=self.config.down_channels[i],
                    out_channels=self.config.down_channels[i + 1],
                    t_embed_dim=self.config.time_embedding_dim,
                    down_sample=self.config.down_sampling_options[i],
                    num_heads=self.config.attention_heads
                )
            )

        self.mid_components = nn.ModuleList([])
        for i in range(len(self.config.mid_channels) - 1):
            self.mid_components.append(
                MiddleSamplingBlock(
                    in_channels=self.config.mid_channels[i],
                    out_channels=self.config.mid_channels[i + 1],
                    t_embed_dim=self.config.time_embedding_dim,
                    num_heads=self.config.attention_heads
                )
            )

        self.up_components = nn.ModuleList([])
        for i in reversed(range(len(self.config.down_channels) - 1)):
            self.up_components.append(
                UpSamplingBlock(
                    in_channels=self.config.down_channels[i],
                    out_channels=self.config.down_channels[i],
                    t_embed_dim=self.config.time_embedding_dim,
                    up_sample=self.up_sample[i],
                    num_heads=self.config.attention_heads
                )
            )

        self.conv2_out = nn.Sequential(
            nn.GroupNorm(8, self.config.down_channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.config.down_channels[0], in_channels, kernel_size=1)
        )

    def forward(self, x, t):
        out = self.conv2_in(x)
        t_embed = self.t_project(t)

        down_outs = []
        for down in self.down_components:
            down_outs.append(out)
            out = down(out, t_embed)

        for mid in self.mid_components:
            out = mid(out, t_embed)

        for up in self.up_components:
            out = up(out, down_outs.pop(), t_embed)

        out = self.conv2_out(out)

        return out

if __name__ == "__main__":
    img = torch.randn(size=(2, 1, 28, 28))
    timestep = torch.randint(1, 1000, size=(2, ))

    model = Unet(
        in_channels=img.shape[1], 
        config=ModelConfig(
            down_channels=[32, 64, 128],
            down_sampling_options=[True, True],
            mid_channels=[128, 128, 128],
            time_embedding_dim=128,
            attention_heads=4
        )
    )

    try:
        output = model(img, timestep)
        print(output.shape)
    except Exception as e: 
        print(e)