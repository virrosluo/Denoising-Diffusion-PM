import torch
import torch.nn as nn

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embed_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.t_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, out_channels)
        )

        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # Self Attention Block
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Ensure that the skip_connection between the input and the first conv2d can add together
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Just a pooling layer to down sampling the "Height" and "Width" of the image (down_sampling_layer) - Can use average pool or learnable kernel
        self.pooling_layer = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor):
        resnet_input = x

        # First Resnet block
        out = self.resnet_conv_first(x)
        out = out + self.t_embedding_projection(t_embed)[:, :, None, None] # the time_embedding_project shape: (-1, -1, 1, 1) to become replicate over all pixel points

        # Second Resnet block
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, w, h = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)

        in_attn = self.attn_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)

        out_attn, _ = self.attn(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)

        out = out + out_attn

        # Down sampling or not
        out = self.pooling_layer(out)
        return out