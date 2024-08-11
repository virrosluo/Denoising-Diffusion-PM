import torch
import torch.nn as nn

class MiddleSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embed_dim, num_heads):
        super().__init__()
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

        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_embed):
        resnet_input = x
        out = self.resnet_conv_first(x)
        out = out + self.t_embedding_projection(t_embed)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        return out

class MiddleSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embed_dim, num_heads):
        super().__init__()

        # First Resnet Block
        self.resnet_block_first = MiddleSubBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            t_embed_dim=t_embed_dim,
            num_heads=num_heads
        )

        # Self Attention Block
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Second Resnet Block
        self.resnet_block_second = MiddleSubBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            t_embed_dim=t_embed_dim,
            num_heads=num_heads
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor):
        resnet_input = x

        # First Resnet Block Forward
        out = self.resnet_block_first(x, t_embed)

        # Attention BLock
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)

        in_attn = self.attn_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)

        out_attn, _ = self.attn(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        # Second Resnet Block Forward
        out = self.resnet_block_second(out, t_embed)

        return out