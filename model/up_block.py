import torch
import torch.nn as nn

class UpSamplingBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        t_embed_dim,
        up_sample,
        num_heads
    ):
        '''
        in_channels: The number of channels after up_sampling (so the input image will have channels = in_channels * 2).
        out_channels: The number of channels which this layer will output 
        up_sample: Will up sampling the image from in_channels * 2 -> in_channels
        '''
        super().__init__()
        self.up_sample = up_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels * 2),
            nn.SiLU(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
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

        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.residual_input_conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        # This convTranspose2d will take the previous up_sampling layers (which will have )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=4, stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_embed):
        '''
            out_down: The output of the DownSamplingBlock output for this one
        '''
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        # Resnet First Block
        resnet_input = x
        out = self.resnet_conv_first(resnet_input)
        out = out + self.t_embedding_projection(t_embed)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)

        in_attn = self.attn_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        
        out_attn, _ = self.attn(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        
        out = out + out_attn

        return out

if __name__ == "__main__":
    previous_feature = torch.randn(size=(1, 64, 128, 128))
    down_feature = torch.randn(size=(1, 32, 256, 256))
    t_embed = torch.randn(size=(1, 128))

    up_layer = UpSamplingBlock(
        in_channels=32,
        out_channels=32,
        t_embed_dim=128,
        up_sample=True,
        num_heads=8
    )

    output = up_layer(
        previous_feature,
        down_feature,
        t_embed
    )

    print(output.shape)