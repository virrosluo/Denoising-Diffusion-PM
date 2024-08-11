from dataclasses import dataclass, field
from typing import *

@dataclass
class ModelConfig():
    down_channels: Tuple[int] = field(
        default=tuple([32, 64, 128, 256])
    )
    down_sampling_options: Tuple[bool] = field(
        default=tuple([True, True, True])
    )

    mid_channels: Tuple[int] = field(
        default=tuple([256, 256, 256])
    )

    time_embedding_dim: int = field(
        default=128
    )

    attention_heads: int = field(
        default=4
    )