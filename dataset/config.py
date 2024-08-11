from dataclasses import dataclass, field
from typing import *

@dataclass
class DataModuleConfig():
    dataset_path: str = field(
        default="./dataset"
    )

    train_valid_ratio: Tuple = field(
        default=tuple([0.9, 0.1])
    )

    train_batch: int = field(
        default=100
    )

    valid_batch: int = field(
        default=300
    )

    test_batch: int = field(
        default=300
    )

    num_worker: int = field(
        default=2
    )