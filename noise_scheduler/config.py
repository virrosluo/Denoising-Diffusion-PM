from dataclasses import dataclass, field

@dataclass
class SchedulerConfig():
    num_timesteps: int = field(
        default=1000
    )
    
    beta_start: float = field(
        default=0.02
    ) 
    
    beta_end: float = field(
        default=0.0001
    )