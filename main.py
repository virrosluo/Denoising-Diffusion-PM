from model import *
from dataset import *
from noise_scheduler import *
from lightning_model import *
from utils import ImageLoggingCallback

from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

import lightning

from transformers import HfArgumentParser

import yaml
import sys
import os
import logging

log = logging.Logger(__name__)

if __name__ == "__main__":

    log.info("READING ARGUMENTS")

    with open(sys.argv[1], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    diffusion_config:SchedulerConfig = HfArgumentParser(SchedulerConfig).parse_dict(config['diffusion_params'])[0]
    dataset_config:DataModuleConfig = HfArgumentParser(DataModuleConfig).parse_dict(config['dataset_params'])[0]
    train_config:TrainingConfig = HfArgumentParser(TrainingConfig).parse_dict(config['train_params'])[0]
    model_config:ModelConfig = HfArgumentParser(ModelConfig).parse_dict(config['model_params'])[0]

    if train_config.use_deepspeed:
        log.info("USING DEEPSPEED")
        deepspeed_config = config['deepspeed_config']
    else:
        log.info("NOT USING DEEPSPEED")

    dtype_map = {
        '16-true': torch.float16,
        '16-mixed': torch.float16,
        'bf16-true': torch.bfloat16,
        'bf16-mixed': torch.bfloat16,
        '32-true': torch.float32,
        '64-true': torch.float64,
        '64': torch.float64,
        '32': torch.float32,
        '16': torch.float16,
        'bf16': torch.bfloat16,
    }
    dataType = dtype_map[train_config.precision]

# --------------------------------- Noise Scheduler INIT
    log.info("INITIALIZING SCHEDULER")
    scheduler = LinearNoiseScheduler(config=diffusion_config, dtype=dataType)

# --------------------------------- Dataset INIT
    log.info("INITIALIZING DATASET")
    dataset = MNIST_DataModule(config=dataset_config)

# --------------------------------- Model INIT
    log.info("INITIALIZING MODEL")
    model = Unet(
        in_channels=dataset.get_image_shape()[0], 
        config=model_config
    )
    training_model = Diffusion_LightningModel(
        model=model,
        noise_scheduler=scheduler,
        config=train_config
    )

# --------------------------------- Logger INIT
    logger = TensorBoardLogger(save_dir=train_config.training_process_log)

# --------------------------------- Profiler INIT
    log.info("INITIALIZING PROFILER")
    profiler = PyTorchProfiler(
        dirpath=train_config.training_process_log,
        filename="profiler",

        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            dir_name=os.path.join(train_config.training_process_log, "profiler")
        )
    )

# --------------------------------- Callback INIT
    log.info("INITIALIZING CALLBACK FUNCTION")
    imgs_log_callback = ImageLoggingCallback(image_shape=list(dataset.get_image_shape()), num_samples=5, num_timestep=diffusion_config.num_timesteps)
    device_usage_callback = DeviceStatsMonitor()

# --------------------------------- DeepSpeed INIT
    log.info("INITIALIZING STRATEGY")
    if train_config.use_deepspeed:
        strategy = DeepSpeedStrategy(
            **deepspeed_config
        )
    else:
        strategy = "auto"

# --------------------------------- Trainer INIT
    log.info("INITIALIZING TRAINER")
    trainer = lightning.Trainer(
        default_root_dir=train_config.training_process_log,
        accelerator="auto",
        strategy=strategy,
        devices="auto",
        precision=train_config.precision,
        callbacks=[imgs_log_callback, device_usage_callback],
        max_epochs=train_config.num_epochs,
        limit_train_batches=dataset_config.train_batch,
        limit_val_batches=dataset_config.valid_batch,
        limit_test_batches=dataset_config.test_batch,
        val_check_interval=train_config.run_valid_step_after,
        enable_model_summary=True,
        profiler=profiler,
        logger=logger,
    )

# --------------------------------- Training Model
    log.info("TRAINING MODEL")
    trainer.fit(
        model=training_model,
        datamodule=dataset,
    )