"""
Module: distributed_strategies
Description:
    This module defines various classes that configure distributed training
    strategies using Lightning Fabric. Each class sets up a specific strategy
    based on the provided configuration and the number of available devices.
"""

# Importing Lightning and other necessary libraries
import os
import time
from typing import Tuple
import torch
import lightning as L
from lightning.fabric.strategies import (
    FSDPStrategy,
    DeepSpeedStrategy,
    DDPStrategy,
    DataParallelStrategy,
)
import torch.optim as optim
from tqdm import tqdm

# Importing custom utilities and base classes
from src.tasks.pretraining.utils import *
from src.tasks.pretraining.fabric.base import FabricTrainerBase
from tasks.pretraining.fabric.generation import FabricGeneration
from tasks.pretraining.fabric.wrappers.fsdp_config import resolve_fsdp_config
from utils import inherit_init_params
from src.tasks.pretraining.fabric.speed_monitor import SpeedMonitorFabric as Monitor


@inherit_init_params
class FSDP(FabricTrainerBase):
    """
    Class to set up the Fully Sharded Data Parallel (FSDP) strategy for distributed training.
    
    This class extends the FabricTrainerBase and configures the FSDP strategy based
    on the configuration parameters provided. When using multiple devices, it resolves
    the appropriate FSDP configuration and initializes the strategy. For a single device,
    it falls back to an automatic strategy.
    """
    def _setup_strategy(self) -> FSDPStrategy:
        """
        Set up and return the FSDP strategy.

        Returns:
            FSDPStrategy or str: A configured FSDP strategy when multiple devices are used,
            otherwise a string indicating an automatic strategy.

        The method logs the steps involved, resolves the FSDP configuration from the provided
        settings, and initializes the strategy accordingly.
        """
        self.cli_logger.info("Setting up FSDP strategy.")
        if self.devices > 1:
            # Resolve FSDP configuration with sensible defaults using the provided config and model name.
            fsdp_config = resolve_fsdp_config(
                config=self.config.__dict__,
                model_name=self.config.model_name
            )
            
            # Initialize FSDP strategy with parameters derived from the resolved configuration.
            self.strategy = FSDPStrategy(
                sharding_strategy=fsdp_config["sharding_strategy"],
                auto_wrap_policy=fsdp_config["auto_wrap_policy"],
                activation_checkpointing_policy=fsdp_config["activation_checkpointing"],
                activation_checkpointing=fsdp_config["activation_checkpointing"],
                state_dict_type=fsdp_config["state_dict_type"],
                limit_all_gathers=fsdp_config["limit_all_gathers"],
                cpu_offload=fsdp_config["cpu_offload"],
            )
            
            self.cli_logger.info(f"Using auto_wrap_policy: {fsdp_config['auto_wrap_policy']}")
            if fsdp_config["activation_checkpointing"]:
                self.cli_logger.info(f"Using activation_checkpointing: {fsdp_config['activation_checkpointing']}")
        else:
            # For a single device, fall back to the automatic strategy.
            self.strategy = "auto"
            self.cli_logger.warning("Using automatic strategy for 1 device.")
            
        return self.strategy


@inherit_init_params
class DeepSpeed(FabricTrainerBase):
    """
    Class to set up the DeepSpeed strategy for distributed training.
    
    This class configures and returns a DeepSpeed strategy that leverages multiple devices.
    It uses DeepSpeed-specific parameters to optimize training and properly handles optimizer
    configuration based on the ZeRO stage being used.
    """
    def _setup_strategy(self) -> DeepSpeedStrategy:
        """
        Set up and return the DeepSpeed strategy.

        Returns:
            DeepSpeedStrategy: A strategy object configured for DeepSpeed training.

        The method creates a DeepSpeed configuration dictionary based on the provided
        config parameters and initializes the strategy with the appropriate settings
        for the specified ZeRO optimization stage.
        """
        self.cli_logger.info("Setting up DeepSpeed strategy.")
        
        # Create a DeepSpeed config dictionary with explicit optimizer configuration
        ds_config = {
            "train_batch_size": self.config.batch_size * self.devices,
            "zero_optimization": {
                "stage": self.config.zero_stage
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.lr,
                    "weight_decay": self.config.weight_decay,
                    "betas": [self.config.beta1, self.config.beta2]
                }
            }
        }
        
        # Handle gradient accumulation specifically for DeepSpeed 
        if hasattr(self.config, "gradient_accumulation_steps") and self.config.gradient_accumulation_steps:
            ds_config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps
        
        # Add additional zero optimization parameters if specified
        if hasattr(self.config, "zero_optimization") and self.config.zero_optimization:
            # Merge specific zero optimization settings
            if isinstance(self.config.zero_optimization, dict):
                ds_config["zero_optimization"].update(self.config.zero_optimization)
        
        # Add optimizer offload settings if specified
        if hasattr(self.config, "offload_optimizer") and self.config.offload_optimizer:
            if "offload_optimizer" not in ds_config["zero_optimization"]:
                ds_config["zero_optimization"]["offload_optimizer"] = {}
            ds_config["zero_optimization"]["offload_optimizer"].update({
                "device": "cpu",
                "pin_memory": True
            })
        
        # Add parameter offload settings if specified
        if hasattr(self.config, "offload_parameters") and self.config.offload_parameters:
            if "offload_param" not in ds_config["zero_optimization"]:
                ds_config["zero_optimization"]["offload_param"] = {}
            ds_config["zero_optimization"]["offload_param"].update({
                "device": "cpu",
                "pin_memory": True
            })
        
        # Configure additional DeepSpeed parameters if specified
        for param in ["gradient_clipping", "fp16", "bf16"]:
            if hasattr(self.config, param):
                ds_config[param] = getattr(self.config, param)
        
        # Initialize DeepSpeed strategy with the configured settings
        strategy = DeepSpeedStrategy(config=ds_config)
        
        self.cli_logger.debug(f"DeepSpeed config: {ds_config}")
        return strategy

    def _pipeline(self, fabric: L.Fabric) -> None:
        """
        Override the _pipeline method to properly handle optimizer configuration for DeepSpeed.
        
        This method overrides the base _pipeline method to ensure that the optimizer is properly 
        configured for DeepSpeed, especially when using ZeRO optimization stages that require 
        an optimizer to be provided.
        
        Parameters:
            fabric (L.Fabric): The Fabric instance coordinating distributed training.
        """
        # Set up deterministic behavior if a seed is provided
        if self.config.get("seed", None) is not None:
            setup_environment(self.config.seed)
            fabric.seed_everything(self.config.seed)

        # Initialize training monitor with given parameters
        self.monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=self.config.log_iter_interval)

        # Create output directory for checkpoints and logs on the main process
        if fabric.global_rank == 0:
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Use safe_barrier instead of fabric.barrier() to handle NVML errors
        safe_barrier(fabric, self.cli_logger)

        # Setup Fabric data loaders
        self.dataloaders = {k: fabric.setup_dataloaders(v) for k, v in self.dataloaders.items()}

        # Explicitly pass the parallelization_strategy and zero_stage parameters to ensure 
        # the model is initialized with the correct settings for DeepSpeed ZeRO
        config_with_strategy = dict(self.config)
        config_with_strategy["parallelization_strategy"] = "deep_speed"
        
        # Ensure zero_stage is correctly passed from config
        if hasattr(self.config, "zero_stage") and isinstance(self.config.zero_stage, int):
            config_with_strategy["zero_stage"] = self.config.zero_stage
        
        # Instantiate and initialize the model within the fabric.init_module() context
        t0 = time.perf_counter()
        with fabric.init_module():
            self.model = FabricGeneration(**config_with_strategy)

        # Enable or disable gradient checkpointing based on configuration
        if self.config.gradient_checkpointing:
            self.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                "use_reentrant": False
            })
        else:
            self.model.model.gradient_checkpointing_disable()

        self.cli_logger.info(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        
        # Define a placeholder optimizer for DeepSpeed that won't interfere with its internal optimizer
        class DeepSpeedPlaceholderOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.001):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)
                
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                return loss
                
            def zero_grad(self, set_to_none=False):
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            if set_to_none:
                                p.grad = None
                            else:
                                p.grad.detach_()
                                p.grad.zero_()

        # Create placeholder optimizer with model parameters
        placeholder_optimizer = DeepSpeedPlaceholderOptimizer(self.model.parameters(), lr=self.config.lr)
        
        # Set up the model with DeepSpeed - DeepSpeed will handle the real optimizer internally
        # based on the config we passed to DeepSpeedStrategy
        self.model = fabric.setup(self.model)
        
        # Setup scheduler with the placeholder optimizer
        scheduler = select_scheduler(
            placeholder_optimizer,
            self.config.lr_scheduler,
            self.config.number_epochs,
            fabric.world_size,
            self.config.batch_size,
            self.dataset['train'],
            self.config.warmup_proportion,
            self.config.gradient_accumulation_steps
        )
        
        # Store initial training state
        self.state = {
            "model": self.model,
            "optimizer": placeholder_optimizer,  # Using placeholder optimizer for state tracking
            "hparams": self.hparams,
            "iter_num": 0,
            "step_count": 0,
            "scheduler": scheduler
        }
        
        # Load training state from a checkpoint if available
        self._load_from_checkpoint(fabric)
        
        # Run the training loop
        train_time = time.perf_counter()
        self._train(fabric)
        self.cli_logger.info(f"Training time: {(time.perf_counter() - train_time):.2f}s")
        
        if fabric.device.type == "cuda":
            self.cli_logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    def _accumulate_training(self, fabric: L.Fabric, model: L.LightningModule, batch: Tuple[torch.Tensor, ...], step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform DeepSpeed-compatible training step with gradient accumulation.
        
        This method overrides the base _accumulate_training method to avoid using no_backward_sync()
        which is not supported by DeepSpeed. Instead, we rely on DeepSpeed's built-in gradient accumulation.
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance.
        - model (L.LightningModule): The model being trained.
        - batch (tuple): A batch of training data.
        - step (int): The current training step.
        
        Returns:
        - tuple: Contains the outputs from the training step and the loss tensor.
        """
        # Get training outputs
        training_output = model.training_step(batch, step)
        outputs = training_output["outputs"]
        loss = training_output["loss"]
        
        # For DeepSpeed, we don't need to manually scale the loss as DeepSpeed's engine 
        # handles gradient accumulation internally based on the config
        fabric.backward(loss)
        
        # DeepSpeed engine checks internally if it should step based on gradient_accumulation_steps
        # We only need to update our step counter when a real step happens
        is_optimizer_step = (self.state["iter_num"] + 1) % self.config.gradient_accumulation_steps == 0
        if is_optimizer_step:
            optimizer = self.state["optimizer"]
            scheduler = self.state["scheduler"]
            
            # No need to clip gradients here as DeepSpeed handles it
            # However we do need to step the optimizer to update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            self.state["step_count"] += 1
            self._try_validate(fabric)
            
        self.state["iter_num"] += 1
        return outputs, loss
    
    def _train(self, fabric: L.Fabric) -> None:
        """
        Execute the main training loop over the specified number of epochs.
        
        This method overrides the base _train method to use DeepSpeed-specific gradient 
        accumulation rather than the standard no_backward_sync approach.
        
        Parameters:
        - fabric (L.Fabric): The Fabric instance driving the training.
        """
        model = self.state["model"]
        self.total_lengths = 0
        self.train_total_t0 = time.perf_counter()
        self.initial_iter = self.state["iter_num"]
        epochs = self.config.number_epochs
        model.train()
        resume_iter = self.state["iter_num"]
        
        for epoch in range(epochs):
            if fabric.global_rank == 0:
                self.cli_logger.debug(f"Running Epoch {epoch + 1} of {epochs}")
                
            # Use tqdm progress bar for the training dataloader if on the main process
            batch_iterator = tqdm(self.dataloaders['train'], mininterval=0, colour="blue") \
                if fabric.global_rank == 0 else self.dataloaders['train']
                
            batch_iterator, resume_iter = self._get_resume_iterator(batch_iterator, resume_iter)
            if batch_iterator is None:
                continue
                
            for step, batch in enumerate(batch_iterator):
                self.train_iter_t0 = time.perf_counter()
                
                # Use DeepSpeed-specific gradient accumulation
                _, loss = self._accumulate_training(fabric, model, batch, step)
                
                self.total_lengths += batch["input_ids"].size(1)
                self.train_t1 = time.perf_counter()
                self._train_logs(fabric, loss)
                
            self._try_validate(fabric, epochFinished=True)
            
        self._try_validate(fabric, trainingFinished=True)


@inherit_init_params
class DistributedDataParallel(FabricTrainerBase):
    """
    Class to set up the Distributed Data Parallel (DDP) strategy for training.
    
    This class configures the DDP strategy using typical parameters from the training configuration.
    When using multiple devices, common settings such as the process group backend, unused parameter
    detection, and static graph optimizations are applied. For a single device, it defaults to an automatic strategy.
    """
    def _setup_strategy(self) -> DDPStrategy:
        """
        Set up and return the DDP strategy.

        Returns:
            DDPStrategy or str: A configured DDP strategy object for multiple devices,
            or a string indicating an automatic strategy if only one device is used.
        """
        self.cli_logger.info("Setting up DDP strategy.")
        if self.devices > 1:
            # Configure the DDPStrategy with key parameters obtained from the configuration.
            strategy = DDPStrategy(
                find_unused_parameters=self.config.get("find_unused_parameters", False),
                process_group_backend=self.config.get("process_group_backend", "nccl"),
                static_graph=self.config.get("static_graph", True),
                # Additional parameters can be added here as needed.
            )
        else:
            strategy = "auto"
        return strategy


@inherit_init_params
class DataParallel(FabricTrainerBase):
    """
    Class to set up the Data Parallel (DP) strategy for distributed training.
    
    This class handles the configuration of the Data Parallel strategy for scenarios where
    multiple devices are available. It assigns the devices used in parallel computations,
    and for a single device configuration, it defaults to an automatic setting.
    """
    def _setup_strategy(self) -> DataParallelStrategy:
        """
        Set up and return the Data Parallel strategy.

        Returns:
            DataParallelStrategy or str: A Data Parallel strategy object configured with the
            provided devices, or a string indicating an automatic strategy for single-device setups.
        """
        self.cli_logger.info("Setting up DP strategy.")
        if self.devices > 1:
            # Initialize DataParallelStrategy by specifying parallel devices and the output device.
            strategy = DataParallelStrategy(
                parallel_devices=self.devices,
                output_device=(
                    self.devices[0] if isinstance(self.devices, list) else self.devices
                ),
            )
        else:
            strategy = "auto"
        return strategy
