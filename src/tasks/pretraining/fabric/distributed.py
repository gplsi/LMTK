"""
Module: distributed_strategies
Description:
    This module defines various classes that configure distributed training
    strategies using Lightning Fabric. Each class sets up a specific strategy
    based on the provided configuration and the number of available devices.
"""


# Importing Lightning and other necessary libraries
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
        self.cli_logger.info("Using Devices: %s", self.devices)
        if self.devices > 1:
            # Resolve FSDP configuration with sensible defaults
            fsdp_config = resolve_fsdp_config(
                config=self.config.__dict__,
                model_name=self.config.model_name
            )
            
            # FSDP strategy for multiple devices
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

    def _setup_strategy(self):
        """
        Set up and return the DeepSpeed strategy.

        Returns:
            DeepSpeedStrategy: A strategy object configured for DeepSpeed training.

        The method creates a DeepSpeed configuration dictionary based on the provided
        config parameters and initializes the strategy with the appropriate settings
        for the specified ZeRO optimization stage.
        """
        
        self.cli_logger.info("Setting up DeepSpeed strategy.")
        if self.devices > 1:
            # Pass DeepSpeed-specific parameters from your config
            strategy = DeepSpeedStrategy(
                zero_stage=self.config.zero_stage,  # e.g. 2 or 3
                offload_optimizer=self.config.offload_optimizer,  # True/False
                offload_parameters=self.config.offload_parameters,  # True/False
                # Add any other DeepSpeed parameters here as needed.
            )
        else:
            raise NotImplementedError(
                "Automatic strategy is not yet implemented for 1 device."
            )
        return strategy
    
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
            # Configure DDPStrategy with common parameters:
            strategy = DDPStrategy(
                find_unused_parameters=self.config.get("find_unused_parameters", False),
                process_group_backend=self.config.get("process_group_backend", "nccl"),
                static_graph=self.config.get("static_graph", True),
                # You can add additional parameters here if needed.
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
            strategy = DataParallelStrategy(
                parallel_devices=self.devices,
                output_device=(
                    self.devices[0] if isinstance(self.devices, list) else self.devices
                ),
            )
        else:
            strategy = "auto"
        return strategy
