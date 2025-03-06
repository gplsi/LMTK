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

# Importing custom utilities and base classes
from src.tasks.pretraining.utils import *
from src.tasks.pretraining.fabric.base import FabricTrainerBase
from tasks.pretraining.fabric.wrappers.fsdp_config import resolve_fsdp_config
from utils import inherit_init_params


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
    It uses DeepSpeed-specific parameters to optimize training. For single device setups,
    it raises an error because the automatic strategy for DeepSpeed is not implemented.
    """
    def _setup_strategy(self) -> DeepSpeedStrategy:
        """
        Set up and return the DeepSpeed strategy.

        Returns:
            DeepSpeedStrategy: A strategy object configured for DeepSpeed training.

        Raises:
            NotImplementedError: If there is only one device available, as an automatic strategy
            is not supported for single-device training.
        """
        self.cli_logger.info("Setting up DeepSpeed strategy.")
        if self.devices > 1:
            # Build the DeepSpeed strategy using specific parameters from the configuration.
            strategy = DeepSpeedStrategy(
                zero_stage=self.config.zero_stage,  # e.g., 2 or 3
                offload_optimizer=self.config.offload_optimizer,  # True/False
                offload_parameters=self.config.offload_parameters,  # True/False
                # Additional DeepSpeed parameters can be added here.
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
