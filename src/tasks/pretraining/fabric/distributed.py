# Importando Lightning y otras librerías necesarias
import lightning as L
from lightning.fabric.strategies import (
    FSDPStrategy,
    DeepSpeedStrategy,
    DDPStrategy,
    DataParallelStrategy,
)

# Importando utilidades personalizadas
from src.tasks.pretraining.utils import *
from src.tasks.pretraining.fabric.base import FabricTrainerBase
from tasks.pretraining.fabric.wrappers.fsdp_config import resolve_fsdp_config
from utils import inherit_init_params


@inherit_init_params
class FSDP(FabricTrainerBase):
    def _setup_strategy(self) -> FSDPStrategy:
        self.cli_logger.info("Setting up FSDP strategy.")
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
    def _setup_strategy(self) -> DeepSpeedStrategy:
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
    def _setup_strategy(self) -> DDPStrategy:
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
    def _setup_strategy(self) -> DataParallelStrategy:
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
