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
from tasks.pretraining.fabric.base import FabricTrainerBase
from utils import inherit_init_params


@inherit_init_params
class FSDP(FabricTrainerBase):
    def _setup_strategy(self):
        self.cli_logger.info("Setting up FSDP strategy.")
        #if self.devices > 1:
            # FSDP strategy for multiple devices
        strategy = FSDPStrategy(
            sharding_strategy=self.config.sharding_strategy,
            auto_wrap_policy=AUTO_WRAPPER[self.config.auto_wrap_policy],
            activation_checkpointing_policy=self.config.auto_wrap_policy,
            state_dict_type=self.config.state_dict_type,
            limit_all_gathers=self.config.limit_all_gathers,
            cpu_offload=self.config.cpu_offload,
        )
        # else:
        #     strategy = "auto"
        #     # TODO: Poner en formato de warning
        #     print("Using automatic strategy for 1 device.")
        #     raise NotImplementedError(
        #         "Automatic strategy is not yet implemented for 1 device."
        #     )

        return strategy

    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(
            devices=self.devices,
            strategy=strategy,
            precision=self.config.precision,
            loggers=[loggers],
        )

        self.hparams = {
            k: v
            for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        self.cli_logger.debug(self.hparams)

        fabric.launch(self._pipeline)


@inherit_init_params
class DeepSpeed(FabricTrainerBase):
    def _setup_strategy(self):
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

    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(
            devices=self.devices,
            strategy=strategy,
            precision=self.config.precision,
            loggers=[loggers],
        )

        self.hparams = {
            k: v
            for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        self.cli_logger.debug(self.hparams)
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)


@inherit_init_params
class DistributedDataParallel(FabricTrainerBase):
    def _setup_strategy(self):
        self.cli_logger.info("Setting up DDP strategy.")
        if self.devices > 1:
            # Configure DDPStrategy with common parameters:
            strategy = DDPStrategy(
                find_unused_parameters=(
                    self.config.find_unused_parameters
                    if hasattr(self.config, "find_unused_parameters")
                    else False
                ),
                process_group_backend=(
                    self.config.process_group_backend
                    if hasattr(self.config, "process_group_backend")
                    else "nccl"
                ),
                static_graph=(
                    self.config.static_graph
                    if hasattr(self.config, "static_graph")
                    else True
                ),
                # You can add additional parameters here if needed.
            )
        else:
            strategy = "auto"
        return strategy

    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = (
            self._set_loggers()
        )  # Ensure _set_loggers returns a list of logger objects.
        fabric = L.Fabric(
            devices=self.devices,
            strategy=strategy,
            precision=self.config.precision,
            loggers=loggers,
        )
        self.hparams = {
            k: v
            for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        self.cli_logger.debug(self.hparams)
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)


@inherit_init_params
class DataParallel(FabricTrainerBase):
    def _setup_strategy(self):
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

    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = (
            self._set_loggers()
        )  # Ensure _set_loggers returns a list of logger objects.
        fabric = L.Fabric(
            devices=self.devices,
            strategy=strategy,
            precision=self.config.precision,
            loggers=loggers,
        )
        self.hparams = {
            k: v
            for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        self.cli_logger.debug(self.hparams)
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)
