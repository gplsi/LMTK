import math
import time
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools
# Importando Lightning y otras librerías necesarias
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DeepSpeedStrategy, DDPStrategy, DataParallelStrategy
from pytorch_lightning.loggers import WandbLogger

# Importando utilidades personalizadas
from src.tasks.continual.fabric.speed_monitor import SpeedMonitorFabric as Monitor
from src.tasks.continual.fabric.logger import step_csv_logger
from src.tasks.continual.utils import *
from src.tasks.continual.distributed_model_classes import FabricGeneration
from src.tasks.continual.fabric.abstract_classes import Fabric_Abstract


class FSDP(Fabric_Abstract):
    def __init__(self, devices, config, resume=False):
        super().__init__(devices, config, resume)
        
        self.devices = devices
        self.config = config
        self.resume = resume
        
    def _setup_strategy(self):
        if self.devices > 1:
            # FSDP strategy for multiple devices
            strategy = FSDPStrategy(
                sharding_strategy=self.config.sharding_strategy,
                auto_wrap_policy=self.config.auto_wrap_policy,
                activation_checkpointing_policy=self.config.auto_wrap_policy,
                state_dict_type=self.config.state_dict_type,
                limit_all_gathers=self.config.limit_all_gathers,
                cpu_offload=self.config.cpu_offload,
            )
        else:
            strategy = "auto"
            # TODO: Poner en formato de warning
            print("Using automatic strategy for 1 device.")
            raise NotImplementedError("Automatic strategy is not yet implemented for 1 device.")
    
        return strategy            
    
    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(devices=self.devices, strategy=strategy, precision=self.config.precision, loggers=[loggers])
        
        self.hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
        fabric.print(self.hparams)
        
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)
    
    
class DeepSpeed(Fabric_Abstract):
    def __init__(self, devices, config, resume=False):
        super().__init__(devices, config, resume)
        
        self.devices = devices
        self.config = config
        self.resume = resume
        
    def _setup_strategy(self):
        if self.devices > 1:
            # Pass DeepSpeed-specific parameters from your config
            strategy = DeepSpeedStrategy(
                zero_stage=self.config.zero_stage,           # e.g. 2 or 3
                offload_optimizer=self.config.offload_optimizer,  # True/False
                offload_parameters=self.config.offload_parameters  # True/False
                # Add any other DeepSpeed parameters here as needed.
            )
        else:
            raise NotImplementedError("Automatic strategy is not yet implemented for 1 device.")
        return strategy            
    
    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()
        fabric = L.Fabric(devices=self.devices, strategy=strategy, precision=self.config.precision, loggers=[loggers])
        
        self.hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
        fabric.print(self.hparams)
        
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)
        
        
class DistributedDataParallel(Fabric_Abstract):
    def __init__(self, devices, config, resume=False):
        super().__init__(devices, config, resume)
        self.devices = devices
        self.config = config
        self.resume = resume
        
    def _setup_strategy(self):
        if self.devices > 1:
            # Configure DDPStrategy with common parameters:
            strategy = DDPStrategy(
                find_unused_parameters=self.config.find_unused_parameters 
                    if hasattr(self.config, "find_unused_parameters") 
                    else False,
                process_group_backend=self.config.process_group_backend 
                    if hasattr(self.config, "process_group_backend") 
                    else "nccl",
                static_graph=self.config.static_graph 
                    if hasattr(self.config, "static_graph") 
                    else True,
                # You can add additional parameters here if needed.
            )
        else:
            strategy = "auto"
        return strategy
            
    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()  # Ensure _set_loggers returns a list of logger objects.
        fabric = L.Fabric(
            devices=self.devices, 
            strategy=strategy, 
            precision=self.config.precision, 
            loggers=loggers
        )
        self.hparams = {
            k: v for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        fabric.print(self.hparams)
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)
        
        
class DataParallel(Fabric_Abstract):
    def __init__(self, devices, config, resume=False):
        super().__init__(devices, config, resume)
        self.devices = devices 
        self.config = config
        self.resume = resume
        
    def _setup_strategy(self):
        if self.devices > 1:
            strategy = DataParallelStrategy(
                parallel_devices=self.devices, 
                output_device=self.devices[0] if isinstance(self.devices, list) else self.devices
            )
        else:
            strategy = "auto"
        return strategy
            
    def setup(self) -> None:
        strategy = self._setup_strategy()
        loggers = self._set_loggers()  # Ensure _set_loggers returns a list of logger objects.
        fabric = L.Fabric(
            devices=self.devices, 
            strategy=strategy, 
            precision=self.config.precision, 
            loggers=loggers
        )
        self.hparams = {
            k: v for k, v in locals().items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        fabric.print(self.hparams)
        fabric.launch(self._pipeline, self.resume, self.config, self.hparams)
        
        
