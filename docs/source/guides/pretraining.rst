.. _pretraining:

Pretraining Guide
================

.. raw:: html

   <div class="feature-box">
      <h3>Overview</h3>
      <p>This guide covers the pretraining capabilities of the ML Training Framework, focusing on distributed training strategies, optimization techniques, and configuration options.</p>
   </div>

.. admonition:: Prerequisites
   :class: note

   * Basic understanding of transformer-based language models
   * Familiarity with PyTorch and distributed training concepts
   * Access to appropriate compute resources for model training

Pretraining Architecture
-----------------------

The pretraining pipeline in the ML Training Framework is built around a modular architecture that supports different distributed training paradigms:

.. tabs::

   .. tab:: FSDP

      .. code-block:: python
         :caption: FSDP Configuration
         :linenos:

         from src.tasks.clm_training.fabric.wrappers.fsdp_config import get_fsdp_config
         
         fsdp_config = get_fsdp_config(
             mixed_precision=True,
             sharding_strategy="FULL_SHARD",
             checkpoint_activation=True,
             activation_checkpointing_policy="contiguous"
         )

      **Key Features:**

      * Full parameter sharding across devices
      * Memory-efficient gradient computation
      * Support for checkpoint activation to reduce memory usage
      * Auto-wrapped module hierarchy

   .. tab:: DDP

      .. code-block:: python
         :caption: DDP Setup
         :linenos:
         
         import torch.distributed as dist
         from src.tasks.clm_training.fabric.distributed import setup_ddp
         
         setup_ddp(
             backend="nccl",
             find_unused_parameters=False
         )
         
         # Model will be wrapped with DDP
         model = setup_model_with_ddp(model)

      **Key Features:**

      * Efficient gradient synchronization
      * Lower communication overhead
      * Compatible with gradient accumulation
      * Suitable for homogeneous hardware setups

   .. tab:: DeepSpeed

      .. code-block:: python
         :caption: DeepSpeed Configuration
         :linenos:
         
         # DeepSpeed configuration via JSON file
         ds_config = {
             "train_batch_size": 32,
             "fp16": {
                 "enabled": True
             },
             "zero_optimization": {
                 "stage": 2,
                 "offload_optimizer": {
                     "device": "cpu"
                 }
             }
         }
         
         # Initialize with DeepSpeed
         model, optimizer, _, _ = deepspeed.initialize(
             model=model,
             config=ds_config
         )

      **Key Features:**

      * ZeRO optimizer for memory efficiency
      * CPU offloading capability
      * Mixed precision training
      * Pipeline parallelism support

Training Orchestration
---------------------

The training process is managed by the ``Orchestrator`` class, which handles data loading, model initialization, training loops, and checkpointing:

.. mermaid::

   flowchart TD
      A[Configuration Loading] --> B[Model Initialization]
      B --> C[Optimizer Setup]
      C --> D[Data Preparation]
      D --> E[Training Loop]
      E --> F[Validation]
      F --> G{Continue?}
      G -->|Yes| E
      G -->|No| H[Save Model]
      E -->|Checkpoint| I[Save Checkpoint]
      I --> E

Configuration Options
-------------------

The framework provides flexible configuration options through YAML files:

.. code-block:: yaml
   :caption: Example Pretraining Configuration
   :linenos:

   model:
     type: llama
     size: 7B
     vocab_size: 32000
     hidden_size: 4096
     intermediate_size: 11008
     num_hidden_layers: 32
     num_attention_heads: 32
     max_position_embeddings: 4096
     rms_norm_eps: 1.0e-6
   
   training:
     batch_size: 32
     gradient_accumulation_steps: 8
     learning_rate: 3.0e-4
     warmup_steps: 2000
     max_steps: 100000
     lr_scheduler: cosine
     weight_decay: 0.1
     clip_grad_norm: 1.0
   
   distributed:
     strategy: fsdp
     fsdp_config:
       sharding_strategy: FULL_SHARD
       mixed_precision: true
       activation_checkpointing: true

Performance Monitoring
--------------------

The framework includes tools for monitoring training performance:

.. code-block:: python
   :linenos:

   from src.tasks.clm_training.fabric.speed_monitor import SpeedMonitor
   
   # Initialize the speed monitor
   speed_monitor = SpeedMonitor(
       window_size=50,
       time_unit="hours"
   )
   
   # Update during training
   for step in training_steps:
       # Training logic
       speed_monitor.update(samples=batch_size)
       
       if step % log_interval == 0:
           metrics = speed_monitor.compute()
           logger.info(f"Training speed: {metrics['samples_per_second']:.2f} samples/second")

.. warning::
   
   Monitoring overhead can impact training performance. Use appropriate logging intervals based on your training scale.

Example Usage
-----------

.. code-block:: python
   :caption: Complete Pretraining Example
   :linenos:

   from src.tasks.clm_training.orchestrator import PretrainingOrchestrator
   from src.config.config_loader import load_config
   
   # Load configuration
   config = load_config("config/experiments/continual_llama_3.1_7b.yaml")
   
   # Initialize orchestrator
   orchestrator = PretrainingOrchestrator(config)
   
   # Start training
   orchestrator.train()

Next Steps
---------

* Explore :ref:`distributed` for detailed information on scaling training
* Learn about :ref:`tokenization` to prepare your data
* Check out the :ref:`configuration` guide for advanced options

.. seealso::
   
   * :doc:`/api/tasks/pretraining`
   * :doc:`/examples/pretraining_llama`