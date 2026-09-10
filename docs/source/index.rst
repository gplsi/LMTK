.. ML Training Framework documentation master file

.. raw:: html

   <div class="hero-section">
      <h1>ML Training Framework</h1>
      <p>A robust, scalable framework for machine learning model training and tokenization with support for distributed training and advanced optimization techniques.</p>
   </div>

.. grid:: 3

    .. grid-item-card:: 📊 Pretraining
        :link: guides/pretraining
        :link-type: doc
        :class-card: feature-box

        Comprehensive tools for pretraining language models with distributed training support via FSDP, DDP, and DeepSpeed.

    .. grid-item-card:: 🔤 Tokenization
        :link: guides/tokenization
        :link-type: doc
        :class-card: feature-box

        Advanced tokenization utilities with support for causal language models and custom vocabulary building.

    .. grid-item-card:: 🚀 Scalability
        :link: guides/scaling
        :link-type: doc
        :class-card: feature-box

        Optimized for performance at scale with support for multi-node training and monitoring.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   guides/installation
   guides/quickstart
   guides/configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guides
   :hidden:

   guides/pretraining
   guides/tokenization
   guides/distributed
   guides/monitoring
   guides/optimization
   guides/scaling

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/tasks/pretraining
   api/tasks/tokenization
   api/config/index
   api/utils/index

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/pretraining_llama
   examples/tokenizer_custom
   examples/distributed_training

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/contributing
   development/testing
   development/release_notes

.. raw:: html

   <div class="feature-box">
      <h3>✨ Key Features</h3>
      <ul>
         <li>Distributed training with FSDP, DDP, and DeepSpeed</li>
         <li>Custom tokenizer implementation with BPE and WordPiece support</li>
         <li>Comprehensive configuration management via YAML</li>
         <li>Integrated performance monitoring and logging</li>
         <li>Modular architecture for extensibility</li>
      </ul>
   </div>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`