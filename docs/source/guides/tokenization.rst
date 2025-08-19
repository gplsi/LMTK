.. _tokenization:

Tokenization Guide
================

.. raw:: html

   <div class="feature-box">
      <h3>Overview</h3>
      <p>This guide explains the tokenization capabilities of the ML Training Framework, including custom vocabulary building, specialized tokenizers for causal language models, and optimization techniques.</p>
   </div>

.. admonition:: Key Concepts
   :class: tip

   * **Tokenization**: The process of converting text into numerical token IDs
   * **Vocabulary**: A mapping between tokens and their numerical IDs
   * **BPE (Byte-Pair Encoding)**: An algorithm for subword tokenization
   * **WordPiece**: An alternative subword tokenization algorithm

Tokenization Architecture
-----------------------

The framework provides a modular tokenization system built around the ``BaseTokenizer`` class:

.. code-block:: text
   :caption: Tokenizer Class Hierarchy

   BaseTokenizer
   ├── CausalTokenizer
   │   ├── GPT2Tokenizer
   │   └── LlamaTokenizer
   └── CustomTokenizer

Core Components
-------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Tokenizer Config
      :class-card: sd-card

      .. code-block:: python

         class TokenizerConfig:
             vocab_size: int
             min_frequency: int = 2
             special_tokens: List[str] = ["<unk>", "<pad>"]
             unk_token: str = "<unk>"
             pad_token: str = "<pad>"
             algorithm: str = "bpe"  # or "wordpiece"

      Configuration for tokenizer training, including vocabulary size and special tokens.

   .. grid-item-card:: Training Process
      :class-card: sd-card

      .. code-block:: python

         from src.tasks.tokenization.orchestrator import train_tokenizer
         
         tokenizer = train_tokenizer(
             corpus_files=["data/corpus.txt"],
             config=config,
             output_dir="tokenizers/custom"
         )

      Process for training a tokenizer on a text corpus.

Causal Tokenization
-----------------

For causal language models, the framework offers specialized tokenizers that handle attention masking and positional information appropriately:

.. code-block:: python
   :caption: Using a Causal Tokenizer
   :linenos:

   from src.tasks.tokenization.tokenizer.causal import CausalTokenizer
   
   tokenizer = CausalTokenizer.from_file("tokenizers/llama/tokenizer.json")
   
   # Tokenize with attention mask
   encoded = tokenizer.encode(
       "Machine learning is fascinating",
       add_special_tokens=True,
       return_attention_mask=True
   )
   
   print(f"Token IDs: {encoded.ids}")
   print(f"Attention mask: {encoded.attention_mask}")

Performance Optimization
---------------------

The tokenization framework includes optimizations for handling large datasets efficiently:

.. tabs::

   .. tab:: Parallel Processing

      .. code-block:: python
         :linenos:
         
         from src.tasks.tokenization.tokenizer.utils import tokenize_corpus_parallel
         
         # Process a large corpus with parallel workers
         tokenized_dataset = tokenize_corpus_parallel(
             corpus_files=["data/large_corpus.txt"],
             tokenizer=tokenizer,
             max_length=512,
             num_workers=8
         )

   .. tab:: Memory-Efficient Processing

      .. code-block:: python
         :linenos:
         
         from src.tasks.tokenization.tokenizer.utils import tokenize_corpus_streaming
         
         # Process a corpus in streaming mode to reduce memory usage
         for batch in tokenize_corpus_streaming(
             corpus_files=["data/huge_corpus.txt"],
             tokenizer=tokenizer,
             batch_size=1000,
             max_length=512
         ):
             # Process batch
             pass

Configuration via YAML
--------------------

The tokenizer can be configured using YAML files for reproducible experiments:

.. code-block:: yaml
   :caption: Example Tokenizer Configuration
   :linenos:

   tokenizer:
     vocab_size: 32000
     min_frequency: 3
     special_tokens:
       - "<unk>"
       - "<pad>"
       - "<s>"
       - "</s>"
     unk_token: "<unk>"
     pad_token: "<pad>"
     bos_token: "<s>"
     eos_token: "</s>"
     algorithm: "bpe"
     normalization: "NFC"
     add_prefix_space: true
   
   training:
     corpus_files:
       - "data/train/*.txt"
     validation_files:
       - "data/validation/*.txt"
     batch_size: 1000
     num_workers: 8

Tokenizer Statistics
------------------

.. raw:: html

   <div class="feature-box">
      <h3>Vocabulary Coverage Analysis</h3>
   </div>

.. mermaid::

   pie
      title Token Type Distribution in GPT-2 Vocabulary
      "Whole Words" : 30
      "Subwords" : 60
      "Special Tokens" : 5
      "Character-level" : 5

Understanding the composition of your vocabulary is crucial for effective tokenization. The framework provides tools to analyze vocabulary statistics:

.. code-block:: python
   :linenos:

   from src.tasks.tokenization.tokenizer.utils import analyze_tokenizer
   
   # Generate vocabulary statistics
   stats = analyze_tokenizer("tokenizers/custom/tokenizer.json")
   
   print(f"Total vocabulary size: {stats['vocab_size']}")
   print(f"Average token length: {stats['avg_token_length']:.2f} characters")
   print(f"Token coverage on test set: {stats['coverage']:.2f}%")

Usage with Pretraining
--------------------

The tokenizer integrates seamlessly with the pretraining pipeline:

.. code-block:: python
   :caption: Integration with Pretraining
   :linenos:

   from src.tasks.clm_training.orchestrator import PretrainingOrchestrator
   from src.tasks.tokenization.tokenizer.causal import CausalTokenizer
   from src.config.config_loader import load_config
   
   # Load tokenizer
   tokenizer = CausalTokenizer.from_file("tokenizers/custom/tokenizer.json")
   
   # Load training config
   config = load_config("config/experiments/llama-3.2-3b/continual.yaml")
   
   # Initialize orchestrator with tokenizer
   orchestrator = PretrainingOrchestrator(
       config=config,
       tokenizer=tokenizer
   )
   
   # Start training
   orchestrator.train()

Next Steps
---------

* Explore :ref:`pretraining` to understand how tokenizers are used in model training
* Check out the :ref:`configuration` guide for advanced tokenizer configuration options
* See :doc:`/examples/tokenizer_custom` for a complete example of custom tokenizer training

.. seealso::
   
   * :doc:`/api/tasks/tokenization`