
# src/tasks/tokenization.py
def execute(config):
    print(f"\n=== Tokenization Configuration ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Special Tokens: {config.special_tokens}")
    print(f"Dataset Path: {config.dataset.path}")