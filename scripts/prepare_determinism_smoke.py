"""Create a local tiny Llama, fixed tokenized data, and two fresh-run configs.

Run inside the same container used for training. No model download is needed.
This prepares files only and never submits SLURM jobs.
"""
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM
import yaml


def prepare(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
    ))
    model.save_pretrained(directory / 'model')
    tokens = torch.randint(3, 128, (66, 128)).tolist()
    Dataset.from_dict({'input_ids': tokens, 'attention_mask': [[1]*128 for _ in tokens],
                       'labels': tokens}).save_to_disk(str(directory / 'dataset'))
    config = {
        'task': 'clm_training', 'experiment_name': 'determinism-smoke', 'verbose_level': 2,
        'dataset': {'source': 'local', 'nameOrPath': str(directory / 'dataset'), 'format': 'hf'},
        'validation_split': {'count': 2, 'seed': 42, 'shuffle': True},
        'seed': 42, 'strict_determinism': True, 'model_name': str(directory / 'model'),
        'precision': 'bf16-true', 'parallelization_strategy': 'fsdp',
        'number_epochs': 2, 'batch_size': 1, 'num_workers': 4,
        'gradient_accumulation_steps': 16, 'gradient_accumulation': True, 'grad_clip': 1.0,
        'lr': 2e-5, 'weight_decay': 0.01, 'beta1': 0.9, 'beta2': 0.999,
        'lr_scheduler': 'warmup_linear', 'warmup_proportion': 0.1,
        'validate_on_end': True, 'validate_after_epoch': True, 'validate_after_k_steps': None,
        'save_on_validate': True, 'save_on_end': True,
        'gradient_checkpointing': True, 'logging_config': 'none', 'log_iter_interval': 1,
        'cpu_offload': False, 'sharding_strategy': 'FULL_SHARD',
        'state_dict_type': 'full', 'limit_all_gathers': True,
    }
    for name in ('a', 'b'):
        config['output_dir'] = str(directory / f'run-{name}')
        path = directory / f'run-{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    prepare(parser.parse_args().directory)
