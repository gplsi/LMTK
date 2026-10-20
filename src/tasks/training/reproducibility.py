"""Strict, fresh-run CLM reproducibility on an unchanged execution stack."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform


def strict_enabled(config):
    return config.get('strict_determinism', config.get('task') == 'clm_training')


def prepare_process(config):
    """Called before importing the training stack; no torch imports here."""
    if not strict_enabled(config):
        return
    seed = config.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Strict CPT requires an integer seed in [0, 2**32).')
    accumulation = config.get('gradient_accumulation_steps')
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('Strict CPT requires gradient_accumulation_steps >= 1 (use 1 to disable accumulation).')
    hash_seed = os.environ.get('PYTHONHASHSEED', '')
    if not hash_seed.isdigit() or not 0 <= int(hash_seed) < 2**32:
        raise RuntimeError('Start Python with PYTHONHASHSEED=0 (or use scripts/run_deterministic.sh).')
    # Must be set before any CUDA context is created.
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in (':4096:8', ':16:8'):
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8.')
    if config.get('checkpoint') is not None or config.get('initial_weights_checkpoint') is not None:
        raise ValueError('Strict mode supports fresh from_pretrained runs only; checkpoint resume is not RNG-complete.')
    if config.get('task') != 'clm_training':
        raise ValueError('Strict mode is implemented for clm_training only.')
    if config.get('parallelization_strategy', 'fsdp') not in ('fsdp', 'ddp'):
        raise ValueError('Strict mode supports the existing fsdp/ddp paths only.')


def seed_worker(worker_id):
    """Seed worker-local Python/NumPy from the loader's independent generator."""
    import random
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def write_manifest(config, fabric, model, datasets):
    """Whitelist metadata: never dump credentials or the full environment."""
    import torch
    packages = {}
    for name in ('torch', 'lightning', 'transformers', 'datasets', 'numpy'):
        packages[name] = importlib.metadata.version(name)
    fields = ('seed', 'model_name', 'model_revision', 'precision', 'parallelization_strategy',
              'batch_size', 'gradient_accumulation_steps', 'num_workers', 'number_epochs',
              'lr', 'lr_scheduler', 'warmup_proportion', 'weight_decay', 'beta1', 'beta2',
              'grad_clip', 'validation_split', 'train_data_ratio')
    metadata = {
        'python': platform.python_version(), 'packages': packages,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'world_size': fabric.world_size, 'rank': fabric.global_rank,
        'python_hash_probe': hash('LMTK-deterministic'),
        'device': str(fabric.device),
        'gpu': torch.cuda.get_device_name(fabric.device) if fabric.device.type == 'cuda' else None,
        'environment': {k: os.environ.get(k) for k in
                        ('PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'NCCL_ALGO', 'NCCL_PROTO')},
        'training': {k: config.get(k) for k in fields},
        'datasets': {k: {'rows': len(v), 'fingerprint': v._fingerprint} for k, v in datasets.items()},
        'model_commit': getattr(model.model.config, '_commit_hash', None),
        'attention': getattr(model.model.config, '_attn_implementation', None),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
    }
    path = Path(config.output_dir) / f'reproducibility-rank-{fabric.global_rank}.json'
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
