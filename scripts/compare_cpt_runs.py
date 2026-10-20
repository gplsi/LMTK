"""Compare corresponding trusted LMTK full checkpoints and run manifests.

Exit 0 only if tensors are bitwise equal, finite, and metadata matches.
This requires PyTorch; it does not compare archive bytes or timing/WandB logs.
"""
import argparse
import json
from pathlib import Path
import struct

import torch


def compare(a, b, path='state'):
    if type(a) is not type(b):
        raise AssertionError(f'{path}: different types')
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or a.layout != b.layout:
            raise AssertionError(f'{path}: different tensor metadata')
        if a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise AssertionError(f'{path}: non-finite values')
        # Byte views also distinguish positive/negative zero. Flatten scalar states.
        av = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bv = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(av, bv):
            raise AssertionError(f'{path}: tensor bytes differ')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f'{path}: different keys')
        for key in a:
            compare(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'{path}: different lengths')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]')
    elif isinstance(a, float):
        if struct.pack('!d', a) != struct.pack('!d', b):
            raise AssertionError(f'{path}: different floats')
    elif a != b:
        raise AssertionError(f'{path}: different values')


def compare_runs(first, second):
    manifests = sorted(p.name for p in first.glob('reproducibility-rank-*.json'))
    if not manifests or manifests != sorted(p.name for p in second.glob('reproducibility-rank-*.json')):
        raise AssertionError('Missing or different rank manifests')
    for name in manifests:
        left = json.loads((first / name).read_text())
        right = json.loads((second / name).read_text())
        if not left['deterministic_algorithms'] or left['warn_only']:
            raise AssertionError('Run was not in strict deterministic mode')
        compare(left, right, name)
    checkpoints = sorted(p.name for p in first.glob('e-*.pth'))
    if not checkpoints or checkpoints != sorted(p.name for p in second.glob('e-*.pth')):
        raise AssertionError('Missing or different checkpoint sets')
    for name in checkpoints:
        # Safe loader: do not silently fall back to arbitrary pickle execution.
        a = torch.load(first / name, map_location='cpu', weights_only=True)
        b = torch.load(second / name, map_location='cpu', weights_only=True)
        for key in ('model', 'optimizer', 'scheduler', 'step_count', 'iter_num', 'current_epoch'):
            if key not in a or key not in b:
                raise AssertionError(f'{name}: missing {key}')
            compare(a[key], b[key], f'{name}.{key}')
        print(f'PASS {name}: model, optimizer, scheduler and counters match exactly')
        del a, b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    compare_runs(args.first, args.second)
