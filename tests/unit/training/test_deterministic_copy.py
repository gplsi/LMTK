"""Dependency-free behavioral checks; GPU acceptance is a separate gate."""
import ast
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def method(name):
    tree = ast.parse((ROOT / 'src/tasks/training/fabric/trainer/base.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FabricTrainerBase')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {'torch': SimpleNamespace(nn=SimpleNamespace(utils=SimpleNamespace(
        clip_grad_norm_=Mock(side_effect=AssertionError('Do not clip before Fabric')))))}
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), '<production-method>', 'exec'), ns)
    return ns[name]


class TrainingRegressionTests(unittest.TestCase):
    def run_group(self, count, group_size=16):
        trainer = SimpleNamespace(
            config=SimpleNamespace(gradient_accumulation_steps=group_size),
            cli_logger=Mock(), dataloaders={'train': list(range(count))},
            state={'iter_num': 0, 'step_count': 0, 'optimizer': Mock(), 'scheduler': Mock()},
            _gradient_clipping=Mock(), _log_learning_rates=Mock(), _try_validate=Mock(),
        )
        fabric = SimpleNamespace(no_backward_sync=lambda *a, **kw: nullcontext(), backward=Mock())
        model = SimpleNamespace(training_step=lambda batch, step: {'outputs': None, 'loss': float(step+1)})
        train = method('_accumulate_training')
        for step in range(count):
            train(trainer, fabric, model, None, step)
        return trainer, [c.args[0] for c in fabric.backward.call_args_list]

    def test_all_sixteen_microbatches_have_equal_coefficients(self):
        trainer, values = self.run_group(16)
        self.assertEqual(values, [(i+1)/16 for i in range(16)])
        self.assertEqual(trainer.state['optimizer'].step.call_count, 1)

    def test_partial_last_group_is_averaged_and_flushed(self):
        trainer, values = self.run_group(18)
        self.assertEqual(values[-2:], [17/2, 18/2])
        self.assertEqual(trainer.state['optimizer'].step.call_count, 2)

    def test_clipping_delegates_once_without_mutating_before_fabric(self):
        trainer = SimpleNamespace(config=SimpleNamespace(grad_clip=1.0), cli_logger=Mock())
        fabric = SimpleNamespace(clip_gradients=Mock(return_value=2.0))
        model, optimizer = object(), object()
        method('_gradient_clipping')(trainer, fabric, model, optimizer)
        fabric.clip_gradients.assert_called_once_with(model, optimizer, max_norm=1.0)


spec = importlib.util.spec_from_file_location('reproducibility', ROOT / 'src/tasks/training/reproducibility.py')
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


class ReproducibilityGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = {'task': 'clm_training', 'seed': 42, 'gradient_accumulation_steps': 16,
                       'parallelization_strategy': 'fsdp'}

    def test_fixed_seed_does_not_silently_accept_random_hashing(self):
        with patch.dict(os.environ, {'PYTHONHASHSEED': 'random'}):
            with self.assertRaises(RuntimeError):
                repro.prepare_process(self.config)

    def test_missing_seed_is_rejected(self):
        self.config.pop('seed')
        with self.assertRaises(ValueError):
            repro.prepare_process(self.config)

    def test_boolean_seed_is_rejected(self):
        self.config['seed'] = True
        with self.assertRaises(ValueError):
            repro.prepare_process(self.config)

    def test_resume_is_not_mislabeled_deterministic(self):
        self.config['checkpoint'] = 'old.pth'
        with patch.dict(os.environ, {'PYTHONHASHSEED': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            with self.assertRaises(ValueError):
                repro.prepare_process(self.config)

    def test_valid_fresh_run_configures_cublas_before_training_import(self):
        with patch.dict(os.environ, {'PYTHONHASHSEED': '0'}, clear=True):
            repro.prepare_process(self.config)
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_zero_accumulation_is_rejected(self):
        self.config['gradient_accumulation_steps'] = 0
        with self.assertRaises(ValueError):
            repro.prepare_process(self.config)

    def test_strict_algorithms_are_required_not_just_warnings(self):
        tree = ast.parse((ROOT / 'src/tasks/training/utils.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'deterministic')
        fake = Mock()
        ns = {'torch': fake, 'random': Mock(), 'np': Mock()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<production-seed>', 'exec'), ns)
        ns['deterministic'](42, strict=True)
        fake.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
        fake.cuda.manual_seed_all.assert_called_once_with(42)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)

    def test_scheduler_counts_the_partial_group_for_each_epoch(self):
        tree = ast.parse((ROOT / 'src/tasks/training/utils.py').read_text())
        parent = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select_scheduler')
        node = next(n for n in parent.body if isinstance(n, ast.FunctionDef))
        import math
        ns = {'math': math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<production-step-count>', 'exec'), ns)
        # 65 examples / 2 ranks => 33 batches/rank => 3 updates per epoch.
        warmup, total = ns['calculate_warmup_steps'](2, 2, 1, 0.1, list(range(65)), 16)
        self.assertEqual(total, 6)
        self.assertEqual(warmup, 0)


if __name__ == '__main__':
    unittest.main()
