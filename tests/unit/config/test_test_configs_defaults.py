"""Tests for alignment between test defaults and smoke configs."""

import unittest
from pathlib import Path

import yaml


class TestTestConfigDefaults(unittest.TestCase):
    """Ensure smoke test configs match config/tests/defaults.yaml."""

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.defaults_path = repo_root / "config/tests/defaults.yaml"
        self.tokenization_path = repo_root / "config/tests/tokenization_smoke.yaml"
        self.training_path = repo_root / "config/tests/clm_training_smoke.yaml"
        self.tokenization_doclevel_path = repo_root / "config/tests/tokenization_doclevel_smoke.yaml"
        self.training_packing_path = repo_root / "config/tests/clm_training_packing_smoke.yaml"

    def test_tokenization_smoke_matches_defaults(self) -> None:
        defaults = self._load_yaml(self.defaults_path)
        tokenization = self._load_yaml(self.tokenization_path)

        self.assertEqual(
            tokenization["tokenizer"]["tokenizer_name"],
            defaults["tokenizer_name"],
        )
        self.assertEqual(tokenization["tokenizer"]["context_length"], defaults["context_length"])
        self.assertEqual(tokenization["tokenizer"]["overlap"], defaults["overlap"])
        self.assertEqual(tokenization["seed"], defaults["seed"])

    def test_tokenization_doclevel_smoke_matches_defaults_and_omits_size(self) -> None:
        defaults = self._load_yaml(self.defaults_path)
        tokenization = self._load_yaml(self.tokenization_doclevel_path)

        self.assertEqual(
            tokenization["tokenizer"]["tokenizer_name"],
            defaults["tokenizer_name"],
        )
        self.assertEqual(tokenization["seed"], defaults["seed"])

        # Doc-level tokenization intentionally omits size/overlap controls.
        self.assertNotIn("context_length", tokenization["tokenizer"])
        self.assertNotIn("max_sequence_length", tokenization["tokenizer"])
        self.assertNotIn("overlap", tokenization["tokenizer"])

    def test_clm_training_smoke_matches_defaults(self) -> None:
        defaults = self._load_yaml(self.defaults_path)
        training = self._load_yaml(self.training_path)

        self.assertEqual(training["model_name"], defaults["model_name"])
        self.assertEqual(training["precision"], defaults["precision"])
        self.assertEqual(training["seed"], defaults["seed"])

    def test_clm_training_packing_smoke_matches_defaults(self) -> None:
        defaults = self._load_yaml(self.defaults_path)
        training = self._load_yaml(self.training_packing_path)

        self.assertEqual(training["model_name"], defaults["model_name"])
        self.assertEqual(training["precision"], defaults["precision"])
        self.assertEqual(training["seed"], defaults["seed"])


if __name__ == "__main__":
    unittest.main()
