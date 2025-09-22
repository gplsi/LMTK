"""Tests for version utilities."""

import unittest
from unittest.mock import patch, mock_open

from src.utils.version import get_version, get_version_info, display_version_info


class TestVersionUtils(unittest.TestCase):
    """Test suite for version utility functions."""

    @patch("importlib.metadata.version")
    def test_get_version_from_metadata(self, mock_version):
        """Test version retrieval from package metadata."""
        mock_version.return_value = "0.2.0"
        version = get_version()
        self.assertEqual(version, "0.2.0")
        mock_version.assert_called_once_with("continual-pretrain")

    @patch("importlib.metadata.version")
    @patch("importlib.metadata.PackageNotFoundError", new=Exception)
    @patch("tomli.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_get_version_from_pyproject(self, mock_exists, mock_file, mock_tomli, mock_version):
        """Test version retrieval from pyproject.toml."""
        # Setup mock to raise exception for importlib.metadata.version
        mock_version.side_effect = Exception("Package not found")
        
        # Setup mock for pyproject.toml file existence and content
        mock_exists.return_value = True
        mock_tomli.return_value = {
            "tool": {
                "poetry": {
                    "version": "0.1.0"
                }
            }
        }
        
        version = get_version()
        self.assertEqual(version, "0.1.0")

    @patch("platform.python_version")
    @patch("platform.platform")
    @patch("src.utils.version.get_version")
    def test_get_version_info(self, mock_get_version, mock_platform, mock_py_version):
        """Test detailed version info gathering."""
        mock_get_version.return_value = "0.1.0"
        mock_platform.return_value = "Linux-5.4.0-X86_64"
        mock_py_version.return_value = "3.10.4"
        
        with patch("torch.__version__", "2.1.0"):
            with patch("transformers.__version__", "4.39.3"):
                with patch("torch.cuda.is_available", return_value=False):
                    info = get_version_info()
        
        self.assertEqual(info["version"], "0.1.0")
        self.assertEqual(info["python"], "3.10.4")
        self.assertEqual(info["torch"], "2.1.0")
        self.assertEqual(info["transformers"], "4.39.3")
        self.assertNotIn("cuda", info)
        self.assertNotIn("gpu", info)

    @patch("src.utils.version.get_version_info")
    def test_display_version_info(self, mock_get_info):
        """Test version info display formatting."""
        mock_get_info.return_value = {
            "version": "0.1.0",
            "python": "3.10.4",
            "platform": "Linux-x86_64",
            "torch": "2.1.0",
            "transformers": "4.39.3"
        }
        
        # Test with custom file object
        from io import StringIO
        output = StringIO()
        display_version_info(file=output)
        
        output_text = output.getvalue()
        self.assertIn("Continual Pretraining Framework", output_text)
        self.assertIn("version   : 0.1.0", output_text)
        self.assertIn("python    : 3.10.4", output_text)
        self.assertIn("torch     : 2.1.0", output_text)


if __name__ == "__main__":
    unittest.main()