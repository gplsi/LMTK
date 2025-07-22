#!/usr/bin/env python3
"""
Tokenized Dataset Visualization Script

This script visualizes samples from tokenized datasets across all tokenization subtasks
(CLM, MLM, instruction tuning). It loads tokenized datasets, extracts N samples from
each split, and saves them as properly formatted JSON files with raw token information.

Usage:
    python visualize_tokenized_dataset.py --dataset_path /path/to/tokenized/dataset --num_samples 5 --output_dir ./visualizations

Features:
- Works with all tokenization subtasks (CLM, MLM, instruction)
- Handles DatasetDict (multiple splits) and single Dataset
- Shows raw token IDs and basic statistics
- Saves formatted JSON with metadata
- Creates separate files for each split
- No tokenizer dependencies - just shows raw tokens
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import sys

try:
    from datasets import load_from_disk, DatasetDict, Dataset
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install datasets")
    sys.exit(1)


class TokenizedDatasetVisualizer:
    """
    Visualizer for tokenized datasets across all tokenization subtasks.
    
    This class provides methods to load tokenized datasets, extract samples,
    and save formatted visualizations with raw token information.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the visualizer.
        
        Args:
            dataset_path: Path to the tokenized dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.dataset = None
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the tokenized dataset from disk."""
        try:
            self.dataset = load_from_disk(str(self.dataset_path))
            print(f"✅ Loaded dataset from: {self.dataset_path}")
            
            if isinstance(self.dataset, DatasetDict):
                print(f"📁 Dataset splits: {list(self.dataset.keys())}")
                for split, ds in self.dataset.items():
                    print(f"   {split}: {len(ds)} samples")
            else:
                print(f"📄 Single dataset: {len(self.dataset)} samples")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)
    
    def _convert_to_list(self, data: Any) -> List[int]:
        """Convert tensor or other data types to Python list."""
        if hasattr(data, 'tolist'):  # PyTorch tensor or numpy array
            return data.tolist()
        elif isinstance(data, list):
            return data
        else:
            return list(data)
    
    def _analyze_tokens(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basic token statistics in the sample."""
        analysis = {}
        
        if 'input_ids' in sample:
            input_ids = self._convert_to_list(sample['input_ids'])
            analysis['total_tokens'] = len(input_ids)
            analysis['unique_tokens'] = len(set(input_ids))
            analysis['min_token_id'] = min(input_ids) if len(input_ids) > 0 else 0
            analysis['max_token_id'] = max(input_ids) if len(input_ids) > 0 else 0
        
        if 'labels' in sample:
            labels = self._convert_to_list(sample['labels'])
            masked_count = sum(1 for label in labels if label == -100)
            analysis['masked_labels'] = masked_count
            analysis['training_tokens'] = len(labels) - masked_count
            if analysis['training_tokens'] > 0:
                training_labels = [label for label in labels if label != -100]
                analysis['unique_training_tokens'] = len(set(training_labels))
        
        if 'attention_mask' in sample:
            attention_mask = self._convert_to_list(sample['attention_mask'])
            analysis['attention_tokens'] = sum(attention_mask)
            analysis['padding_tokens'] = len(attention_mask) - sum(attention_mask)
        
        return analysis
    
    def _format_sample(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """Format a single sample for JSON output."""
        formatted = {
            'sample_index': sample_idx,
            'metadata': self._analyze_tokens(sample),
            'tokenization': {}
        }
        
        # Add raw tokenization data (convert tensors to lists)
        for key, value in sample.items():
            if hasattr(value, 'tolist'):  # PyTorch tensor or numpy array
                formatted['tokenization'][key] = value.tolist()
            elif isinstance(value, list):
                formatted['tokenization'][key] = value
            else:
                formatted['tokenization'][key] = str(value)
        
        # Add token previews for easier inspection
        if 'input_ids' in sample:
            input_ids = self._convert_to_list(sample['input_ids'])
            formatted['token_preview'] = {
                'first_10_tokens': input_ids[:10],
                'last_10_tokens': input_ids[-10:] if len(input_ids) > 10 else input_ids
            }
            
            # Show training tokens for MLM/instruction tasks
            if 'labels' in sample:
                labels = self._convert_to_list(sample['labels'])
                training_positions = [i for i, label in enumerate(labels) if label != -100]
                if training_positions:
                    formatted['token_preview']['training_positions'] = training_positions[:10]
                    formatted['token_preview']['training_tokens'] = [labels[i] for i in training_positions[:10]]
        
        return formatted
    
    def visualize_samples(self, num_samples: int, output_dir: str) -> None:
        """
        Extract and visualize samples from the tokenized dataset.
        
        Args:
            num_samples: Number of samples to extract from each split
            output_dir: Directory to save the visualization JSON files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 Extracting {num_samples} samples per split...")
        print(f"💾 Output directory: {output_path}")
        
        if isinstance(self.dataset, DatasetDict):
            # Handle multiple splits
            for split_name, split_dataset in self.dataset.items():
                self._visualize_split(split_dataset, split_name, num_samples, output_path)
        else:
            # Handle single dataset
            self._visualize_split(self.dataset, "dataset", num_samples, output_path)
        
        print(f"\n✅ Visualization complete! Files saved in: {output_path}")
    
    def _visualize_split(self, dataset: Dataset, split_name: str, num_samples: int, output_path: Path) -> None:
        """Visualize samples from a single dataset split."""
        # Limit samples to available data
        actual_samples = min(num_samples, len(dataset))
        
        print(f"\n📊 Processing split '{split_name}' ({actual_samples}/{len(dataset)} samples)")
        
        # Extract samples
        samples = dataset[:actual_samples]
        
        # Format samples for JSON output
        formatted_samples = []
        for i in range(actual_samples):
            sample = {key: values[i] for key, values in samples.items()}
            formatted_sample = self._format_sample(sample, i)
            formatted_samples.append(formatted_sample)
        
        # Create output data
        output_data = {
            'dataset_info': {
                'dataset_path': str(self.dataset_path),
                'split_name': split_name,
                'total_samples': len(dataset),
                'visualized_samples': actual_samples,
                'sample_keys': list(samples.keys()) if samples else []
            },
            'samples': formatted_samples
        }
        
        # Save to JSON file
        output_file = output_path / f"{split_name}_samples.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Saved: {output_file}")
        
        # Print sample preview
        if formatted_samples:
            sample = formatted_samples[0]
            print(f"   📝 Sample preview:")
            print(f"      Total tokens: {sample['metadata'].get('total_tokens', 'N/A')}")
            print(f"      Training tokens: {sample['metadata'].get('training_tokens', 'N/A')}")
            if 'token_preview' in sample:
                preview = sample['token_preview']
                print(f"      First tokens: {preview.get('first_10_tokens', [])}")
                if 'training_tokens' in preview:
                    print(f"      Training tokens: {preview.get('training_tokens', [])}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Visualize samples from tokenized datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize 5 samples from each split
  python visualize_tokenized_dataset.py --dataset_path /workspace/output/tokenized_bert_mlm --num_samples 5
  
  # Custom output directory
  python visualize_tokenized_dataset.py --dataset_path /workspace/output/tokenized_data --num_samples 10 --output_dir ./my_visualizations
  
  # Quick preview with 3 samples
  python visualize_tokenized_dataset.py --dataset_path /workspace/output/tokenized_data --num_samples 3
        """
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the tokenized dataset directory'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to visualize from each split (default: 5)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./tokenized_visualizations',
        help='Output directory for visualization files (default: ./tokenized_visualizations)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_samples <= 0:
        print("❌ Error: num_samples must be positive")
        exit(1)
    
    if not Path(args.dataset_path).exists():
        print(f"❌ Error: Dataset path does not exist: {args.dataset_path}")
        exit(1)
    
    print("🚀 Tokenized Dataset Visualizer")
    print("=" * 50)
    
    try:
        # Create visualizer and process dataset
        visualizer = TokenizedDatasetVisualizer(dataset_path=args.dataset_path)
        
        visualizer.visualize_samples(
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
