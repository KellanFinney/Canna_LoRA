"""
HuggingFace dataset integration utilities for LoRA DataGen
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
from config_loader import get_config
import logging

logger = logging.getLogger(__name__)

class HuggingFaceIntegration:
    """Utility class for HuggingFace dataset operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.hf_config = self.config.huggingface
        self.api = HfApi()
    
    def prepare_dataset_from_batches(self, data_dir: str = None) -> Dataset:
        """
        Load and prepare dataset from generated batch files
        
        Args:
            data_dir: Directory containing batch files. If None, uses config.
            
        Returns:
            HuggingFace Dataset ready for upload
        """
        from data_loader import load_science_training_data
        
        logger.info("Loading dataset from batch files...")
        dataset = load_science_training_data(data_dir)
        
        # Add dataset metadata
        dataset_info = {
            'description': self.hf_config.get('description', 'Synthetic Q&A dataset'),
            'tags': self.hf_config.get('tags', []),
            'generated_by': 'LoRA DataGen',
            'total_samples': len(dataset)
        }
        
        # Add metadata as dataset info
        dataset.info.description = dataset_info['description']
        dataset.info.features_metadata = dataset_info
        
        return dataset
    
    def create_train_test_split(self, dataset: Dataset) -> DatasetDict:
        """Create train/test split for HuggingFace upload"""
        from data_loader import create_train_test_split
        
        train_dataset, test_dataset = create_train_test_split(dataset)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    
    def upload_to_hub(
        self, 
        dataset: Dataset | DatasetDict, 
        repo_name: str = None,
        private: bool = False,
        token: str = None
    ) -> str:
        """
        Upload dataset to HuggingFace Hub
        
        Args:
            dataset: Dataset or DatasetDict to upload
            repo_name: Repository name. If None, uses config.
            private: Whether to make repository private
            token: HF token. If None, uses environment variable.
            
        Returns:
            Repository URL
        """
        if repo_name is None:
            repo_name = self.hf_config.get('dataset_repo')
            if not repo_name:
                raise ValueError("Repository name must be provided in config or as parameter")
        
        logger.info(f"Uploading dataset to HuggingFace Hub: {repo_name}")
        
        # Login if token provided
        if token:
            login(token=token)
        
        # Upload dataset
        dataset.push_to_hub(
            repo_name,
            private=private,
            token=token
        )
        
        repo_url = f"https://huggingface.co/datasets/{repo_name}"
        logger.info(f"Dataset successfully uploaded: {repo_url}")
        
        return repo_url
    
    def create_dataset_card(self, dataset: Dataset, output_path: str = "README.md") -> str:
        """
        Generate a dataset card for HuggingFace
        
        Args:
            dataset: The dataset to create a card for
            output_path: Where to save the dataset card
            
        Returns:
            Generated dataset card content
        """
        config = self.hf_config
        
        # Calculate statistics
        total_samples = len(dataset)
        unique_pdfs = len(set(dataset['pdf_title'])) if 'pdf_title' in dataset.column_names else 0
        avg_question_length = sum(len(q.split()) for q in dataset['question']) / total_samples
        avg_answer_length = sum(len(a.split()) for a in dataset['answer']) / total_samples
        
        card_content = f"""---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
{chr(10).join(f'- {tag}' for tag in config.get('tags', []))}
size_categories:
- {self._get_size_category(total_samples)}
---

# {config.get('dataset_repo', 'LoRA DataGen Dataset')}

## Dataset Description

{config.get('description', 'High-quality synthetic Q&A dataset for LoRA fine-tuning')}

This dataset was generated using [LoRA DataGen](https://github.com/nickrenotte/lora-datagen), an automated pipeline for creating instruction-tuning datasets from PDF documents.

## Dataset Statistics

- **Total Samples**: {total_samples:,}
- **Source PDFs**: {unique_pdfs}
- **Average Question Length**: {avg_question_length:.1f} words
- **Average Answer Length**: {avg_answer_length:.1f} words

## Dataset Structure

Each sample contains:
- `question`: The generated question
- `answer`: The corresponding answer
- `context`: Original text chunk from PDF
- `pdf_title`: Source PDF title
- `chunk_id`: Chunk identifier within PDF
- `source_pdf`: Source PDF filename
- `batch_file`: Generation batch file

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{config.get('dataset_repo', 'your-repo')}")

# Access train/test splits
train_data = dataset['train']
test_data = dataset['test']

# Example sample
print(train_data[0])
```

## Generation Process

1. **PDF Processing**: Documents parsed using Docling
2. **Text Chunking**: Intelligent chunking with context preservation
3. **Q&A Generation**: OpenAI GPT models generate diverse Q&A pairs
4. **Quality Assessment**: Local LLM filtering for high-quality samples

## Quality Metrics

- Minimum accuracy score: 6/10
- Minimum style score: 6/10
- Human-reviewed prompt engineering
- Context-aware generation

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{lora_datagen_dataset,
  author = {{Nick Renotte}},
  title = {{{config.get('dataset_repo', 'LoRA DataGen Dataset')}}},
  year = {{2024}},
  url = {{https://huggingface.co/datasets/{config.get('dataset_repo', 'your-repo')}}},
  note = {{Generated using LoRA DataGen pipeline}}
}}
```

## License

This dataset is released under the MIT License.
"""
        
        # Save to file
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        logger.info(f"Dataset card saved to: {output_path}")
        return card_content
    
    def _get_size_category(self, num_samples: int) -> str:
        """Determine HuggingFace size category"""
        if num_samples < 1000:
            return "n<1K"
        elif num_samples < 10000:
            return "1K<n<10K"
        elif num_samples < 100000:
            return "10K<n<100K"
        elif num_samples < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"


def main():
    """Example usage of HuggingFace integration"""
    hf = HuggingFaceIntegration()
    
    # Load dataset from batches
    dataset = hf.prepare_dataset_from_batches()
    
    # Create train/test split
    dataset_dict = hf.create_train_test_split(dataset)
    
    # Generate dataset card
    hf.create_dataset_card(dataset_dict['train'])
    
    print(f"Dataset prepared with {len(dataset_dict['train'])} training samples")
    print(f"and {len(dataset_dict['test'])} test samples")
    print("Ready for upload to HuggingFace Hub!")


if __name__ == "__main__":
    main() 