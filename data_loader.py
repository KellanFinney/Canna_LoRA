import json
import os
from pathlib import Path
from datasets import Dataset
from typing import List, Dict, Any
import logging
from config_loader import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_science_training_data(data_dir: str = None) -> Dataset:
    """
    Load and flatten the nested JSON structure from science training files.
    
    Args:
        data_dir: Directory containing the JSON files. If None, uses config.
    
    Expected structure:
    {
      "PDF_TITLE": {
        "chunk_N": {
          "generated": [{"question": "...", "answer": "..."}],
          "context": "source text",
          "source_pdf": "filename"
        }
      }
    }
    """
    
    # Use config if data_dir not provided
    if data_dir is None:
        config = get_config()
        data_dir = config.data_generation.get('output_directory', './data')
        batch_pattern = config.data_loading.get('batch_file_pattern', 'science_training_batch_*.json')
    else:
        batch_pattern = 'science_training_batch_*.json'
    
    data_path = Path(data_dir)
    all_qa_pairs = []
    
    # Use cross-platform path globbing
    json_files = list(data_path.glob(batch_pattern))
    
    logger.info(f"Found {len(json_files)} JSON files in {data_path}")
    
    for json_file in json_files:
        logger.info(f"Processing {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten the nested structure
            for pdf_title, pdf_data in data.items():
                for chunk_id, chunk_data in pdf_data.items():
                    if 'generated' in chunk_data:
                        for qa_pair in chunk_data['generated']:
                            # Extract Q&A and add metadata
                            flattened_item = {
                                'question': qa_pair['question'],
                                'answer': qa_pair['answer'],
                                'pdf_title': pdf_title,
                                'chunk_id': chunk_id,
                                'context': chunk_data.get('context', ''),
                                'source_pdf': chunk_data.get('source_pdf', ''),
                                'batch_file': json_file.name
                            }
                            all_qa_pairs.append(flattened_item)
        
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue
    
    logger.info(f"Total Q&A pairs extracted: {len(all_qa_pairs)}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(all_qa_pairs)
    
    # Log some statistics
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Unique PDFs: {len(set(dataset['pdf_title']))}")
    logger.info(f"Sample question: {dataset[0]['question'][:100]}...")
    
    return dataset

def create_train_test_split(dataset: Dataset, test_size: float = None, seed: int = None):
    """Create train/test split from the dataset."""
    
    # Use config defaults if not provided
    if test_size is None or seed is None:
        config = get_config()
        test_size = test_size or config.data_loading.get('test_size', 0.1)
        seed = seed or config.data_loading.get('random_seed', 42)
    
    logger.info(f"Creating train/test split with test_size={test_size}")
    
    # Split the dataset
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def preview_dataset(dataset: Dataset, num_samples: int = 3):
    """Preview the first few samples from the dataset."""
    
    print("\n" + "="*80)
    print("DATASET PREVIEW")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"PDF: {sample['pdf_title'][:50]}...")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer'][:200]}...")
        print(f"Source: {sample['source_pdf']}")
        print("-" * 40)

if __name__ == "__main__":
    # Test the data loader
    data_dir = "./data"  # Adjust path as needed
    
    if os.path.exists(data_dir):
        dataset = load_science_training_data(data_dir)
        preview_dataset(dataset)
        
        # Test train/test split
        train_ds, test_ds = create_train_test_split(dataset)
        print(f"\nSplit successful: {len(train_ds)} train, {len(test_ds)} test")
    else:
        print(f"Data directory {data_dir} not found. Please create it and add your JSON files.") 