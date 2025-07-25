# LoRA DataGen + Training Pipeline 📚→🤖→🎯

> **Complete end-to-end LoRA fine-tuning pipeline: From PDFs to trained models**
> 
> *Adapted from [Nick Renotte's Parameter Efficient Fine Tuning project](https://github.com/nickrenotte/lora-finetuning)*

Transform your PDF documents into instruction-tuning datasets and train high-quality LoRA models, all in one comprehensive pipeline.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Complete Pipeline

This project provides a full LoRA fine-tuning workflow:

1. **📄 PDF Processing**: Extracts and chunks text from PDFs using intelligent document parsing
2. **🤖 AI Generation**: Creates diverse, contextual Q&A pairs using OpenAI's GPT models
3. **🔍 Quality Control**: Filters generated content using local LLM quality assessment
4. **📊 Dataset Creation**: Outputs HuggingFace-compatible datasets ready for training
5. **🎯 LoRA Training**: Train models using the generated datasets with optimized configurations
6. **🚀 Model Deployment**: Ready-to-use trained LoRA adapters

## 📜 Attribution & License

This project is an adapted and enhanced version of the data generation components from [Nick Renotte's LoRA Fine-tuning Tutorial](https://github.com/nickrenotte/lora-finetuning). 

**Original Author**: Nick Renotte  
**Adaptations by**: Kellan Finney

### What We Modified:
- ✅ **Security improvements**: Removed hardcoded API keys, added environment variable support
- ✅ **Configuration management**: Added YAML-based configuration system
- ✅ **Cross-platform compatibility**: Fixed path handling for Windows/Linux/macOS
- ✅ **Enhanced documentation**: Comprehensive README and setup instructions
- ✅ **HuggingFace integration**: Added utilities for dataset publishing
- ✅ **Project structure**: Modern Python packaging with pyproject.toml
- ✅ **Error handling**: Improved resilience and error reporting

Both the original and this adapted version are licensed under the MIT License.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Ollama installed for quality assessment ([install guide](https://ollama.ai/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KellanFinney/lora-datagen.git
   cd lora-datagen
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script**
   ```bash
   python setup.py
   ```

4. **Set up your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

5. **Configure your settings**
   ```bash
   # Edit the generated config.yaml file with your preferences
   nano config.yaml
   ```

### Basic Usage

1. **Place your PDFs** in the configured directory (default: `./pdfs/`)

2. **Generate synthetic data**
   ```bash
   python syntheticdatageneration_openai.py
   ```

3. **Assess quality** (optional but recommended)
   ```bash
   python dataquality.py
   ```

4. **Load and preview your dataset**
   ```bash
   python data_loader.py
   ```

5. **Train your LoRA model**
   ```bash
   python train.py
   ```

### Training Details

The training script (`train.py`) includes:
- **IBM Granite 3.3-8B** as base model (configurable)
- **Optimized LoRA configuration** (r=128, alpha=256)
- **4-bit quantization** for memory efficiency
- **Chat template formatting** for instruction following
- **Comprehensive evaluation** and checkpointing
- **TensorBoard logging** for monitoring

## 🎯 Training Configuration

The training script supports various configurations:

### Model Selection
```python
# In train.py, modify base_model:
base_model = "/workspace/models/granite-3.3-8b-instruct"  # Default
# Or use other models:
# base_model = "microsoft/DialoGPT-medium"
# base_model = "meta-llama/Llama-2-7b-hf"
```

### LoRA Parameters
```python
peft_config = LoraConfig(
    r=128,                    # Rank (adjust based on model size)
    lora_alpha=256,          # Alpha scaling
    lora_dropout=0.1,        # Dropout for regularization
    target_modules="all-linear",  # Target all linear layers
)
```

### Training Hyperparameters
```python
training_args = SFTConfig(
    num_train_epochs=3,           # Number of epochs
    per_device_train_batch_size=2, # Batch size per GPU
    learning_rate=1e-4,           # Learning rate
    max_seq_length=2048,          # Maximum sequence length
)
```

## 📁 Project Structure

```
lora-datagen/
├── 📄 syntheticdatageneration_openai.py  # Main data generation script
├── 📊 data_loader.py                     # Dataset loading and processing
├── 🔍 dataquality.py                     # Quality assessment with local LLM
├── 📝 generated_prompt.py                # Prompt template for Q&A generation
├── 🎯 train.py                           # LoRA training script
├── ⚙️ config_loader.py                   # Configuration management
├── 📋 config.yaml                        # Main configuration file
├── 📦 pyproject.toml                     # Project metadata and dependencies
├── 📚 README.md                          # This file
└── 🗂️ science_training_data.json         # Example output format
```

## ⚙️ Configuration

Edit `config.yaml` to customize your pipeline:

```yaml
# Data generation settings
data_generation:
  pdf_directory: "./pdfs"           # Where your PDFs are stored
  output_directory: "./data"        # Where to save generated datasets
  pdfs_per_batch: 5                # PDFs per JSON file
  qa_pairs_per_chunk: 5            # Q&A pairs per text chunk

# OpenAI settings  
openai:
  model: "gpt-4o-mini"             # Model to use (gpt-4o-mini recommended)
  temperature: 0.7                 # Response creativity (0.0-1.0)
  rate_limit: 490                  # Requests per minute

# Processing settings
processing:
  max_workers: 30                  # Parallel processing threads
  resume_from_existing: true       # Resume interrupted batches

# Quality assessment
quality:
  local_model: "ollama_chat/qwen2.5:14b"  # Local LLM for quality check
  min_accuracy_score: 6            # Minimum quality threshold (1-10)
  min_style_score: 6              # Minimum style threshold (1-10)
```

## 📊 Output Format

Generated datasets follow this structure:

```json
{
  "PDF_TITLE": {
    "chunk_0": {
      "generated": [
        {
          "question": "What is the main topic discussed?",
          "answer": "The document discusses..."
        }
      ],
      "context": "Original text chunk...",
      "source_pdf": "document.pdf"
    }
  }
}
```

## 🔧 Advanced Usage

### Custom Prompts

Edit `generated_prompt.py` to customize how Q&A pairs are generated:

```python
def prompt_template(data: str, num_records: int = 5):
    return f"""Your custom prompt here...
    
    Data to process:
    {data}
    """
```

### Quality Filtering

The quality assessment uses a local LLM to score each Q&A pair on:
- **Accuracy**: How well the answer addresses the question
- **Style**: Clarity, helpfulness, and appropriateness

Only pairs scoring ≥6 on both metrics are included in the final dataset.

### Batch Processing

Large PDF collections are processed in batches:
- Configurable batch size (`pdfs_per_batch`)
- Automatic resumption if interrupted
- Progress tracking and cost estimation
- Parallel processing for maximum speed

## 💰 Cost Estimation

Using `gpt-4o-mini` (recommended):
- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens
- Typical cost: **$0.10-0.50 per PDF** (depending on size)

## 🎛️ Model Options

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| `gpt-4o-mini` | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | **Recommended** - Best balance |
| `gpt-4o` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Maximum quality needed |
| `gpt-3.5-turbo` | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | Budget-conscious |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨🏾‍💻 Authors

**Original Work**: [Nick Renotte](https://github.com/nickrenotte)
- GitHub: [@nickrenotte](https://github.com/nickrenotte)
- YouTube: [Nicholas Renotte](https://www.youtube.com/c/NicholasRenotte)
- Original Project: [LoRA Fine-tuning Tutorial](https://github.com/nickrenotte/lora-finetuning)

**Adaptations & Enhancements**: Kellan Finney
- Organization: 8th Revolution
- GitHub: [@KellanFinney](https://github.com/KellanFinney)
- Focus: Data generation pipeline improvements and standalone packaging

## 🙏 Acknowledgments

- **[Nick Renotte](https://github.com/nickrenotte)** for the original LoRA fine-tuning pipeline and data generation concepts
- [Docling](https://github.com/DS4SD/docling) for PDF processing
- [OpenAI](https://openai.com/) for language model APIs
- [HuggingFace](https://huggingface.co/) for dataset infrastructure
- [Ollama](https://ollama.ai/) for local LLM quality assessment

## 📚 Related Projects

- **[LoRA Fine-tuning Tutorial](https://github.com/nickrenotte/lora-finetuning)** - Original complete LoRA training pipeline by Nick Renotte
- [Transformers LoRA](https://github.com/huggingface/peft) - HuggingFace PEFT library

---

<div align="center">

**⭐ Star this repo if it helped you create better datasets! ⭐**

[Report Bug](https://github.com/nickrenotte/lora-datagen/issues) | [Request Feature](https://github.com/nickrenotte/lora-datagen/issues) | [Documentation](https://github.com/nickrenotte/lora-datagen#readme)

</div> 