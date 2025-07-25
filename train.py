from datasets import load_dataset
from colorama import Fore
from data_loader import load_science_training_data, create_train_test_split, preview_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the custom science training data
DATA_DIR = "./data"  # Directory containing your 82 JSON files
logger.info("Loading science training data...")

# Load and preview the dataset
dataset = load_science_training_data(DATA_DIR)
preview_dataset(dataset, num_samples=2)

# Create train/test split
train_dataset, eval_dataset = create_train_test_split(dataset, test_size=0.05)  # 5% for eval

logger.info(f"Training samples: {len(train_dataset)}")
logger.info(f"Evaluation samples: {len(eval_dataset)}")

def format_chat_template(batch, tokenizer):
    system_prompt = """You are a helpful, honest and harmless assistant designed to help with engineers with their scientific research and analysis. Think through each question logically and provide accurate, well-reasoned answers based on scientific principles. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template and append the result to the list
        text = tokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }

# Use IBM Granite 3.3 model (latest version)
base_model = "./models/granite-3.3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),  # Use environment variable for security
)

# Granite models should have proper padding token setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Process datasets with chat template
logger.info("Applying chat template to training data...")
train_dataset = train_dataset.map(
    lambda x: format_chat_template(x, tokenizer), 
    num_proc=8,  # Utilize more CPU cores on A100 SXM
    batched=True, 
    batch_size=100,  # Larger batch for efficiency
    remove_columns=train_dataset.column_names  # Remove original columns
)

logger.info("Applying chat template to evaluation data...")
eval_dataset = eval_dataset.map(
    lambda x: format_chat_template(x, tokenizer), 
    num_proc=8,  # Utilize more CPU cores on A100 SXM 
    batched=True, 
    batch_size=100,
    remove_columns=eval_dataset.column_names
)

logger.info("Sample processed training example:")
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]['text'][:500]) + "..." + Fore.RESET)

# Quantization config optimized for large dataset training
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",  # Changed to auto for better GPU utilization
    quantization_config=quant_config,
    token=os.getenv("HF_TOKEN"),  # Use environment variable for security
    torch_dtype=torch.bfloat16,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config optimized for large dataset
peft_config = LoraConfig(
    r=128,  # Reduced from 256 for better memory efficiency
    lora_alpha=256,  # Reduced proportionally
    lora_dropout=0.1,  # Slightly increased for regularization
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Training configuration optimized for large dataset (20M tokens)
training_args = SFTConfig(
    output_dir="./granite-science-sft",
    
    # Training schedule
    num_train_epochs=3,  # Reduced epochs for large dataset
    max_steps=-1,  # Let epochs control training
    
    # Batch configuration (optimized for A100 SXM)
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  
    
    # Learning rate and optimization
    learning_rate=1e-4,  # Conservative for large dataset
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    
    # Evaluation and logging
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=3,  # Keep only 3 checkpoints
    
    # Memory and performance
    dataloader_drop_last=True,
    dataloader_num_workers=2,  
    bf16=True,
    gradient_checkpointing=True,
    
    # Monitoring
    report_to="tensorboard",
    run_name="granite-science-training",
    
    # Early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

logger.info("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=2048,  # Reasonable max length
    packing=False,  # Don't pack for better quality
)

logger.info("Starting training...")
trainer.train()

logger.info("Saving model...")
trainer.save_model('./granite-science-final')
trainer.model.save_pretrained("./granite-science-lora-adapter")

logger.info("Training completed successfully!")