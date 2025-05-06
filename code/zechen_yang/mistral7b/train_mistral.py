import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from sklearn.model_selection import train_test_split
import bitsandbytes as bnb
from typing import Dict, List
from huggingface_hub import login
import sys
from transformers.utils import logging
import torch.nn as nn
import gc

# Check for HF token
if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    print("Please set your HuggingFace token as an environment variable:")
    print("export HUGGING_FACE_HUB_TOKEN='your_token_here'")
    print("\nOr you can pass it directly (less secure):")
    print("python train_mistral.py 'your_token_here'")
    sys.exit(1)

# Login to HuggingFace
if len(sys.argv) > 1:
    # Token provided as command line argument
    login(sys.argv[1])
else:
    # Token from environment variable
    login(os.environ["HUGGING_FACE_HUB_TOKEN"])

# Set logging to error only
logging.set_verbosity_error()

# Set critical memory configurations before anything else
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'  # Increased from 32 to 64MB for RTX A5000

# Memory management helper function
def clear_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

# Constants
SEED = 42
MAX_LENGTH = 256
MICRO_BATCH_SIZE = 2  # Increased from 1 to 2
GRADIENT_ACCUMULATION_STEPS = 64  # Reduced from 128 for faster iterations with more memory
EPOCHS = 3
LEARNING_RATE = 2e-4
OUTPUT_DIR = "/mnt/mistral_data/mistral_mental_health"

# Updated subreddit mapping (including mentalhealth)
SUBREDDIT_TO_LABEL = {
    "depression": 0,
    "Anxiety": 1,
    "bipolar": 2,
    "mentalhealth": 3,
    "BPD": 4,
    "schizophrenia": 5,
    "autism": 6
}

LABEL_TO_SUBREDDIT = {v: k for k, v in SUBREDDIT_TO_LABEL.items()}

def prepare_data():
    """Load and prepare the dataset."""
    # Read the CSV file
    df = pd.read_csv('cleaned_reddit_posts.csv')
    
    # Basic filtering
    df = df[df['cleaned_text'].notna()]
    df = df[df['cleaned_text'].str.len() > 5]
    df = df[df['cleaned_text'].str.len() <= MAX_LENGTH * 4]
    
    # Create labels
    df['label'] = df['Subreddit'].map(SUBREDDIT_TO_LABEL)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("\nFull dataset class distribution:")
    total_samples = 0
    for name, label in SUBREDDIT_TO_LABEL.items():
        count = len(df[df['label'] == label])
        total_samples += count
        print(f"{name}: {count:,} samples")
    print(f"Total samples: {total_samples:,}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, 
        test_size=0.1,  # Smaller validation set
        stratify=df['label'],
        random_state=SEED
    )
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
    
    return train_dataset, val_dataset

def preprocess_function(examples, tokenizer):
    """Tokenize the texts and include labels."""
    tokenized = tokenizer(
        examples["cleaned_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    # Include labels in the tokenized output
    tokenized["labels"] = examples["label"]
    return tokenized

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for name, label in SUBREDDIT_TO_LABEL.items():
        mask = labels == label
        if mask.sum() > 0:
            class_accuracies[name] = (predictions[mask] == labels[mask]).mean()
    
    # Calculate balanced accuracy
    balanced_acc = sum(class_accuracies.values()) / len(class_accuracies)
    
    # Add confusion between mentalhealth and other classes
    mentalhealth_idx = SUBREDDIT_TO_LABEL['mentalhealth']
    mentalhealth_mask = labels == mentalhealth_idx
    if mentalhealth_mask.sum() > 0:
        # Calculate how often mentalhealth is confused with other classes
        for name, label in SUBREDDIT_TO_LABEL.items():
            if name != 'mentalhealth':
                confusion = np.mean(predictions[mentalhealth_mask] == label)
                class_accuracies[f'mentalhealth_confused_as_{name}'] = confusion
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        **{f"accuracy_{k}": v for k, v in class_accuracies.items()}
    }

def train():
    """Main training function."""
    print("Loading tokenizer and model...")
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # Set cache directory
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/mistral_data/huggingface_cache'
    os.environ['HF_HOME'] = '/mnt/mistral_data/huggingface_home'
    os.environ['HF_DATASETS_CACHE'] = '/mnt/mistral_data/datasets_cache'
    
    # Explicitly set cache paths for downloading
    cache_dir = '/mnt/mistral_data/huggingface_cache'
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    clear_memory()
    
    # Enable memory efficient attention with 4-bit quantization for better memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        num_labels=len(SUBREDDIT_TO_LABEL),
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    
    clear_memory()
    
    # Memory optimizations
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Prepare for 4-bit training with memory optimization
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    
    clear_memory()
    
    # More aggressive LoRA config for RTX A5000
    peft_config = LoraConfig(
        r=8,  # Increased from 4 to 8 since we have more memory
        lora_alpha=32,  # Increased back to 32
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Added more projection layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
    )
    
    model = get_peft_model(model, peft_config)
    clear_memory()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE * 2,  # Double eval batch size
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=100,  # Reduced logging frequency
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        dataloader_num_workers=2,  # Increased from 0 to 2 since we have more system RAM
        remove_unused_columns=True,
        optim="adamw_torch",  # Changed from fused to standard version
        warmup_ratio=0.03,  # Add warmup for better training stability
        save_strategy="epoch",  # Save after each epoch
        save_total_limit=1,  # Keep only the latest checkpoint
    )

    print("Training arguments:", vars(training_args))
    
    # Prepare datasets with memory cleanup
    print("Preparing dataset...")
    clear_memory()
    train_dataset, val_dataset = prepare_data()
    
    print("Tokenizing datasets...")
    # More memory-efficient tokenization with more parallelism
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=64,  # Increased batch size for tokenization
        remove_columns=train_dataset.column_names,
        num_proc=4,  # Increased processing threads since we have more system memory
    )
    clear_memory()
    
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=64,  # Increased batch size for tokenization
        remove_columns=val_dataset.column_names,
        num_proc=4,  # Increased processing threads
    )
    clear_memory()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    print(f"Training completed! Model saved to {final_dir}")
    
    return trainer

if __name__ == "__main__":
    train() 