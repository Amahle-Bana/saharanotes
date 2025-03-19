import os
import gc
import glob
import math
import random
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
import heapq
from typing import Dict, List, Tuple
from model1 import DecoderOnlyModel

# Defining epochs, learning_rate, batch_size
epochs = 2000
learning_rate = 0.00001
batch_size = 16  # Reduced batch size for memory efficiency

# Defining model parameters
model_name = "GROUP3-GPT-TRANSFORMER-DEMO"
num_layers = 3
num_heads = 8
embed_dim = 512
dropout_rate = 0.1
context_length = 128
with_encoder = False

# Setting device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initializing the model
model_instance = DecoderOnlyModel(
    num_layers=num_layers,
    num_heads=num_heads,
    embed_dim=embed_dim,
    dropout_rate=dropout_rate,
    with_encoder=with_encoder,
    device=device,
    model_name=model_name
)
model_instance = model_instance.to(device)




# Defining the LanguageModellingDataset class
class LanguageModellingDataset(Dataset):
    def __init__(self, text_lines, tokenizer, context_length=128, stride=50):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.samples = []
        
        # Combine shorter lines to reach context length
        all_tokens = []
        
        # If text_lines is a file object, read it line by line
        if hasattr(text_lines, 'readlines'):
            text_lines = text_lines.readlines()
        # If text_lines is already a string, split it by newlines
        elif isinstance(text_lines, str):
            text_lines = text_lines.split('\n')
        
        # First, tokenize all lines and collect tokens
        print("Tokenizing text...")
        for line in tqdm(text_lines, desc="Tokenizing lines"):
            if line.strip():  # Skip empty lines
                tokens = tokenizer.encode(line, add_special_tokens=False)
                if tokens:  # Only add if there are tokens
                    all_tokens.extend(tokens)
                    # Add a newline token if available, or space token
                    if tokenizer.sep_token_id:
                        all_tokens.append(tokenizer.sep_token_id)
                    elif tokenizer.eos_token_id:
                        all_tokens.append(tokenizer.eos_token_id)
        
        # Now create chunks from the combined tokens
        print(f"Creating chunks with context length {context_length}...")
        for i in range(0, len(all_tokens) - context_length, stride):
            chunk = all_tokens[i:i + context_length]
            if len(chunk) == context_length:  # Ensure chunk is exactly context_length
                self.samples.append(chunk)
        
        print(f"Created {len(self.samples)} samples")
        
        # If we still don't have samples, reduce context length as a fallback
        if not self.samples and all_tokens:
            fallback_context_length = min(len(all_tokens), 512)  # Use smaller context length
            print(f"No samples with context length {context_length}, falling back to {fallback_context_length}")
            
            for i in range(0, len(all_tokens) - fallback_context_length, fallback_context_length // 2):
                chunk = all_tokens[i:i + fallback_context_length]
                if len(chunk) == fallback_context_length:
                    self.samples.append(chunk)
            
            self.context_length = fallback_context_length
            print(f"Created {len(self.samples)} samples with fallback context length")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = self.tokenizer.eos_token_id  # Last token predicts EOS

        # Ensure labels are within the valid vocabulary range
        labels[labels >= self.tokenizer.vocab_size] = self.tokenizer.pad_token_id

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Loading the training data
print("Loading training data...")
with open("Transformer_Training_Dataset.txt", "r", encoding="utf-8") as file:
    pretraining_text_src = file.readlines()

# Calculating split index
split_index = int(0.8 * len(pretraining_text_src))

# Splitting data
train_data = pretraining_text_src[:split_index]
test_data = pretraining_text_src[split_index:]

# Save the split data
with open('training_data.txt', 'w', encoding="utf-8") as train_file:
    train_file.writelines(train_data)

with open('testing_data.txt', 'w', encoding="utf-8") as test_file:
    test_file.writelines(test_data)

# Load the train data
with open("training_data.txt", "r", encoding="utf-8") as file:
    train_text_src = file.readlines()

# Join the list of strings into a single string
train_text_src = ''.join(train_text_src)


# Take a subset of the data if it's too large
max_lines = 50000  # Adjust based on your memory constraints
if len(train_text_src) > max_lines:
    print(f"Dataset too large, using {max_lines} lines out of {len(train_text_src)}")
    text_lines = train_text_src[:max_lines]

# Creating the dataset
print(f"Creating dataset...")
dataset = LanguageModellingDataset(
    text_lines=text_lines, 
    tokenizer=model_instance.tokenizer,
    context_length=model_instance.context_length,
    stride=model_instance.context_length // 2,
)

# Ensure we have samples
if len(dataset) == 0:
    raise ValueError("Dataset is empty! Check your text file and context length settings.")

# Printing the length of the dataset
print(f"Dataset contains {len(dataset)} samples")

# Creating the train loader with num_workers=0 to avoid multiprocessing issues
train_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)