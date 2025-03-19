import math
import torch
import torch.nn as nn
from dataloading2 import LanguageModellingDataset
from torch.utils.data import DataLoader
from model1 import DecoderOnlyModel
from dataloading2 import text_lines
import wandb
from tqdm import tqdm

# Load the train data
with open("Transformer_Training_Dataset.txt", "r", encoding="utf-8") as file:
    train_text_src = file.readlines()

# Join the list of strings into a single string
train_text_src = ''.join(train_text_src)


# Take a subset of the data if it's too large
max_lines = 10000  # Adjust based on your memory constraints
if len(train_text_src) > max_lines:
    print(f"Dataset too large, using {max_lines} lines out of {len(train_text_src)}")
    text_lines = train_text_src[:max_lines]

# Initialize the dataset
def initialize_dataset(model_instance, batch_size):
    dataset = LanguageModellingDataset(
        text_lines=text_lines, tokenizer=model_instance.tokenizer,
        context_length=model_instance.context_length,
        stride=(model_instance.context_length // 2))
    if model_instance.tokenizer.eos_token_id is None:
        model_instance.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        model_instance.tokenizer.eos_token_id = model_instance.tokenizer.convert_tokens_to_ids('[EOS]')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the optimizer
def initialize_optimizer(model_instance, optimizer, learning_rate):
    if optimizer == "adam":
        optimizer_instance = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        optimizer_instance = torch.optim.AdamW(model_instance.parameters(), lr=learning_rate)
    return optimizer_instance

# Initialize the model
def initialize_model(config):
    model_instance = DecoderOnlyModel(
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        embed_dim=config.embed_dim,
        dropout_rate=config.dropout_rate,
        with_encoder=config.with_encoder,
        device=config.device
    )
    return model_instance.to(model_instance.device)

# Train the model for each epoch
def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False, total=len(dataloader)):
        inputs, attn_mask, labels = batch['input_ids'].to(model.device), batch['attention_mask'].to(model.device), batch['labels'].to(model.device)
        optimizer.zero_grad() # Zero the parameter gradients

        outputs = model(inputs) # Forward pass
        outputs = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_len, vocab_size]
        labels = labels.view(-1)  # Reshape to [batch_size * seq_len]

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    loss = running_loss / len(dataloader)
    perplexity = math.exp(loss)
    return loss, perplexity

# Train the model for the number of epochs specified in the config
def train_fn (config=None):
    with wandb.init(config=config):
        config = wandb.config

        model = initialize_model(config)
        optimizer = initialize_optimizer(model, config.optimizer, config.learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
        dataloader = initialize_dataset(model, config.batch_size)

        for epoch in range(config.epochs):
            avg_loss, perplexity = train_epoch(model, dataloader, optimizer, loss_fn)
            wandb.log({"loss": avg_loss, "perplexity": perplexity})
            print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    print(f"\n==================================Finished Training!== in {config.epochs} epochs==================================\n")
    return model.eval()


# Sweep Configuration
sweep_config = {
    'method': 'random',

    'metric': {
        'name': 'loss', # The Metric that we want to optimize for.
        'goal': 'minimize'   # What we want to achieve about this Metric.
    },

    'parameters': {
        'epochs': {
            'value': 2000
        },
        'device': {
            'value': ('cuda' if torch.cuda.is_available() else 'cpu')
        },
        'context_length': {
            'value': 128
        },
        'with_encoder': {
            'value': False
        },
        'num_layers': {
            'values': [2, 3, 4]
        },
        'num_heads': {
            'values': [4, 8, 16]
        },
        'embed_dim': {
            'values': [256, 512, 1024]
        },
        'dropout_rate': {
            'values': [0.1, 0.15, 0.2]
        },
        'learning_rate': {
            'values': [0.001, 0.0001, 0.00001]
        },
        'optimizer': {
            'values': ['adam', 'adamw']
        },
        'batch_size': {
            'values': [8, 16, 32]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Sahara-AI-Hyperparameter-Tuning")

wandb.agent(sweep_id, train_fn, count=20)