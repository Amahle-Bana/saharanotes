import os
import gc
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from transformers import AutoTokenizer, AutoModelForCausalLM
from model1 import DecoderOnlyModel
from dataloading2 import LanguageModellingDataset
from torch.utils.data import DataLoader

# Model Parameters From WANDB Hyperparameter Tuning Sweep 'Generous-sweep-19'
# Hyperparameter Tuning Results:
# Perplexity: 1.02476
# Loss: 0.024462
epochs = 5000
learning_rate = 0.0001
batch_size = 32
model_name = "Sahara-GPT-1"
num_layers = 3
num_heads = 8
embed_dim = 512
dropout_rate = 0.2
optimizer = "AdamW"
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

# Assuming model_instance.tokenizer is your tokenizer
if model_instance.tokenizer.eos_token_id is None:
    # Add the EOS token if it's not already set
    model_instance.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    model_instance.tokenizer.eos_token_id = model_instance.tokenizer.convert_tokens_to_ids('[EOS]')
    print(f"EOS token ID set to: {model_instance.tokenizer.eos_token_id}")

# Verify that the EOS token ID is set
if model_instance.tokenizer.eos_token_id is None:
    raise ValueError("Failed to set EOS token ID. Please check the tokenizer configuration.")

# Load the train data
with open("training_data.txt", "r", encoding="utf-8") as file:
    train_text_src = file.readlines()

# Join the list of strings into a single string
train_text_src = ''.join(train_text_src)


# Take a subset of the data if it's too large
max_lines = 75000000  # Adjust based on your memory constraints
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


# Initialize Weights & Biases
wandb.init(project="Sahara-AI-Trained-Model", name=model_name)

def save_model(model, optimizer, epoch, total_epochs, loss, perplexity, checkpoint_dir="./checkpoints", num_checkpoints=3, save_every=10):
    if epoch % save_every == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model.model_name}_ckpt_{epoch}_loss_{loss:.6f}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'total_epochs': total_epochs,
            'loss': loss,
            'perplexity': perplexity,
        }, checkpoint_path)
        print("Model checkpoint saved.")

def save_artifact(model, tokenizer, model_name=model_name):
    artifact = wandb.Artifact(model_name, type="model")
    save_dir = f"./trained_model/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    artifact.add_dir(save_dir)
    wandb.log_artifact(artifact)

def train_model(model, train_loader, epochs):
    print(f"Using device: {device}")
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    accumulation_steps = 8

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for i, batch in enumerate(progress_bar):
            inputs = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(inputs)  # Use the raw output tensor
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1)) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

        loss = running_loss / len(train_loader)
        perplexity = math.exp(loss)
        wandb.log({"train/loss": loss, "train/perplexity": perplexity})
        save_model(model, optimizer, epoch, epochs, loss, perplexity)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    save_artifact(model, model.tokenizer, model_name=model_name)
    print("Training complete. Model saved to W&B.")
    return model

# Model Evaluation
def evaluate_model(model, test_loader, criterion):
    wandb.init(project="Sahara-AI-Trained-Model", name=model_name)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=-1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    wandb.log({"test_loss": avg_test_loss, "test_accuracy": accuracy})
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return {"test_loss": avg_test_loss, "test_accuracy": accuracy}

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    trained_model = train_model(model_instance, train_loader, epochs)
    
    # Load test data for evaluation
    with open("testing_data.txt", "r", encoding="utf-8") as file:
        test_text_src = file.readlines()
    test_text_src = ''.join(test_text_src)
    max_lines = 10000
    text_lines = test_text_src[:max_lines] if len(test_text_src) > max_lines else test_text_src

    dataset = LanguageModellingDataset(text_src=text_lines, tokenizer=model_instance.tokenizer,
                                       context_length=model_instance.context_length,
                                       stride=model_instance.context_length // 2)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    evaluate_model(model_instance, test_loader, criterion)