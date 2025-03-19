import torch

# Importing the custom model
from model1 import DecoderOnlyModel

# Defining model parameters
epochs = 500
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

# Define the checkpoint path
checkpoint_path = "./checkpoints/Sahara-GPT-1_ckpt_60_loss_2.287507.pt"  # Replace with the actual checkpoint file name

# Load the model
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load the model's state dictionary
model_instance.load_state_dict(checkpoint["model_state_dict"])
model_instance.eval()  # Set model to evaluation mode

# Define a function for text generation
def generate_text(prompt, max_length=128):
    with torch.no_grad():
        output, _ = model_instance.generate(inputs=prompt, max_output=max_length)  # Expecting text output
    return output  # Directly return the generated text since it's already decoded


# # Example usage
# prompt = "Hello, how are you?"
# response = generate_text(prompt)
# print("Generated Response:", response)
