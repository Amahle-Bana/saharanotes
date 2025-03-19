import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

# Feed Forward Multi Layer Perceptron Class
class FeedForwardMLP(nn.Module):
    def __init__(self, embed_dim, dropout_rate, device, dtype):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4, device=device, dtype=dtype)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, context_length, dtype, device):
        super().__init__()
        pe = torch.zeros(context_length, embed_dim)
        position = torch.arange(0, context_length, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=dtype) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Full Embedding Class
class FullEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, dtype, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.positional_encoding = PositionalEncoding(embed_dim, context_length, dtype=dtype, device=device)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding(x)
        return x

# Decoder Layer Class
class DecoderLayer(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout_rate, dtype, device):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True, dtype=dtype, device=device)
        self.mlp = FeedForwardMLP(embed_dim=embed_dim, dropout_rate=dropout_rate, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(attn_output + x)
        output = self.mlp(x)
        output = self.norm2(output + x)
        return output

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, embed_dim, vocab_size, context_length, dropout_rate, device, dtype):
        super().__init__()
        self.embedding = FullEmbedding(vocab_size, embed_dim, context_length, dtype=dtype, device=device)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(num_heads, embed_dim, dropout_rate, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        for dec_layer in self.decoder_layers:
            x = dec_layer(x)
        return x

# Decoder Only Model Class
class DecoderOnlyModel(nn.Module):
    def __init__(self, num_layers, num_heads, embed_dim, context_length=128, dropout_rate=0.1, device='cpu', dtype=torch.float32, model_name="DecoderOnlyModel", with_encoder=False):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Explicitly set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.vocab_size = self.tokenizer.vocab_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.dtype = dtype

        self.decoder = Decoder(self.num_layers, self.num_heads, self.embed_dim, self.vocab_size, self.context_length, self.dropout_rate, self.device, self.dtype)
        self.final_linear = nn.Linear(self.embed_dim, self.vocab_size, dtype=self.dtype)

    def forward(self, x):
        x = self.decoder(x)
        logits = self.final_linear(x)
        return logits if self.training else F.softmax(logits, dim=-1)

    # @torch.no_grad()
    # def generate(self, inputs=None, max_output=None):
    #     self.eval()
    #     max_output = min(max_output or self.context_length, self.context_length - 1)

    #     if inputs is None:
    #         return None, None

    #     x = torch.tensor([self.tokenizer.encode(inputs, truncation=True, max_length=self.context_length)], device=self.device)
    #     print(x.shape)
    #     output = x[:, :self.context_length - 1]  # Ensure initial size is within limits
    #     for _ in range(min(max_output, self.context_length - output.shape[1])):
    #         probs = self(output)
    #         next_word_probs = probs[:, -1, :]
    #         next_word = torch.argmax(next_word_probs, dim=-1, keepdim=True)
    #         if next_word.item() == self.tokenizer.eos_token_id:
    #             break
    #         output = torch.cat([output, next_word], dim=1)
    #     text = self.tokenizer.decode(output[0])
    #     return text, output
    
    @torch.no_grad()
    def generate(self, inputs=None, max_output=None):
        self.eval()
        max_output = min(max_output or self.context_length, self.context_length - 1)

        if inputs is None:
            return None, None

        x = torch.tensor([self.tokenizer.encode(inputs, truncation=True, max_length=self.context_length)], device=self.device)
        output = x[:, :self.context_length - 1]  # Ensure initial size is within limits
        for _ in range(min(max_output, self.context_length - output.shape[1])):
            probs = self(output)
            next_word_probs = probs[:, -1, :]
            next_word = torch.argmax(next_word_probs, dim=-1, keepdim=True)
            if next_word.item() == self.tokenizer.eos_token_id:
                break
            output = torch.cat([output, next_word], dim=1)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)  # Use skip_special_tokens=True
        return text, output
