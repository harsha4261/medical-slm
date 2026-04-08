import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from tqdm import tqdm
import os

# -----------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (Must match training exactly)
# -----------------------------------------------------------------------------
# ... (We re-define the classes to ensure this script is standalone) ...

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        return logits, None

# -----------------------------------------------------------------------------
# 2. EVALUATION LOGIC
# -----------------------------------------------------------------------------

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}...")

    # 1. Load Model
    config = GPTConfig(
        vocab_size=50257, block_size=128, 
        n_layer=12, n_head=12, n_embd=768, 
        dropout=0.0, bias=True
    )
    model = GPT(config)
    
    checkpoint_path = "best_model_params.pt"
    if not os.path.exists(checkpoint_path):
        print("Error: best_model_params.pt not found.")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Clean up DDP prefixes if necessary
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Load Validation Data
    if not os.path.exists("validation.bin"):
        print("Error: validation.bin not found. Did you run training?")
        return
        
    data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    print(f"Loaded validation data with {len(data)} tokens.")

    # 3. Evaluation Loop
    batch_size = 32
    block_size = 128
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    # We will sample 100 batches to get a statistical estimate
    num_samples = 100
    
    print("Calculating metrics...")
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            # Random batch sampling
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]).to(device)

            logits, loss = model(x, y)
            
            # --- CALCULATION ---
            total_loss += loss.item()
            
            # Accuracy: Compare predicted token vs actual token
            # logits shape: (B, T, Vocab)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1) # Shape (B, T)
            
            # Check where predictions match targets
            correct = (predictions == y).float()
            accuracy = correct.sum() / correct.numel()
            total_accuracy += accuracy.item()
            
            num_batches += 1

    # 4. Final Metrics
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    avg_accuracy = (total_accuracy / num_batches) * 100

    print("\n" + "="*30)
    print(f"📊 MEDICAL SLM EVALUATION REPORT")
    print("="*30)
    print(f"Validation Loss:      {avg_loss:.4f}")
    print(f"Perplexity (PPL):     {perplexity:.2f}")
    print(f"Next-Token Accuracy:  {avg_accuracy:.2f}%")
    print("="*30)
    
    if perplexity < 30:
        print("✅ Rating: Excellent for a small model.")
    elif perplexity < 50:
        print("✅ Rating: Good. Model learned structure well.")
    else:
        print("⚠️ Rating: High perplexity. Model might be underfitting.")

if __name__ == "__main__":
    evaluate()
