import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from contextlib import nullcontext
from dataclasses import dataclass
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
import tiktoken
import matplotlib.pyplot as plt

ddp = int(os.environ.get('RANK', -1)) != -1 # Check if we're in DDP mode
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    torch.manual_seed(42 + ddp_rank) # Set different seed for each process
    device_type = 'cuda'
    print(f"Running DDP on rank {ddp_rank}, local rank {ddp_local_rank}, device {device}")
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    torch.manual_seed(42)
    print(f"Running in non-DDP mode on device {device}")

print("Loading and splitting dataset...")
ds = load_dataset("NoeFlandre/Medical-FineWeb")
ds_split = ds['train'].train_test_split(test_size=0.01, seed=42)
ds = DatasetDict({
    'train': ds_split['train'],
    'validation': ds_split['test']
})

enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    out = {'ids': ids, 'len': len(ids)}
    return out

if ddp_rank == 0:
    if not os.path.exists("train.bin"):
        print(f"Rank {ddp_rank}: Tokenizing and writing data files...")
        tokenized = ds.map(
            process,
            remove_columns=['text', 'record_id', 'url', 'date', 'dump', 'language', 'language_score', 'token_count', 'medical_keyword_count', 'quality_score', 'content_hash', 'processing_timestamp'],
            desc="tokenizing the splits",
            num_proc=8,
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}.bin'
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            # Safety check: don't use more batches than examples
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
        print(f"Rank {ddp_rank}: Data preparation complete.")

if ddp:
    dist.barrier()

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
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
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
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

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
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
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
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training Config
learning_rate = 1e-4
max_iters = 20000
warmup_steps = 1000
min_lr = 5e-4
eval_iters = 500
batch_size = 32 # This is PER-GPU.
block_size = 128

gradient_accumulation_steps = 32

# Dtype and Ctx setup
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.set_default_device(device) # IMPORTANT for DDP

# Batch loading function (relies on globals: block_size, batch_size, device, device_type)
def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,), device='cpu')

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Loss estimation function (MODIFIED for DDP)
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model_to_eval = model.module if ddp else model
    model_to_eval.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device) # Move losses to device
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model_to_eval(X, Y)
            losses[k] = loss.item()

        split_loss = losses.mean()
        if ddp:
            dist.all_reduce(split_loss, op=dist.ReduceOp.AVG)

        out[split] = split_loss.item()

    model_to_eval.train()
    return out

# Model Config (124M Parameter Model)
config = GPTConfig(
    vocab_size=50257,       # gpt2 tokenizer's vocab size
    block_size=block_size,  # This will be 128
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True
)
# Model Initialization
model = GPT(config)
model = model.to(device) # Move model to the correct GPU first

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)

# Schedulers
scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters = warmup_steps) # Smoother start
scheduler_decay = CosineAnnealingLR(optimizer, T_max = max_iters - warmup_steps, eta_min = min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

best_val_loss = float('inf')
best_model_params_path = "best_model_params.pt"
train_loss_list, validation_loss_list = [], []

print("Starting training...")
# Use tqdm only on rank 0
for epoch in tqdm(range(max_iters), disable=(ddp_rank != 0)):

    # --- MODIFIED: Evaluate and log only on rank 0 ---
    if epoch % eval_iters == 0 and epoch != 0:
        losses = estimate_loss(model) # All processes participate

        if ddp_rank == 0: # Only rank 0 prints and saves
            print(f"\nEpoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['val'])

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"New best val loss: {best_val_loss:.4f}. Saving model...")
                # Save the unwrapped model's state
                model_to_save = model.module if ddp else model
                torch.save(model_to_save.state_dict(), best_model_params_path)

    X, y = get_batch("train")

    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()

    # Gradient accumulation
    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()

if ddp_rank == 0:
    print("Training finished. Plotting loss...")
    plt.plot(train_loss_list, 'g', label='train_loss')
    plt.plot(validation_loss_list, 'r', label='validation_loss')
    plt.xlabel(f"Steps - Every {eval_iters} iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('plot.png')
    print("Loss plot saved to plot.png")

    # Load the best model
    print("\nLoading best model for generation...")
    model_gen = GPT(config)  # re-create the model
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_gen.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(gen_device)))
    model_gen = model_gen.to(gen_device)
    model_gen.eval()

    # Generate text
    print("--- Generating Text ---")
    # Use a medically-relevant prompt
    sentence = "Patient presents with a chief complaint of"
    print(f"Starting prompt: {sentence}")

    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0)).to(gen_device)
    y = model_gen.generate(context, 200, temperature=0.8, top_k=20)

    print("--- Generated Output ---")
    print(enc.decode(y.squeeze().tolist()))
    print("------------------------")

if ddp:
    dist.destroy_process_group()

print(f"Process {ddp_rank} finished.")

