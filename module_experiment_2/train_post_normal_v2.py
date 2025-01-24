"""
Training script for a GPT model on the 'lalka_prus_normal' dataset,
with modifications aimed at preventing overfitting.

To run on a single GPU (example):
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs on 1 node (example):
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 8 GPUs across 2 nodes (example):
- First (master) node with IP 123.456.123.456:
  $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=123.456.123.456 --master_port=1234 train.py
- Worker node:
  $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
        --master_addr=123.456.123.456 --master_port=1234 train.py

(If your cluster does not have Infiniband, prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 1. Configuration Values - Adjusted to Prevent Overfitting
# -----------------------------------------------------------------------------
# I/O
out_dir = os.path.join(os.getcwd(), 'models', 'pl_v4')
eval_interval = 20
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume', or 'gpt2*'

# WandB Logging
wandb_log = True
wandb_entity = 'uwr-projects-general'
wandb_project = 'research-llm'
wandb_run_name = 'gpt2-post-normal' + str(time.time())

# Data
dataset = 'lalka_prus_normal'
gradient_accumulation_steps = 5 * 8
batch_size = 6
block_size = 1024

# Model (Dropout increased to 0.1)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1  # was 0.0, now 0.1 to add regularization
bias = False

# Optimizer (LR reduced, weight decay lowered)
learning_rate = 3e-4     # was 6e-4
max_iters = 200
weight_decay = 1e-2      # was 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning Rate Decay
decay_lr = True
warmup_iters = 1000      # was 2000
lr_decay_iters = 300000  # was 600000
min_lr = 3e-5            # was 6e-5

# DDP Settings
backend = 'nccl'
# System
device = 'cuda:1'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# Early Stopping
early_stopping_patience = 5  # Number of evaluations to wait for improvement
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# If you have a configurator, it could override command line or config file options here
# exec(open(os.path.join(os.getcwd(),'module_experiment_2','configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2. DDP Setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else nullcontext()

print(f"Using device: {device}, dtype={dtype}, ddp={ddp}")
print(f"Tokens per iteration: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------
# 3. Data Loader
# -----------------------------------------------------------------------------
data_dir = os.path.join(os.getcwd(), 'module_experiment_2')

def get_batch(split):
    """
    Returns a batch of (x, y) from the memmapped dataset.
    Each call reloads the memmap to avoid memory issues.
    """
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, f'{dataset}_train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f'{dataset}_val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# -----------------------------------------------------------------------------
# 4. Initialize/Load Model
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
meta_vocab_size = tokenizer.vocab_size if tokenizer else 50304

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size,
    dropout=dropout
)

if init_from == 'scratch':
    print("Initializing a new model from scratch...")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}...")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Force certain attributes to match
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint['model_args'][k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    checkpoint = None

else:
    # GPT-2 weights
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Crop block_size if needed
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# -----------------------------------------------------------------------------
# 5. Optimizer & LR Scheduler
# -----------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# wrap model in DDP
if ddp:
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# 6. (Optional) Loading Optimizer from Checkpoint
# -----------------------------------------------------------------------------
if init_from == 'resume' and 'optimizer' in locals():
    optimizer.load_state_dict(checkpoint['optimizer'])

# -----------------------------------------------------------------------------
# 7. Optional Model Compilation
# -----------------------------------------------------------------------------
if compile:
    print("Compiling model... (this may take some time)")
    unoptimized = model
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# 8. Evaluation Helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    """
    Cosine LR scheduler with linear warmup.
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# 9. Weights & Biases Logging
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# 10. Training Loop with Early Stopping
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
no_improvement_steps = 0  # For early stopping

while True:
    # 1) Set current LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2) Evaluate on train/val
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss = losses['val']
        train_loss = losses['train']
        print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": lr,
                "mfu": running_mfu * 100
            })

        # Early Stopping + Checkpointing
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            no_improvement_steps = 0

            if iter_num > 0:
                checkpoint_dict = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                ckpt_name = f'ckpt_{dataset}_{iter_num}.pt'
                ckpt_path = os.path.join(out_dir, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)

            # Save logs
            with open(os.path.join(out_dir, f'log_{dataset}.txt'), 'a') as f:
                f.write(f"{iter_num} {train_loss:.4f} {val_loss:.4f} {lr}\n")

        else:
            no_improvement_steps += 1
            if no_improvement_steps >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    if iter_num == 0 and eval_only:
        # If eval_only, exit right after the first evaluation
        break

    # 3) Gradient Accumulation for micro-steps
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale for accum.

        # prefetch next batch
        X, Y = get_batch('train')

        # backward
        scaler.scale(loss).backward()

    # 4) Gradient Clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 5) Optimizer Step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 6) Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # Approx the total loss (scaled back up)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # 7) Termination Conditions
    if iter_num > max_iters:
        break

# -----------------------------------------------------------------------------
# 11. DDP Cleanup
# -----------------------------------------------------------------------------
if ddp:
    destroy_process_group()

# -----------------------------------------------------------------------------
# 12. Finalize W&B
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.finish()
