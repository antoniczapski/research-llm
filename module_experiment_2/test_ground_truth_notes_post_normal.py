"""
Sample from a trained model (baseline or POS-tagged)
"""
import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from transformers import AutoTokenizer
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore")

# Import your custom GPT model code
from model import GPTConfig, GPT

###############################################################################
# 1. Define the set of 'rare' characters used for POS tags
###############################################################################
POS_CHARS = {
    '豆', '車', '通', '道', '郎', '郡', '部', '都', '野', '金', '鉄', '長', '門',
    '関', '青', '音', '風', '馬', '高', '가', '고', '국', '기', '김', '나', '는',
    '다', '대', '도', '독', '라', '로', '리', '마', '부', '사', '서', '스', '시',
    '아', '에', '을', '의', '이', '일', '전', '제', '조', '주', '지', '하', '한'
}

###############################################################################
# 2. Set up directories and CSV for results
###############################################################################
models_dir = os.path.join(os.getcwd(), 'models', 'pl')
names = [name for name in os.listdir(models_dir) if 'normal' in name and name.endswith('.pt')]
dataset = 'lalka_prus_normal'
data_dir = os.path.join(os.getcwd(), 'module_experiment_2')
out_dir = os.path.join(os.getcwd(), 'reports', 'experiment_2')
os.makedirs(out_dir, exist_ok=True)

results_csv = os.path.join(out_dir, "results_ground_post_normal.csv")
if not os.path.exists(results_csv):
    with open(results_csv, "w", encoding="utf-8") as file:
        file.write("training iterations,perplexity,perplexity POS\n")

###############################################################################
# 3. Load pretrained tokenizer (HerBERT)
###############################################################################
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
# Make sure the tokenizer vocab size matches what your model expects, or
# that your GPT model was trained in a compatible way.

###############################################################################
# 4. Evaluation loop
###############################################################################
for name in tqdm(names, desc="Evaluating models"):
    # Attempt to parse iteration from filename, e.g. "ckpt_1000.pt" => 1000
    iter_str = name.split("_")[-1].split(".")[0]
    iter_num = int(iter_str)
    
    # GPU / dtype settings
    seed = 1337
    device = 'cuda:1'  # or 'cpu', 'cuda:0', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    # Load the checkpoint
    ckpt_path = os.path.join(models_dir, name)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Recreate the model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Fix any unwanted prefix in state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Load val.bin that matches your training procedure
    data_path = os.path.join(data_dir, f"{dataset}_val.bin")
    data = np.memmap(data_path, dtype=np.uint16, mode='r').astype(np.int64)

    log_prob_sum = 0.0
    log_prob_sum_POS = 0.0
    total_tokens = 0
    total_tokens_POS = 0

    # We'll sample 2000 positions for speed; feel free to iterate more
    max_index = len(data) - 256  # to avoid index error at i+255
    eval_steps = min(2000, max_index)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=ptdtype):
            for i in tqdm(range(eval_steps), desc=f"Scoring {name}", leave=False):
                # context tokens
                context = data[i : i + 255]
                true_next = data[i + 255]  # next token to predict

                # shape = (1, 255)
                x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                
                # Get next-token probabilities from model
                # Make sure your GPT model provides a method .get_probs(x)
                # that returns logit probabilities for the last position
                probs = model.get_probs(x)  # shape = (255, vocab_size) or (1, 255, vocab_size)
                
                # If shape is (1, 255, vocab_size), the final logit is at [0, -1]
                # If shape is (255, vocab_size), the final logit is at [-1]
                # Adjust accordingly if your model differs
                if probs.dim() == 3:
                    # shape: (1, seq_len, vocab_size)
                    next_token_probs = probs[0, -1]  # shape: (vocab_size,)
                else:
                    # shape: (seq_len, vocab_size)
                    next_token_probs = probs[-1]     # shape: (vocab_size,)

                # Probability of the true next token
                true_next_prob = next_token_probs[true_next]  # a single float (logits => prob)

                # Convert logits => probability if your .get_probs returns raw logits
                # If get_probs already returns probabilities, skip this softmax.
                # Check your model definition to confirm. 
                # Below is a typical approach if it returned logits:
                # next_token_probs = F.softmax(next_token_probs, dim=-1)
                # true_next_prob = next_token_probs[true_next]

                # Decode the next token to check if it's a POS char
                # We'll do a single-token decode
                token_str = tokenizer.decode([true_next]).strip()

                # Accumulate log2(prob). If prob = 0 => log2(prob) is -inf, so watch for that
                # Usually 'true_next_prob' shouldn't be exactly zero unless the model is certain it's not the token
                # but let's clamp for safety.
                prob_val = torch.clamp(true_next_prob, 1e-12, 1.0)
                log_val = torch.log2(prob_val)

                if token_str in POS_CHARS:
                    log_prob_sum_POS += log_val
                    total_tokens_POS += 1
                else:
                    log_prob_sum += log_val
                    total_tokens += 1

    # Avoid division by zero if no POS tokens were encountered
    if total_tokens == 0:
        perplexity = float('inf')
    else:
        average_neg_log_prob = -log_prob_sum / total_tokens
        perplexity = 2 ** average_neg_log_prob

    if total_tokens_POS == 0:
        perplexity_POS = float('inf')
    else:
        average_neg_log_prob_POS = -log_prob_sum_POS / total_tokens_POS
        perplexity_POS = 2 ** average_neg_log_prob_POS

    print(f"\nModel: {name}")
    print(f"Perplexity: {perplexity.item() if perplexity != float('inf') else 'inf'}")
    print(f"Perplexity POS: {perplexity_POS.item() if perplexity_POS != float('inf') else 'inf'}\n")

    # Append results to CSV
    with open(results_csv, "a", encoding="utf-8") as file:
        file.write(f"{iter_num},{perplexity},{perplexity_POS}\n")
