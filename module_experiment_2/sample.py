#!/usr/bin/env python3
"""
Script to generate text samples from a trained GPT model using random prompts from training data.
All parameters are initialized directly within the script.
"""

import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# Import your custom GPT model classes
from model import GPTConfig, GPT

# ---------------------------------------------------------------------
# Configuration (Set parameters here)
# ---------------------------------------------------------------------
CHECKPOINT_PATH = './models/pl_v3/ckpt_lalka_prus_normal_100.pt'  # Path to your model checkpoint
DATA_DIR = './module_experiment_2/lalka_prus_train.txt'          # Directory containing training .txt files
OUTPUT_DIR = './reports/generated_samples/prus_normal_100'               # Directory to save generated text samples
NUM_SAMPLES = 5                                          # Number of samples to generate
MAX_LENGTH = 100                                         # Number of tokens to generate per sample
TEMPERATURE = 1.0                                        # Sampling temperature (higher = more random)
TOP_K = 50                                               # Top-K sampling (retain top K tokens)
TOP_P = 0.95                                             # Top-P sampling (retain tokens with cumulative prob >= top_p)
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
TOKENIZER_NAME = "allegro/herbert-base-cased"            # Name of the tokenizer
SEED = 42                                                # Random seed for reproducibility
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Initialize random seed for reproducibility
# ---------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
print(f"Random seed set to {SEED}.")

# ---------------------------------------------------------------------
# Load the pretrained tokenizer
# ---------------------------------------------------------------------
def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer '{tokenizer_name}' loaded.")
    return tokenizer

tokenizer = load_tokenizer(TOKENIZER_NAME)

# ---------------------------------------------------------------------
# Load the GPT model from checkpoint
# ---------------------------------------------------------------------
def load_model(checkpoint_path, tokenizer, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from '{checkpoint_path}'.")
    
    # Extract model configuration
    model_args = checkpoint.get('model_args', {})
    if 'vocab_size' not in model_args:
        model_args['vocab_size'] = tokenizer.vocab_size
    
    # Initialize the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load model state
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            new_key = k[len(unwanted_prefix):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    
    return model

model = load_model(CHECKPOINT_PATH, tokenizer, DEVICE)

# ---------------------------------------------------------------------
# Select a random prompt from training data
# ---------------------------------------------------------------------
def select_random_prompt(random_file):
    with open(random_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError(f"No lines found in '{random_file}'.")
    
    prompt_line = random.choice(lines).strip()
    print(f"Selected prompt:\n{prompt_line}\n")
    return prompt_line

# ---------------------------------------------------------------------
# Encode the prompt into token IDs
# ---------------------------------------------------------------------
def encode_prompt(prompt, tokenizer, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, seq_len)
    print(f"Encoded prompt token IDs: {prompt_ids}\n")
    return prompt_tensor

# ---------------------------------------------------------------------
# Top-k and top-p filtering
# ---------------------------------------------------------------------
def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float('Inf')):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter the mask back to the original ordering
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

# ---------------------------------------------------------------------
# Generate text using the model
# ---------------------------------------------------------------------
def generate_text(model, tokenizer, prompt_tensor, max_length, temperature, top_k, top_p):
    generated = prompt_tensor.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            
            # Unpack logits from the outputs tuple
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Focus on the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            
            # Convert logits to probabilities
            probabilities = torch.softmax(filtered_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            # Append the sampled token to the generated sequence
            generated = torch.cat((generated, next_token), dim=1)
    
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

# ---------------------------------------------------------------------
# Save the generated text
# ---------------------------------------------------------------------
def save_generated_text(output_dir, sample_id, text):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generated_sample_{sample_id}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Generated text saved to '{output_file}'.\n")
    
import re

def remove_chinese_characters(text):
    # Define a pattern to match Chinese characters
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    # Replace Chinese characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text    

# ---------------------------------------------------------------------
# Main script logic
# ---------------------------------------------------------------------
if __name__ == "__main__":
    for sample_id in tqdm(range(1, NUM_SAMPLES + 1), desc="Generating Samples"):
        # Select a random prompt
        prompt = select_random_prompt(DATA_DIR)
        
        # Encode the prompt
        prompt_tensor = encode_prompt(prompt, tokenizer, DEVICE)
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_tensor=prompt_tensor,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )
        
        clean = True
        if clean:
            generated_text = remove_chinese_characters(generated_text)
            
        # Print the generated text
        print(f"### Sample {sample_id} ###")
        print(generated_text)
        print("\n" + "#" * 50 + "\n")
        
        # Save the generated text
        save_generated_text(OUTPUT_DIR, sample_id, generated_text)
        
        
# Generating Samples:   0%|                                                                                                                                                                                                                                                       | 0/5 [00:00<?, ?it/s]Selected prompt:
# I 道 tyle 部 . 車 Spojrzałem 都 na 野 bilet 豆 , 車 czytam 都 : 車 „ 車 Wiljam 都 Colins 通 , 車 nauczyciel 豆 języka 豆 angielskiego 郎 “ 車 . 車 . 車 . 車 Cóż 金 to 関 za 野 farsa 豆 ? 車 . 車 . 車 . 車 Przecie 部 chyba 鉄 Wokulski 通 nie 鉄 będzie 関 uczył 都 się 金 poangielsku 豆 ? 車 . 車 . 車 . 車

# Encoded prompt token IDs: [1056, 1863, 4108, 1871, 1899, 1671, 34186, 24504, 1737, 1998, 1790, 16287, 1738, 1947, 1671, 26623, 1737, 1335, 1671, 1791, 1671, 4924, 34467, 1737, 4277, 5422, 1010, 1604, 1947, 1671, 11396, 1738, 7124, 1738, 12888, 1383, 1781, 1671, 1899, 1671, 1899, 1671, 1899, 1671, 14343, 1357, 2063, 1803, 2163, 1790, 6827, 2543, 1738, 1550, 1671, 1899, 1671, 1899, 1671, 1899, 1671, 2398, 2104, 1871, 4242, 1712, 2599, 25549, 1604, 1997, 1712, 2282, 1803, 22267, 1737, 2022, 1357, 1991, 26239, 1738, 1550, 1671, 1899, 1671, 1899, 1671, 1899, 1671]

# ### Sample 1 ###
# I  tyle .  Spojrzałem  na  bilet ,  czytam  :  „  Wiljam  Colins ,  nauczyciel  języka  angielskiego  “ . . .  Cóż  to  za  farsa ? . . .  Przecie  chyba  Wokulski  nie  będzie  uczył  się  poangielsku ? . . . . . . . . . . .  Ale  że  się  się  nawet  będzie  pewny .  —  —  Już  ten  pani . . .  —  rzekł  Wokulski . . . . .  —  Nie  nie  nie ,  który  razie .  —  —  Wokulski  —  —  myślał .  —  a 

# ##################################################

# Generated text saved to './reports/generated_samples/generated_sample_1.txt'.

# Generating Samples:  20%|███████████████████████████████████████████████▊                                                                                                                                                                                               | 1/5 [00:00<00:03,  1.32it/s]Selected prompt:
# — 車 Panna 豆 Łęcka 通 ma 都 także 鉄 otwarty 郎 kredyt 豆 . 車 . 車 . 車 bardzo 部 dobrze 部 — 車 ciągnął 都 Wokulski 通 , 車 zbliżywszy 都 twarz 豆 do 野 księgi 豆 , 車 jakby 門 w 野 niej 金 pismo 豆 było 関 niewyraźne 郎 . 車 — 車 A 郡 . 車 . 車 . 車 a 郡 . 車 . 車 . 車 Onegdaj 豆 wzięła 都 portmonetkę 豆 . 車 . 車 . 車 Trzy 青 ruble 豆 ? 車 . 車 . 車 . 車 to 関 chyba 鉄 zadrogo 都 . 車 . 車 . 車

# Encoded prompt token IDs: [1679, 1671, 49628, 1738, 8872, 3056, 1604, 2185, 1737, 2413, 1712, 13902, 1383, 11397, 1738, 1899, 1671, 1899, 1671, 1899, 1671, 2450, 1871, 3394, 1871, 1679, 1671, 28167, 1737, 2599, 25549, 1604, 1947, 1671, 16111, 27140, 1737, 10433, 1738, 2041, 1790, 20789, 1738, 1947, 1671, 5596, 1425, 1019, 1790, 2233, 1357, 9923, 1738, 2404, 1803, 2013, 25904, 1383, 1899, 1671, 1679, 1671, 1012, 1708, 1899, 1671, 1899, 1671, 1899, 1671, 1011, 1708, 1899, 1671, 1899, 1671, 1899, 1671, 51, 2029, 9038, 1013, 1738, 16535, 1737, 10806, 16445, 6039, 1738, 1899, 1671, 1899, 1671, 1899, 1671, 8832, 1787, 2061, 9015, 1738, 1550, 1671, 1899, 1671, 1899, 1671, 1899, 1671, 2063, 1803, 4242, 1712, 2003, 33846, 1737, 1899, 1671, 1899, 1671, 1899, 1671]

# ### Sample 2 ###
# —  Panna  Łęcka  ma  także  otwarty  kredyt . . .  bardzo  dobrze  —  ciągnął  Wokulski ,  zbliżywszy  twarz  do  księgi ,  jakby  w  niej  pismo  było  niewyraźne .  —  A . . .  a . . .  Onegdaj  wzięła  portmonetkę . . .  Trzy  ruble ? . . .  to  chyba  zadrogo . . . . .  —  —  Pan  —  To  pan  pani ?  —  — .  —  —  że  mnie  nie  nie  —  —  że  co  ja  panna . . . . . .  —  —  Ach ,  czy  zpoczątku ,  ale  pani  pani  —  Wokulski  — . .  — .  — 

# ##################################################

# Generated text saved to './reports/generated_samples/generated_sample_2.txt'.

# Generating Samples:  40%|███████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                               | 2/5 [00:01<00:01,  1.75it/s]Selected prompt:
# Mnie 金 już 鉄 zabrakło 都 cierpliwości 豆 . 車

# Encoded prompt token IDs: [15899, 1357, 2267, 1712, 8278, 1737, 33343, 1738, 1899, 1671]

# ### Sample 3 ###
# Mnie  już  zabrakło  cierpliwości .  Gdy  zarobiłby  się  do  niej  w  drzwi . . .  —  Pan  ma ,  jeżeli  nawet  pani .  “  — .  —  odparł  Ochocki .  —  Co  pan  Łęcki  nie  —  że  ten  mi ,  to  mi  ja  jest . . .  —  —  pomyślał  do  pana , 

# ##################################################

# Generated text saved to './reports/generated_samples/generated_sample_3.txt'.

# Generating Samples:  60%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                               | 3/5 [00:01<00:00,  2.05it/s]Selected prompt:
# — 車 Nic 金 nagłego 郎 . 車 . 車 . 車 Poco 都 się 金 wielmożny 郎 pan 豆 ma 都 tak 部 śpieszyć 都 ? 車 . 車 . 車 . 車 — 車 odparł 都 Szpigelman 通 . 車

# Encoded prompt token IDs: [1679, 1671, 7579, 1357, 1996, 18407, 1383, 1899, 1671, 1899, 1671, 1899, 1671, 2105, 2249, 1737, 2022, 1357, 2713, 2292, 2067, 1383, 2937, 1738, 2185, 1737, 2342, 1871, 4906, 5198, 1737, 1550, 1671, 1899, 1671, 1899, 1671, 1899, 1671, 1679, 1671, 23848, 1737, 23404, 31051, 4099, 1604, 1899, 1671]

# ### Sample 4 ###
# —  Nic  nagłego . . .  Poco  się  wielmożny  pan  ma  tak  śpieszyć ? . . .  —  odparł  Szpigelman . . . .  —  —  No  mi  —  mówił  się  i .  —  —  Nie  zgóry  pan  —  To  tylko  nie  mi ,  do  Wokulski .  —  —  czy  ząb ?  —  spytał .  — . . . . . . . . .  — . .  — .

# ##################################################

# Generated text saved to './reports/generated_samples/generated_sample_4.txt'.

# Generating Samples:  80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                               | 4/5 [00:01<00:00,  2.26it/s]Selected prompt:
# — 車 Ośmdziesiąt 豆 pięć 青 . 車 . 車 . 車 — 車 wtrąca 都 Szlangbaum 通 . 車

# Encoded prompt token IDs: [1679, 1671, 46446, 81, 6745, 1738, 5622, 1787, 1899, 1671, 1899, 1671, 1899, 1671, 1679, 1671, 91, 2518, 3841, 1737, 2387, 2509, 75, 2289, 2607, 1604, 1899, 1671]

# ### Sample 5 ###
# —  Ośmdziesiąt  pięć . . .  —  wtrąca  Szlangbaum . .  „ .  —  szepnął  z  pani . . . . .  — . . .  —  szepnął  baum ,  ale  i  nawet  o  moję ,  jak  do  tej  lat . . .  —  —  Ale  co  jest  pan  : .  —  Więc  i  nic  się  pani , 

# ##################################################

# Generated text saved to './reports/generated_samples/generated_sample_5.txt'.