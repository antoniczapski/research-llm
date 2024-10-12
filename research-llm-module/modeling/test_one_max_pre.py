"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import numpy as np
from tqdm import tqdm
from model import GPTConfig, GPT
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

models_dir = os.path.join(os.getcwd() ,'models', 'pre')
names = os.listdir(models_dir)
data_dir = os.path.join(os.getcwd(),'data', 'processed', 'fineweb_pre')
out_dir = os.path.join(os.getcwd(),'reports','pre')

# create results csv file
with open(os.path.join(out_dir,"results_one_max_pre.csv"), "w") as file:
    file.write("training iterations,perplexity,perplexity POS\n")

for name in tqdm(names):
    try:
    # if True:
        iter = int(name.split("_")[1].split(".")[0])
        seed = 1337
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device = 'cuda:1' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]


        # init from a model saved in a specific directory
        ckpt_path = os.path.join(models_dir, name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)

        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ' '.join([itos[i] for i in l])

        # data = np.memmap(os.path.join('data','shakespeare_POS', 'val_SPACY_SPACY_SUFFIX.bin'), dtype=np.uint16, mode='r').astype(np.int64)
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r').astype(np.int64)
        # print(data.max(), data.min())
        # show values for min max (tokens)
        # print(repr(decode([data.max()])), repr(decode([data.min()]))) 
        log_prob_sum = 0
        log_prob_sum_POS = 0
        total_tokens = 0
        total_tokens_POS = 0

        context = data[:255]
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=ptdtype):            
                # iterate over 255 long sliding window on the data
                # for i in tqdm(range(len(data) - 255 - 1)):
                for i in tqdm(range(100000)):
                # for i in range(len(data) - 255 - 1):
                    true_next = data[i+255]

                    # context = data[i:i+255]

                    # context_string = decode(context)
                    # x = (torch.tensor(context, dtype=torch.long, device=device)[None, ...])
                    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                    probs = model.get_probs(x)
                    
                    # true_next_prob = probs[0, -1, true_next] 
                    true_next_prob = probs[-1, true_next]
                    
                    # print the context in natural language and true next token
                    # context_string = decode(context)
                    # print(f"CONTEXT: {context_string} \n\nTRUE NEXT: {itos[true_next]}\n\n")
                    
                    if itos[true_next][0] == "[" and itos[true_next][-1] == "]":
                        log_prob_sum_POS += torch.log2(true_next_prob)
                        total_tokens_POS += 1
                        # choose randomly the next token with zero temperature
                        # next_token = torch.multinomial(probs[-1], 1).item()
                        
                        # choose next token as one-max
                        next_token = torch.argmax(probs[-1]).item()
                        
                        # convert context to a tensor and append the next token to the context
                        context = torch.cat((torch.tensor(context, dtype=torch.long, device=device), torch.tensor([next_token], dtype=torch.long, device=device)), 0)
                    else:
                        log_prob_sum += torch.log2(true_next_prob)
                        total_tokens += 1
                        # convert context to a tensor and append the true next token to the context
                        context = torch.cat((torch.tensor(context, dtype=torch.long, device=device), torch.tensor([true_next], dtype=torch.long, device=device)), 0)

                    # trim the context
                    context = context.cpu().numpy()
                    context = context[-255:]

        average_negative_log_prob = -log_prob_sum / total_tokens
        perplexity = 2 ** average_negative_log_prob

        average_negative_log_prob_POS = -log_prob_sum_POS / total_tokens_POS
        perplexity_POS = 2 ** average_negative_log_prob_POS

        print(f"Perplexity: {perplexity.item()}")
        print(f"Perplexity POS: {perplexity_POS.item()}")
        
        # append results to csv file
        with open(os.path.join(out_dir,"results_one_max_pre.csv"), "a") as file:
            file.write(f"{iter},{perplexity.item()},{perplexity_POS.item()}\n")
    except Exception as e:
        print(f"Error: {e}")
        print(f"File: {name}")
        continue
    # break