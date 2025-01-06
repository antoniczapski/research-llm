#!/usr/bin/env python3
import os
import glob
import random
import numpy as np
import spacy
from transformers import AutoTokenizer
from tqdm import tqdm

# spacy.require_gpu()
# check if cuda is available
# print("GPU available: ", spacy.prefer_gpu())

###############################################################################
# 1. Setup spaCy for Polish
###############################################################################
# Make sure you've installed:
#   pip install spacy==3.5.3
#   python -m spacy download pl_core_news_sm
nlp = spacy.load("pl_core_news_sm")


###############################################################################
# 2. Rare characters mapping
#    We'll parse them from the multi-line string. Each line has `char code`
###############################################################################
RARE_CHARS_RAW = """
豆 948
車 949
通 950
道 951
郎 952
郡 953
部 954
都 955
野 956
金 957
鉄 958
長 959
門 960
関 961
青 962
音 963
風 964
馬 965
高 966
가 967
고 968
국 969
기 970
김 971
나 972
는 973
다 974
대 975
도 976
독 977
라 978
로 979
리 980
마 981
부 982
사 983
서 984
스 985
시 986
아 987
에 988
을 989
의 990
이 991
일 992
전 993
제 994
조 995
주 996
지 997
하 998
한 999
""".strip().splitlines()

RARE_CHARS = []
for line in RARE_CHARS_RAW:
    ch, _ = line.strip().split()
    RARE_CHARS.append(ch)

# As spaCy sees new POS, we pop from RARE_CHARS.
pos2char = {}
char2pos = {}

###############################################################################
# 3. Function to inject POS as a (word + rare_char).
###############################################################################
def inject_pos_as_rare_char(text: str) -> str:
    """
    Example:
        Input:  "Ala ma kota."
        Output: "Ala 豆 ma 車 kota 通 . 郎"
        (assuming 'VERB' => '豆', 'NOUN' => '車', 'PUNCT' => '郎', etc.)
        Note the extra space for clarity, but you could also do "Ala豆".
    """
    doc = nlp(text)
    output_tokens = []
    for token in doc:
        if not token.text.strip():
            continue
        
        pos_label = token.pos_
        # If we haven't mapped this POS yet, pop a new rare char from RARE_CHARS.
        if pos_label not in pos2char:
            if not RARE_CHARS:
                raise ValueError("Ran out of rare characters for new POS tags!")
            rare_char = RARE_CHARS.pop(0)
            pos2char[pos_label] = rare_char
            char2pos[rare_char] = pos_label
        
        mapped_char = pos2char[pos_label]
        
        # e.g. "kota" + " " + "車" => "kota 車"
        # Or if you prefer no space: "kota車". Just be consistent.
        new_token = f"{token.text} {mapped_char}"
        output_tokens.append(new_token)
        
    return " ".join(output_tokens)

###############################################################################
# 4. Initialize the Polish model tokenizer (HerBERT)
###############################################################################
#   pip install transformers==4.31
# Define the model name
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

###############################################################################
# 5. Define the main streaming function
###############################################################################
def process_corpus_line_by_line():
    """
    Reads text files from `data/raw/pl` one line at a time, 
    injects POS, appends to either train or val.
    Saves both .txt and .bin (tokenized) output for train/val.
    """
    # Hard-coded path and config
    corpus_path = os.path.join("data","raw","pl")  # do not take from STDIN
    out_prefix = os.path.join("data","preprocessed","pl","pl_corpus")
    val_ratio = 0.1
    seed = 2357

    random.seed(seed)  # for reproducible line-level split

    # Collect all .txt paths
    txt_files = glob.glob(os.path.join(corpus_path, "*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {corpus_path}.")

    print(f"Found {len(txt_files)} text file(s) in {corpus_path}:")
    for p in txt_files:
        print("  ", p)

    # Open our four output files
    train_txt_path = f"{out_prefix}_train.txt"
    val_txt_path   = f"{out_prefix}_val.txt"
    train_bin_path = f"{out_prefix}_train.bin"
    val_bin_path   = f"{out_prefix}_val.bin"

    print(f"Opening:\n  {train_txt_path}\n  {train_bin_path}\n  {val_txt_path}\n  {val_bin_path}\n")

    # Open text files in write mode
    train_txt_f = open(train_txt_path, "w", encoding="utf-8")
    val_txt_f   = open(val_txt_path,   "w", encoding="utf-8")

    # Open binary files in append-binary mode
    train_bin_f = open(train_bin_path, "ab")
    val_bin_f   = open(val_bin_path,   "ab")

    try:
        # Stream lines from each file
        total_lines = 0
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                # count line and set up tqdm
                lines = sum(1 for line in f)
                f.seek(0)
                f = tqdm(f, total=lines, desc=f"Processing {file_path}", unit="lines")
                for line in f:
                    line = line.strip()
                    if not line or len(line) > 500:
                        continue

                    total_lines += 1
                    # Decide if goes to train or val
                    if random.random() < val_ratio:
                        is_val = True
                    else:
                        is_val = False

                    # 1) Transform line => inject POS
                    processed_line = inject_pos_as_rare_char(line)

                    # 2) Write processed_line to the correct .txt
                    # if is_val:
                    #     val_txt_f.write(processed_line + "\n")
                    # else:
                    #     train_txt_f.write(processed_line + "\n")

                    # 3) Tokenize
                    enc_ids = tokenizer.encode(processed_line, add_special_tokens=False)
                    arr = np.array(enc_ids, dtype=np.uint16)

                    # 4) Append token IDs to the correct .bin
                    if is_val:
                        arr.tofile(val_bin_f)
                    else:
                        arr.tofile(train_bin_f)
                                            
        print(f"Processed {total_lines} lines in total.")

    finally:
        # Close all files
        train_txt_f.close()
        val_txt_f.close()
        train_bin_f.close()
        val_bin_f.close()

        # Save pos2char mapping
        pos_map_file = "pos2char.txt"
        print(f"Saving POS => Char mapping to {pos_map_file}")
        with open(pos_map_file, "w", encoding="utf-8") as f_map:
            for pos, ch in pos2char.items():
                f_map.write(f"{pos}\t{ch}\n")
        
        char_map_file = "char2pos.txt"
        print(f"Saving Char => POS mapping to {char_map_file}")
        with open(char_map_file, "w", encoding="utf-8") as f_map:
            for ch, pos in char2pos.items():
                f_map.write(f"{ch}\t{pos}\n")

    print("Done!")

###############################################################################
# 6. (Optional) A small demonstration
###############################################################################
def toy_test():
    text = "Ala ma kota i bardzo go kocha."
    processed = inject_pos_as_rare_char(text)
    print("Processed line:", processed)
    enc_ids = tokenizer.encode(processed, add_special_tokens=False)
    print("Encoded IDs:", enc_ids)
    for tid in enc_ids:
        print(f"  ID={tid} => {tokenizer.decode([tid])}")

###############################################################################
# 7. Entry point
###############################################################################
if __name__ == "__main__":
    # If you want, you can just call toy_test() here for debugging
    toy_test()

    # process_corpus_line_by_line()
