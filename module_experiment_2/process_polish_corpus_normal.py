#!/usr/bin/env python3
import os
import glob
import random
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

###############################################################################
# 1. (Removed spaCy and POS-related parts, we just do raw text now)
###############################################################################

###############################################################################
# 2. Identity transform (no POS, just return the line as-is)
###############################################################################
def pass_through_line(text: str) -> str:
    """
    Returns the line as-is, without any POS tagging or modifications.
    """
    return text

###############################################################################
# 3. Initialize the Polish model tokenizer (HerBERT)
###############################################################################
#   pip install transformers==4.31
# Define the model name
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

###############################################################################
# 4. Main streaming function, but no POS tagging now
###############################################################################
def process_corpus_line_by_line():
    """
    Reads text files one line at a time,
    applies the pass-through function, and appends to train/val sets.
    Saves both .txt and .bin (tokenized) output.
    """

    # Paths, hard-coded for demonstration
    corpus_path = os.path.join("module_experiment_2", "lalka_prus.txt")
    out_prefix = os.path.join("module_experiment_2", "lalka_prus_normal")

    val_ratio = 0.1
    seed = 2357

    random.seed(seed)  # for reproducible line-level split

    # Collect all .txt paths (in this example, just one)
    txt_files = [corpus_path]
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
        total_lines = 0
        # Stream lines from each file
        for file_path in txt_files:
            # Count lines for tqdm
            with open(file_path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)  # Just to get total line count
                f.seek(0)

                # TQDM progress bar
                f = tqdm(f, total=lines, desc=f"Processing {file_path}", unit="lines")
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1

                    # Decide if this line goes to train or val
                    if random.random() < val_ratio:
                        is_val = True
                    else:
                        is_val = False

                    # 1) Transform line => pass-through
                    processed_line = pass_through_line(line)

                    # 2) Write processed_line to the correct .txt
                    if is_val:
                        val_txt_f.write(processed_line + "\n")
                    else:
                        train_txt_f.write(processed_line + "\n")

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

    print("Done! Baseline (no POS tagging) corpus created.\n")

###############################################################################
# 5. (Optional) A small demonstration
###############################################################################
def toy_test():
    text = "Ala ma kota i bardzo go kocha."
    processed = pass_through_line(text)
    print("Processed line:", processed)
    enc_ids = tokenizer.encode(processed, add_special_tokens=False)
    print("Encoded IDs:", enc_ids)
    for tid in enc_ids:
        print(f"  ID={tid} => {tokenizer.decode([tid])}")

###############################################################################
# 6. Entry point
###############################################################################
if __name__ == "__main__":
    # toy_test()  # Uncomment for a quick demo
    process_corpus_line_by_line()
