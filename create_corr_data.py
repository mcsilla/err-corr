#!/usr/bin/env python3

from pathlib import Path
from transformers import BertTokenizerFast
import numpy as np
import tensorflow as tf
import random

BASE_DIR = Path(__file__).parent
LANGUAGE = "ro"

def read_files(language = LANGUAGE):
    cleaned_dir = BASE_DIR / "data/wikiextracted" / language / "cleaned"
    # for file_path in cleaned_dir.glob("**/wiki_*"):
    for file_path in cleaned_dir.glob("AA/wiki_00"):
        with open(file_path) as f:
           yield f.readlines()

if __name__ == "__main__":
    character_tokenizer = BertTokenizerFast(BASE_DIR / "data/tokenizer/alphabet", do_lower_case=False)
    max_line_lengths = []
    for lines in read_files():
        # max_index = np.argmax([len(character_tokenizer.tokenize(line)) for line in lines])
        max_line_lengths.append(max([len(character_tokenizer.tokenize(line)) for line in lines]))
        print(max_line_lengths)
        break
    max_line_length = max(max_line_lengths)
    print(max_line_length)