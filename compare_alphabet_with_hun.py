#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).parent
ALPHABET_ROOT = BASE_DIR / "data/tokenizer"
ALPHABET_HUN =  BASE_DIR / "data/tokenizer/hu/alphabet"

def get_alphabet(path):
    with open(path) as f:
        return f.readlines()


def read_files(language):
    cleaned_dir = BASE_DIR / "data/wikiextracted" / language / "cleaned"
    for file_path in cleaned_dir.glob("**/wiki_*"):
        with open(file_path) as f:
           yield f.read() 


def get_bare_chars(alphabet_lines):
    chars = set()
    for line in alphabet_lines:
        if len(line) >= 3:
            chars.add(line[2:].strip("\n"))
        else:
            chars.add(line.strip("\n"))
    return chars


def get_setminus(path1, path2 = ALPHABET_HUN):
    setminus = []
    second_alpha = get_bare_chars(get_alphabet(path2))
    for char in get_bare_chars(get_alphabet(path1)):
        if char not in second_alpha:
            setminus.append(char) 
    return setminus


def calculate_frequency(chars, language):
    frequency_dict = {char : 0 for char in chars}
    for content in read_files(language):
        for char in content:
            if char in frequency_dict:
                frequency_dict[char] += 1
    return dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))


def main():
    parser = argparse.ArgumentParser() # creating an ArgumentParser object

    # input data and model directories
    # parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)

    args, _ = parser.parse_known_args()
    language = args.language
    alphabet_path = ALPHABET_ROOT / language / "alphabet"
    # print(f"Only RO:\n{get_setminus(ALPHABET_RO, ALPHABET_HUN)}")
    # print(f"Only HUN:\n{get_setminus(ALPHABET_HUN, ALPHABET_RO)}")
    output_root = Path("/home/mcsilla/machine_learning/gitrepos/err-corr/data/alphabets") / language
    os.makedirs(output_root, exist_ok=True) 
    with open( output_root / "new_chars.txt", "w") as f:
        standard_out = sys.stdout
        sys.stdout = f
        for key, value in calculate_frequency(get_setminus(alphabet_path), language).items():
            print(f"{key}: {value}")
        sys.stdout = standard_out



if __name__ == "__main__":
    main()