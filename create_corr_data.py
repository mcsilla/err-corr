#!/usr/bin/env python3

from pathlib import Path

BASE_DIR = Path(__file__).parent
LANGUAGE = "ro"

def read_files(language = LANGUAGE):
    cleaned_dir = BASE_DIR / "data/wikiextracted" / language / "cleaned"
    # for file_path in cleaned_dir.glob("**/wiki_*"):
    for file_path in cleaned_dir.glob("AA/wiki_00"):
        with open(file_path) as f:
           yield f.readlines()

if __name__ == "__main__":
    for lines in read_files():
        print(lines[:10])
        break