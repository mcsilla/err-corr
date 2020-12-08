from tokenizers import BertWordPieceTokenizer
from pathlib import Path
import argparse
import re
import os
import json

SPEC_TOKENS = ['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']
BASE_DIR = Path(__file__).parent
MIN_FREQ = 2
SIZE_OF_ALPHABET = 180

def read_vocab(file_path = "data/tokenizer/vocab"):
    with open(BASE_DIR / file_path, "r") as f:
        content = f.read()
        return content


def prepare_alphabet(vocab):
    alphabet = [token for token in SPEC_TOKENS]
    prefixed_char_pattern = re.compile("##.")
    for token in vocab:
        if len(token) == 1 or prefixed_char_pattern.fullmatch(token):
            alphabet.append(token)
    return alphabet


def write_alphabet_to_file(alphabet, file_path = "data/tokenizer/alphabet"):
    with open(BASE_DIR / file_path, "w") as f:
        for char in alphabet:
            f.write(f"{char}\n")


def main(language):
    # Initialize an empty BERT tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    cleaned_dir = BASE_DIR / "data/wikiextracted" / language / "cleaned"

    # prepare text files to train vocab on them
    # use only one subdir
    # files = [str(file_path) for file_path in cleaned_dir.glob("AA/wiki_*")]
    # use all wiki articles (in the given language)
    files = [str(file_path) for file_path in cleaned_dir.glob("**/wiki_*")]

    # train BERT tokenizer
    tokenizer.train(
        files,
        # vocab_size=100, # default value is 30000
        min_frequency=MIN_FREQ,
        show_progress=True,
        special_tokens=SPEC_TOKENS,
        limit_alphabet=SIZE_OF_ALPHABET, # default value is 1000
        wordpieces_prefix="##"
    )

    # save the vocab
    tokenizer.save("data/tokenizer/vocab")

    # save the alphabet
    vocab = json.loads(read_vocab())['model']['vocab']
    alphabet = prepare_alphabet(vocab)
    write_alphabet_to_file(alphabet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True, help="wiki language code")
    language = parser.parse_args().language
    main(language)