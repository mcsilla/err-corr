#!/usr/bin/env python3

import re
import os
from pathlib import Path
import multiprocessing
import argparse

OPEN_DOC_TAG_AND_TITLE = re.compile("^<doc.*>\n.*\n", re.MULTILINE)
EMPTY_LINES = re.compile("^\s*\n", re.MULTILINE)
CLOSE_DOC_TAG = re.compile("^</doc>$", re.MULTILINE)
END_OF_SENTENCE = re.compile("(?<=[\w\"”)\.][\w\"” )][.?!]) +(?=[A-Z\"])")
BAD_NEW_LINE = re.compile("(?<=[^.?!])\n", re.MULTILINE)
BR_TAG = re.compile("<br>", re.MULTILINE)
LINK_WITH_PIPE = re.compile(r"\[\[([^\]\[]*)\|([^\|\]\[]+)\]\]")
LINK_WITHOUT_PIPE = re.compile(r"\[\[([^\|\]\[]+)\]\]")


def read_files(language):
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "data/wikiextracted" / language / "raw"
    cleaned_dir = base_dir / "data/wikiextracted" / language / "cleaned"
    for file_path in raw_dir.glob("**/wiki_*"):
        cleaned_file_path = cleaned_dir / file_path.relative_to(raw_dir)
        with open(file_path) as f:
           yield cleaned_file_path, f.read()   


def clean_lines(content):
    rules = (
        lambda content: OPEN_DOC_TAG_AND_TITLE.sub("", content),
        lambda content: CLOSE_DOC_TAG.sub("", content),
        lambda content: BAD_NEW_LINE.sub(" ", content),
        lambda content: BR_TAG.sub("\n", content),
        lambda content: END_OF_SENTENCE.sub("\n", content),
        lambda content: EMPTY_LINES.sub("", content),
        lambda content: LINK_WITH_PIPE.sub(r"\2", content),   
        lambda content: LINK_WITHOUT_PIPE.sub(r"\1", content)    
    )
    for rule in rules:
        content = rule(content)
    return content


def write_file(content, path_to_file):
    os.makedirs(path_to_file.parent, exist_ok=True)
    with open(path_to_file, "w") as f:
        f.write(content)


def worker_function(tup):
    path_to_file, content = tup
    write_file(clean_lines(content), path_to_file)


def main(language):
    with multiprocessing.Pool(2) as pool:
        pool.map(worker_function, read_files(language))
    # for rel_path_end, content in read_files():
    #     rel_path = os.path.join("data/wikiextracted/cleaned", rel_path_end)
    #     write_file(clean_lines(content), rel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True, help="wiki language code")
    language = parser.parse_args().language
    main(language)
