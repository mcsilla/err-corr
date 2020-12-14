from copy import deepcopy
import itertools
from collections import defaultdict
import random
import csv
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from official.nlp.bert.tokenization import _is_punctuation

# TODO(mcsilla): not to use global dynamic variables.
SEQ_LENGTH=None

def detokenize_char(char_token):
    if char_token.startswith("##"):
        return char_token[2:]
    if char_token in set(['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']):
        return char_token
    if _is_punctuation(char_token):
        return char_token
    return (" " + char_token)


def corrected_tokenizer(sequence, tokenizer):
    ids_object = tokenizer(text="a" + sequence, padding='max_length', max_length=SEQ_LENGTH + 1)
    for key in ids_object:
        ids_object[key] = ids_object[key][:1] + ids_object[key][2:]
    return ids_object


class ErrorTable:
       
    def __init__(self, _tokenizer):
        self.vocab_set = set(_tokenizer.get_vocab().keys()).difference(set(['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']))
        self.vocab = sorted(self.vocab_set)
        self.tokenizer = _tokenizer
        self.error_table = None

    def create_correction(self, tokens_correct, tokens_error):
        correct_token_triples = []
        for repl_from, repl_to in itertools.zip_longest(tokens_correct, tokens_error):
            # it was probably wrong
            if repl_to is None:
                if correct_token_triples[-1][1] == self.tokenizer.pad_token:
                    correct_token_triples[-1][1] = repl_from
                else:
                    correct_token_triples[-1][2] = repl_from
                continue
            if repl_from is None:
                repl_from = self.tokenizer.pad_token
            correct_token_triples.append([repl_from, self.tokenizer.pad_token, self.tokenizer.pad_token])
        return correct_token_triples

    def add_to_error_table(self, chars1, chars2, error_table):
        tokenized_chars1 = tuple(self.tokenizer.tokenize(chars1))
        tokenized_chars2 = tuple(self.tokenizer.tokenize(chars2))
        # if there is a general OCR_errors file, then it is possible to have [UNK] here
        if "[UNK]" in tokenized_chars2 or "[UNK]" in tokenized_chars1:
            return
        error_table[tokenized_chars1].append(tokenized_chars2)
        if "##" + tokenized_chars1[0] in self.vocab_set and "##" + tokenized_chars2[0] in self.vocab_set:
            tokenized_chars1_hash = tuple(["##" + tokenized_chars1[0]] + list(tokenized_chars1[1:]))
            tokenized_chars2_hash = tuple(["##" + tokenized_chars2[0]] + list(tokenized_chars2[1:]))
            error_table[tokenized_chars1_hash].append(tokenized_chars2_hash)

    def load_table_from_file(self, file_object):
        csv_reader = csv.reader(file_object, delimiter='\t')
        error_rows = list(csv_reader)

        error_table = defaultdict(list)
        for possible_mistake in error_rows:
            for chars1 in possible_mistake:
                for chars2 in possible_mistake:
                    if chars1 == chars2:
                        continue
                    self.add_to_error_table(chars1, chars2, error_table)
        self.error_table = error_table


class CorrectionDatasetGenerator:
    error_frequency = 0.15
    sparse_frequency = 0.2
    common_extra_chars = "{}jli;|\\/(:)!1.t'"
    hyphens = "\xad-"

    def __init__(self, _tokenizer, _ocr_errors_generator):
        self.vocab_set = set(_tokenizer.get_vocab().keys()).difference(set(['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']))
        self.vocab = sorted(self.vocab_set)
        self.tokenizer = _tokenizer
        self.corr_gen = _ocr_errors_generator

    
    def replace_with_random_char(self, tokens, error_tokens, correct_tokens, token_idx):
        error_tokens.append(random.choice(self.vocab)) # can it be special token?
        correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
        token_idx += 1    


    def run(self, tokens, doc):
        error_tokens = []
        correct_tokens = []
        token_idx = 0
        random.seed(42)
        while token_idx < len(tokens):
            if random.random() < self.error_frequency: # random.random() in [0, 1)
                # Random
                if random.random() < 0.1:
                    # error_tokens.append(random.choice(self.vocab)) # can it be special token?
                    # correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                    self.replace_with_random_char(tokens, error_tokens, correct_tokens, token_idx)
                    token_idx += 1    
                # Deletion
                elif random.random() < 0.05 and correct_tokens and correct_tokens[-1][2] == self.tokenizer.pad_token:
                    for corr_token_idx in range(len(correct_tokens[-1])): # range(3)
                        if correct_tokens[-1][corr_token_idx] == self.tokenizer.pad_token:
                            correct_tokens[-1][corr_token_idx] = tokens[token_idx]
                            break
                    token_idx += 1
                # Extra char
                elif random.random() < 0.05:
                    if random.random() < 0.2 or token_idx >= len(tokens) - 1: # in this case there is no next_token
                        error_tokens.append(random.choice(self.vocab))
                        correct_tokens.append([self.tokenizer.pad_token, self.tokenizer.pad_token,
                                               self.tokenizer.pad_token])
                    else:
                        extra_token = random.choice(self.common_extra_chars)
                        if random.random() < 0.6:
                            extra_token = random.choice(self.hyphens)
                        error_tokens.append(extra_token)
                        correct_tokens.append([self.tokenizer.pad_token, self.tokenizer.pad_token,
                                               self.tokenizer.pad_token])
                        next_token = tokens[token_idx]
                        if next_token.startswith("##") and random.random() < 0.8:
                            next_token = next_token[2:]
                        error_tokens.append(next_token)
                        correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                        token_idx += 1
                # Swap
                # elif random.random() < 0.05 and token_idx < len(tokens) - 1:
                #     error_tokens.append(tokens[token_idx + 1])
                #     error_tokens.append(tokens[token_idx])
                #     correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                #     correct_tokens.append([tokens[token_idx + 1], self.tokenizer.pad_token, self.tokenizer.pad_token])
                #     token_idx += 2
                # Add space
                elif random.random() < 0.1 and "##" + tokens[token_idx] in self.vocab_set:
                    error_tokens.append("##" + tokens[token_idx])
                    correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                    token_idx += 1
                # Remove space
                elif random.random() < 0.1 and tokens[token_idx].startswith("##") and \
                        tokens[token_idx][2:] in self.vocab_set:
                    error_tokens.append(tokens[token_idx][2:])
                    correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                    token_idx += 1
                # OCR typos
                else:
                    in_table = []
                    for i in range(3):
                        if tuple(tokens[token_idx:token_idx + i + 1]) in self.corr_gen.error_table:
                            in_table.append(i)
                    if in_table:
                        join_tokens = random.choice(in_table)
                        slice_correct_tokens = tuple(tokens[token_idx:token_idx + join_tokens + 1])
                        slice_error_tokens = random.choice(self.corr_gen.error_table[slice_correct_tokens])
                        correct_tokens.extend(self.corr_gen.create_correction(slice_correct_tokens, slice_error_tokens))
                        error_tokens.extend(slice_error_tokens)
                        token_idx += join_tokens + 1
                    else:
                        error_tokens.append(random.choice(self.vocab))
                        correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                        token_idx += 1
            else:
                error_tokens.append(tokens[token_idx])
                correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
                token_idx += 1

        for token_idx in range(1, len(error_tokens)):
            if error_tokens[token_idx].startswith("##") and len(error_tokens[token_idx - 1]) == 1 and _is_punctuation(
                    error_tokens[token_idx - 1]):
                error_tokens[token_idx] = error_tokens[token_idx][2:]

        for sequence_start in range(0, len(error_tokens), SEQ_LENGTH - 2):
            if random.random() < self.sparse_frequency:
                for sparse_idx in range(0, 10):
                    sparse_start = random.randint(sequence_start,
                                                  min(sequence_start + SEQ_LENGTH - 2, len(error_tokens)))
                    sparse_length = random.randint(1, 20)
                    for token_idx in range(sparse_start, min(sparse_start + sparse_length, len(error_tokens))):
                        if error_tokens[token_idx].startswith("##"):
                            error_tokens[token_idx] = error_tokens[token_idx][2:]

        # with open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", "w") as f:
        #     standard_out = sys.stdout
        #     sys.stdout = f
        #     print("Error tokens: \n\n", error_tokens, "\n\n Correction tokens: \n\n", correct_tokens)
        #     # print("alma")
        #     sys.stdout = standard_out

        return error_tokens, correct_tokens


def generate_dataset(tokenizer, correction_dataset_generator, dataset_dir):
    input_files = tf.io.gfile.glob(dataset_dir + "*/*")
    print(input_files)
    random.shuffle(input_files)
    open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", 'w').close()
    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, mode='r') as inf:
            document = ""
            for line in inf:
                if not line.strip(): # if there are only white spaces in line 
                    document_upper = document.upper()
                    for doc in (document, document_upper):
                        tokens = tokenizer.tokenize(doc) # tokenize the textblocks between empty lines
                        document = ""
                        if not tokens:
                            continue
                        all_modified_tokens, all_corrected_tokens = correction_dataset_generator.run(tokens, doc)
                        input_len = len(all_modified_tokens)
                        for start_index in range(0, input_len, SEQ_LENGTH - 2):
                            modified_tokens = all_modified_tokens[start_index:start_index + SEQ_LENGTH - 2]
                            corrected_tokens = all_corrected_tokens[start_index:start_index + SEQ_LENGTH - 2]
                            modified_chars = [detokenize_char(token) for token in modified_tokens]
                            # dictionary with keys: 'input_ids', 'attention_mask', 'token_type_ids'
                            inputs = corrected_tokenizer("".join(modified_chars), tokenizer)
                            instance_input_ids = inputs["input_ids"]
                            instance_attention_mask = inputs["attention_mask"]
                            instance_token_type_ids = inputs["token_type_ids"]
                            corrected_input_ids = list(map(tokenizer.convert_tokens_to_ids, corrected_tokens))
                            corrected_input_ids = np.pad(corrected_input_ids,
                                                         [(1, 1), (0, 0)],
                                                         constant_values=tokenizer.pad_token_id)
                            instance_input_len = len(modified_tokens) + 2
                            corrected_input_ids = np.pad(corrected_input_ids,
                                                         [(0, SEQ_LENGTH - instance_input_len), (0, 0)],
                                                         constant_values=-100)
                            corrected_input_ids = np.swapaxes(corrected_input_ids, 0, 1)
                            with open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", "a") as f:
                                standard_out = sys.stdout
                                sys.stdout = f
                                print((instance_input_ids, instance_attention_mask, instance_token_type_ids), \
                                  (corrected_input_ids[0], corrected_input_ids[1], corrected_input_ids[2]))
                                sys.stdout = standard_out
                            yield (instance_input_ids, instance_attention_mask, instance_token_type_ids), \
                                  (corrected_input_ids[0], corrected_input_ids[1], corrected_input_ids[2])

                else:
                    document += line


def int64feature(int_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def printable_format(tokenizer, token_ids):
    tokens = tokenizer.convert_ids_to_tokens([token_id for token_id in token_ids if token_id >= 0])
    return "".join([detokenize_char(token) for token in tokens])


def write_examples_to_tfrecord(examples, tf_records_writer):
    random.shuffle(examples)
    for instance in examples:
        tf_records_writer.write(instance.SerializeToString())

