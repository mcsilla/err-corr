import logging
import time
from copy import deepcopy
import argparse
import itertools
from collections import OrderedDict, defaultdict
import random
import csv

import sys

import numpy as np
import tensorflow as tf

from transformers import BertTokenizerFast
from official.nlp.bert.tokenization import _is_punctuation


class CorrectionDatasetGenerator:
    error_frequency = 0.15
    sparse_frequency = 0.2
    common_extra_chars = "{}jli;|\\/(:)!1.t'"
    hyphens = "\xad-"

    def __init__(self, _tokenizer):
        self.vocab = sorted(_tokenizer.get_vocab().keys()) 
        self.vocab_set = set(self.vocab)
        self.tokenizer = _tokenizer
        self.error_table, self.correction_table = self.create_error_table_from_file("ocr_errors.txt")

    def create_correction(self, tokens1, tokens2):
        correct_tokens = []
        error_tokens = []
        for repl_from, repl_to in itertools.zip_longest(tokens1, tokens2):
            # it was probably wrong
            if repl_to is None:
                if correct_tokens[-1][1] == self.tokenizer.pad_token:
                    correct_tokens[-1][1] = repl_from
                else:
                    correct_tokens[-1][2] = repl_from
                continue
            if repl_from is None:
                repl_from = self.tokenizer.pad_token
            error_tokens.append(repl_to)
            correct_tokens.append([repl_from, self.tokenizer.pad_token, self.tokenizer.pad_token])
        return correct_tokens, error_tokens

    def add_to_error_table(self, chars1, chars2, error_table, correction_table):
        tokenized_chars1 = tuple(self.tokenizer.tokenize(chars1))
        tokenized_chars2 = tuple(self.tokenizer.tokenize(chars2))
        if "[UNK]" in tokenized_chars2 or "[UNK]" in tokenized_chars1:
            return
        error_table[tokenized_chars1].append(tokenized_chars2)
        correction_table[tokenized_chars1, tokenized_chars2] = self.create_correction(tokenized_chars1,
                                                                                tokenized_chars2)
        if "##" + tokenized_chars1[0] in self.vocab_set and "##" + tokenized_chars2[0] in self.vocab_set:
            tokenized_chars1_hash = tuple(["##" + tokenized_chars1[0]] + list(tokenized_chars1[1:]))
            tokenized_chars2_hash = tuple(["##" + tokenized_chars2[0]] + list(tokenized_chars2[1:]))
            error_table[tokenized_chars1_hash].append(tokenized_chars2_hash)
            correction_table[tokenized_chars1_hash, tokenized_chars2_hash] = self.create_correction(
                tokenized_chars1_hash, tokenized_chars2_hash)

    def create_error_table_from_file(self, file_path):
        with open(file_path, encoding="utf-8") as errors_file:
            csv_reader = csv.reader(errors_file, delimiter='\t')
            error_rows = list(csv_reader)

        error_table = defaultdict(list)
        correction_table = {}
        for possible_mistake in error_rows:
            for chars1 in possible_mistake:
                for chars2 in possible_mistake:
                    if chars1 == chars2:
                        continue
                    self.add_to_error_table(chars1, chars2, error_table, correction_table)

        print(correction_table)
        return error_table, correction_table

    def run(self, tokens):
        error_tokens = []
        correct_tokens = []
        token_idx = 0
        random.seed(42)
        while token_idx < len(tokens):
            if random.random() < self.error_frequency: # random.random() in [0, 1)
                # Random
                if random.random() < 0.1:
                    error_tokens.append(random.choice(self.vocab))
                    correct_tokens.append([tokens[token_idx], self.tokenizer.pad_token, self.tokenizer.pad_token])
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
                        if tuple(tokens[token_idx:token_idx + i + 1]) in self.error_table:
                            in_table.append(i)
                    if in_table:
                        join_tokens = random.choice(in_table)
                        slice_correct_tokens = tuple(tokens[token_idx:token_idx + join_tokens + 1])
                        slice_error_tokens = random.choice(self.error_table[slice_correct_tokens])
                        correct_tokens.extend(deepcopy(self.correction_table[slice_correct_tokens,
                                                                             slice_error_tokens][0]))
                        error_tokens.extend(deepcopy(self.correction_table[slice_correct_tokens,
                                                                           slice_error_tokens][1]))
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

        return error_tokens, correct_tokens


def generate_dataset(tokenizer, correction_dataset_generator, dataset_dir):
    # print('generate_dataset')
    input_files = tf.io.gfile.glob(dataset_dir + "/input_file.txt")
    # print(input_files)
    # random.shuffle(input_files)
    open('output.txt', 'w').close()
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
                        all_modified_tokens, all_corrected_tokens = correction_dataset_generator.run(tokens)
                        with open("output.txt", 'a') as f:
                                original = sys.stdout
                                sys.stdout = f
                                print("All tokens A:")
                                print(all_modified_tokens, "\n\n", all_corrected_tokens, "\n\n")
                                sys.stdout = original
                        input_len = len(all_modified_tokens)
                        for start_index in range(0, input_len, SEQ_LENGTH - 2):
                            modified_tokens = all_modified_tokens[start_index:start_index + SEQ_LENGTH - 2] # ? end can be out of range
                            corrected_tokens = all_corrected_tokens[start_index:start_index + SEQ_LENGTH - 2]
                            # isn't cls and sep id added automatically ?
                            modified_input_ids = [tokenizer.cls_token_id] + \
                                tokenizer.convert_tokens_to_ids(modified_tokens) + \
                                [tokenizer.sep_token_id]
                            instance_input_len = len(modified_input_ids)
                            instance_input_ids = np.pad(modified_input_ids, (0, SEQ_LENGTH - instance_input_len),
                                                        constant_values=(0, 0))
                            instance_attention_mask = np.concatenate((
                                np.ones(instance_input_len, dtype=np.int32),
                                np.zeros(SEQ_LENGTH - instance_input_len, dtype=np.int32)
                            ))
                            instance_token_type_ids = np.zeros(SEQ_LENGTH, dtype=np.int32)
                            corrected_input_ids = [[tokenizer.pad_token_id, tokenizer.pad_token_id,
                                                    tokenizer.pad_token_id]]
                            for corrected_token in corrected_tokens:
                                corrected_token_id = tokenizer.convert_tokens_to_ids(corrected_token)
                                corrected_input_ids.append(corrected_token_id)
                            corrected_input_ids.append([tokenizer.pad_token_id, tokenizer.pad_token_id,
                                                        tokenizer.pad_token_id])
                            corrected_input_ids = np.pad(corrected_input_ids,
                                                         [(0, SEQ_LENGTH - instance_input_len), (0, 0)],
                                                         constant_values=((0, -100), (0, 0)))
                            corrected_input_ids = np.swapaxes(corrected_input_ids, 0, 1)
                            with open("output.txt", 'a') as f:
                                original = sys.stdout
                                sys.stdout = f
                                print("Output of generate_dataset()\n")
                                print(instance_input_ids, "\n", instance_attention_mask, "\n", instance_token_type_ids, "\n\n",\
                                    corrected_input_ids[0], "\n", corrected_input_ids[1], "\n", corrected_input_ids[2], "\n\n")
                                sys.stdout = original
                            yield (instance_input_ids, instance_attention_mask, instance_token_type_ids), \
                                  (corrected_input_ids[0], corrected_input_ids[1], corrected_input_ids[2])

                else:
                    document += line


def int64feature(int_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def printable_format(tokenizer, token_ids):
    return tokenizer.convert_ids_to_tokens([token_id for token_id in token_ids if token_id >= 0])


def write_examples_to_tfrecord(examples, tf_records_writer):
    random.shuffle(examples)
    for instance in examples:
        tf_records_writer.write(instance.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # creating an ArgumentParser object

    # input data and model directories
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    # parser.add_argument('--tokenizer_config', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, required=True)
    parser.add_argument('--dupe_factor', type=int, default=10)

    args, _ = parser.parse_known_args()

    SEQ_LENGTH = args.sequence_length
    # character_tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_config) # why is from_pretrained needed? 
    character_tokenizer = BertTokenizerFast("config/vocab.txt", do_lower_case=False)
    dataset_generator = CorrectionDatasetGenerator(character_tokenizer)
    writer = tf.io.TFRecordWriter(args.output_file, options="GZIP")
    logging.basicConfig(level=logging.INFO)
    inst_idx = 0
    start_time = time.time()

    for repeat in range(args.dupe_factor):
        example_cache = []
        for inputs, outputs in generate_dataset(character_tokenizer, dataset_generator, args.input_file):
            inst_idx += 1

    #         feature = OrderedDict()
    #         feature['input_ids'] = int64feature(inputs[0])
    #         feature['attention_mask'] = int64feature(inputs[1])
    #         feature['token_type_ids'] = int64feature(inputs[2])
    #         feature['output1'] = int64feature(outputs[0])
    #         feature['output2'] = int64feature(outputs[1])
    #         feature['output3'] = int64feature(outputs[2])
    #         example = tf.train.Example(features=tf.train.Features(feature=feature))
    #         example_cache.append(example)

    #         if not (inst_idx % 10000):
    #             logging.info(f"*** repeat: {repeat} total_instances: {inst_idx} time: {time.time() - start_time}s ***")

    #         if len(example_cache) >= 100000:
    #             write_examples_to_tfrecord(example_cache, writer)
    #             example_cache = []

    #         if inst_idx < 20:
    #             logging.info("*** Example ***")
    #             logging.info("text_with_errors: " + str(printable_format(character_tokenizer, inputs[0])))
    #             logging.info(f"attention_mask: {inputs[1]}")
    #             logging.info(f"token_type_ids: {inputs[2]}")
    #             logging.info("text_corrected1: " + str(printable_format(character_tokenizer, outputs[0])))
    #             logging.info("text_corrected1: " + str(printable_format(character_tokenizer, outputs[1])))
    #             logging.info("text_corrected1: " + str(printable_format(character_tokenizer, outputs[2])))

    #     write_examples_to_tfrecord(example_cache, writer)

    # writer.close()
    # runtime = time.time() - start_time
    # logging.info(f"*** {inst_idx} files wrote to file in {runtime} seconds ***")
