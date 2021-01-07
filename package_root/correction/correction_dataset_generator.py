from copy import deepcopy
import itertools
from collections import defaultdict
import random
import csv
from pathlib import Path
import sys
import tqdm
import numpy as np
import tensorflow as tf

from official.nlp.bert.tokenization import _is_punctuation


def detokenize_char(tokenizer, char_token):
    if char_token.startswith("##"):
        return char_token[2:]
    if char_token in set([tokenizer.cls_token, '[SEP]', '[MASK]', '[PAD]', '[UNK]']):
        return char_token
    if _is_punctuation(char_token):
        return char_token
    return (" " + char_token)


def corrected_tokenizer(sequence, tokenizer, seq_length):
    ids_object = tokenizer(text="a" + sequence, padding='max_length', max_length=seq_length + 1)
    for key in ids_object:
        ids_object[key] = ids_object[key][:1] + ids_object[key][2:]
    return ids_object


class ErrorTable:
    def __init__(self, _tokenizer):
        self.tokenizer = _tokenizer
        self.vocab_set = set(_tokenizer.get_vocab().keys()).difference()
        self.error_table = None

    def create_correction(self, tokens_correct, tokens_error):
        correct_token_triples = []
        for repl_from, repl_to in itertools.zip_longest(tokens_correct, tokens_error):
            if repl_to is None and len(correct_token_triples) < 3:
                correct_token_triples[-1].append(repl_from)
                continue
            if repl_from is None:
                repl_from = self.tokenizer.pad_token
            correct_token_triples.append([repl_from])
        return correct_token_triples

    def add_to_error_table(self, chars1, chars2, error_table):
        tokenized_chars1 = tuple(self.tokenizer.tokenize(chars1))
        tokenized_chars2 = tuple(self.tokenizer.tokenize(chars2))
        if "[UNK]" in tokenized_chars2 or "[UNK]" in tokenized_chars1:
            return
        error_table[tokenized_chars1].append(tokenized_chars2)
        if "##" + tokenized_chars1[0] in self.vocab_set and "##" + tokenized_chars2[0] in self.vocab_set:
            tokenized_chars1_hash = tuple(["##" + tokenized_chars1[0]] + list(tokenized_chars1[1:]))
            tokenized_chars2_hash = tuple(["##" + tokenized_chars2[0]] + list(tokenized_chars2[1:]))
            error_table[tokenized_chars1_hash].append(tokenized_chars2_hash)

    def load_table_from_file(self, file_object):
        csv_reader = csv.reader(file_object, delimiter='\t', quotechar=None)
        error_rows = list(csv_reader)
        error_table = defaultdict(list)
        for possible_mistake in error_rows:
            for chars1 in possible_mistake:
                for chars2 in possible_mistake:
                    if chars1 == chars2:
                        continue
                    self.add_to_error_table(chars1, chars2, error_table)
        self.error_table = error_table

    
    def get_error(self, tokens_list):
        error_table_dict = dict(self.error_table)
        key = tuple(tokens_list)
        if key in error_table_dict:
            return random.choice(error_table_dict[key])
        return None


class CorrectionDatasetGenerator:
    error_frequency = 0.15
    sparse_frequency = 0.2
    dense_frequency = 0.2
    common_extra_chars = "{}jli;|\\/(:)!1.t'"
    hyphens = "\xad-"

    def __init__(self, _tokenizer, _ocr_errors_generator, _seq_length):
        self.tokenizer = _tokenizer
        non_vocab_tokens = (
            self.tokenizer.cls_token,
            self.tokenizer.sep_token,
            self.tokenizer.mask_token, 
            self.tokenizer.pad_token)
        self.vocab_set = set(_tokenizer.get_vocab().keys()).difference(non_vocab_tokens)
        self.vocab = sorted(self.vocab_set)
        self.seq_length = _seq_length
        self.corr_gen = _ocr_errors_generator

        self._tokens = None
        self._error_tokens = None
        self._correct_tokens = None

    
    def replace_with_random_token(self, token_idx):
        self._error_tokens.append(random.choice(self.vocab))
        self._correct_tokens.append([self._tokens[token_idx]])


    def delete_token(self, token_idx):
        self._correct_tokens[-1].append(self._tokens[token_idx])


    def add_extra_token(self, token_idx):
        if random.random() < 0.2 or token_idx >= len(self._tokens) - 1:
            self._error_tokens.append(random.choice(self.vocab))
            self._correct_tokens.append([self.tokenizer.pad_token])
            return token_idx
        else:
            extra_token = random.choice(self.common_extra_chars)
            if random.random() < 0.6:
                extra_token = random.choice(self.hyphens)
            self._error_tokens.append(extra_token)
            self._correct_tokens.append([self.tokenizer.pad_token])
            next_token = self._tokens[token_idx]
            if next_token.startswith("##") and random.random() < 0.8:
                next_token = next_token[2:]
            self._error_tokens.append(next_token)
            self._correct_tokens.append([self._tokens[token_idx]])
            return token_idx + 1

    def add_space(self, token_idx):
        self._error_tokens.append("##" + self._tokens[token_idx])
        self._correct_tokens.append([self._tokens[token_idx]])


    def remove_space(self, token_idx):
        self._error_tokens.append(self._tokens[token_idx][2:])
        self._correct_tokens.append([self._tokens[token_idx]])


    def make_ocr_typo(self, token_idx):
        in_table = []
        for i in range(3):
            if self.corr_gen.get_error(self._tokens[token_idx:token_idx + i + 1]):
                in_table.append(i)
        if in_table:
            join_tokens = random.choice(in_table)
            slice_correct_tokens = self._tokens[token_idx:token_idx + join_tokens + 1]
            slice_error_tokens = self.corr_gen.get_error(slice_correct_tokens)
            self._correct_tokens.extend(self.corr_gen.create_correction(slice_correct_tokens, slice_error_tokens))
            self._error_tokens.extend(slice_error_tokens)
            return token_idx + join_tokens + 1
        else:
            self._error_tokens.append(random.choice(self.vocab))
            self._correct_tokens.append([self._tokens[token_idx]])
            return token_idx + 1

    def make_sparse(self):
        for sequence_start in range(0, len(self._error_tokens), self.seq_length - 2):
            if random.random() < self.sparse_frequency:
                for sparse_idx in range(0, 10):
                    sparse_start = random.randint(sequence_start,
                                                  min(sequence_start + self.seq_length - 2, len(self._error_tokens)))
                    sparse_length = random.randint(1, 20)
                    for token_idx in range(sparse_start, min(sparse_start + sparse_length, len(self._error_tokens))):
                        if self._error_tokens[token_idx].startswith("##"):
                            self._error_tokens[token_idx] = self._error_tokens[token_idx][2:]

    def make_dense(self):
        for sequence_start in range(0, len(self._error_tokens), self.seq_length - 2):
            if random.random() < self.dense_frequency:
                for dense_idx in range(0, 10):
                    dense_start = random.randint(sequence_start,
                                                  min(sequence_start + self.seq_length - 2, len(self._error_tokens)))
                    dense_length = random.randint(1, 20)
                    for token_idx in range(dense_start, min(dense_start + dense_length, len(self._error_tokens))):
                        if len(self._error_tokens[token_idx]) == 1 and "##" + self._error_tokens[token_idx] in self.vocab_set:
                            self._error_tokens[token_idx] = "##" + self._error_tokens[token_idx]

    def make_old(self, token_idx):
        # ha a hibagenerálás előtt csináljuk, akkor üres az error és correct tokens tömb
        pass



    def reset_space_after_punctuation(self):
        for token_idx in range(1, len(self._error_tokens)):
            if self._error_tokens[token_idx].startswith("##") and len(self._error_tokens[token_idx - 1]) == 1 and _is_punctuation(
                    self._error_tokens[token_idx - 1]):
                self._error_tokens[token_idx] = self._error_tokens[token_idx][2:]

    def pad_to_length_3(self):
        for correct_token in self._correct_tokens:
            correct_token += [self.tokenizer.pad_token] * (3 - len(correct_token))

    def run(self, tokens):
        self._tokens = tokens
        self._error_tokens = []
        self._correct_tokens = []
        token_idx = 0
        # random.seed(42)
        while token_idx < len(self._tokens):
            if random.random() < self.error_frequency: # random.random() in [0, 1)
                if random.random() < 0.1:
                    self.replace_with_random_token(token_idx)
                    token_idx += 1    
                elif random.random() < 0.05 and self._correct_tokens and len(self._correct_tokens[-1]) < 3:
                    self.delete_token(token_idx)
                    token_idx += 1
                elif random.random() < 0.05:
                    token_idx = self.add_extra_token(token_idx)
                elif random.random() < 0.1 and "##" + self._tokens[token_idx] in self.vocab_set:
                    self.add_space(token_idx)
                    token_idx += 1
                elif random.random() < 0.1 and self._tokens[token_idx].startswith("##") and \
                        self._tokens[token_idx][2:] in self.vocab_set:
                    self.remove_space(token_idx)
                    token_idx += 1
                else:
                    token_idx = self.make_ocr_typo(token_idx)
            else:
                self._error_tokens.append(tokens[token_idx])
                self._correct_tokens.append([tokens[token_idx]])
                token_idx += 1



        self.make_sparse()

        self.make_dense()

        self.reset_space_after_punctuation()

        self.pad_to_length_3()

        # with open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", "w") as f:
        #     standard_out = sys.stdout
        #     sys.stdout = f
        #     print("Error tokens: \n\n", self._error_tokens, "\n\n Correction tokens: \n\n", correct_tokens)
        #     # print("alma")
        #     sys.stdout = standard_out

        return self._error_tokens, self._correct_tokens
    
    def create_input(self, tokens):
        modified_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(modified_tokens)
        input_len = len(modified_tokens)
        instance_input_ids = np.pad(input_ids, (0, self.seq_length - input_len), constant_values=self.tokenizer.pad_token_id)
        instance_attention_mask = np.concatenate((
            np.ones(input_len, dtype=np.int32),
            np.zeros(self.seq_length - input_len, dtype=np.int32)
        ))
        instance_token_type_ids = np.zeros(self.seq_length, dtype=np.int32)
        return {
            "input_ids": instance_input_ids,
            "attention_mask": instance_attention_mask,
            "token_type_ids": instance_token_type_ids
        }

    def create_label(self, corrected_tokens):
        padded_tokens = np.pad(corrected_tokens, [(1, 1), (0, 0)], constant_values=self.tokenizer.pad_token)
        input_ids = list(map(self.tokenizer.convert_tokens_to_ids, padded_tokens))
        input_len = len(padded_tokens)
        input_ids = np.pad(input_ids,
                                    [(0, self.seq_length - input_len), (0, 0)],
                                    constant_values=-100)
        input_ids = np.swapaxes(input_ids, 0, 1)
        return {
            "label_0": input_ids[0],
            "label_1": input_ids[1],
            "label_2": input_ids[2]
        }

    def restore_text_from_corrected_tokens(self, corrected_tokens):
        text = "".join([detokenize_char(self.tokenizer, token) for triple in corrected_tokens for token in triple if token != "[PAD]"])
        return text

    def translate_old_hun(self, corrected_tokens):
        return(corrected_tokens)


    def generate_dataset(self, dataset_dir, thread, seed_input):
        NUM_OF_THREADS = 1
        input_files = sorted(tf.io.gfile.glob(dataset_dir + "/AA/wiki_00_test"))
        # input_files = sorted(tf.io.gfile.glob(dataset_dir + "/*/wiki_*"))
        random.seed(42)
        #random.shuffle(input_files)
        # input_files = tf.io.gfile.glob(dataset_dir + "/*")
        thread_input_length = len(input_files) // NUM_OF_THREADS
        if (len(input_files) % NUM_OF_THREADS):
            thread_input_length += 1 
        thread_start = (thread - 1) * thread_input_length
        thread_input_files = input_files[thread_start:thread_start + thread_input_length]
        # print(thread_input_files)
        open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", 'w').close()
        for input_file in tqdm.tqdm(thread_input_files):
            with tf.io.gfile.GFile(input_file, mode='r') as inf:
                document_lines = []
                for line in inf:
                    if not line.strip(): # if there are only white spaces in line 
                        document = "".join(document_lines)
                        document_upper = document.upper()
                        document_lines = []
                        for doc in (document, document_upper):
                            # doc2 = make_old(doc)
                            # tokens2 = self.tokenizer.tokenize(doc2)
                            tokens = self.tokenizer.tokenize(doc) # tokenize the textblocks between empty lines
                            if not tokens:
                                continue
                            all_modified_tokens, all_corrected_tokens = self.run(tokens)
                            input_len = len(all_modified_tokens)
                            for start_index in range(0, input_len, self.seq_length - 2):
                                modified_tokens = all_modified_tokens[start_index:start_index + self.seq_length - 2]
                                corrected_tokens = all_corrected_tokens[start_index:start_index + self.seq_length - 2]
                                inputs = self.create_input(modified_tokens)
                                labels = self.create_label(corrected_tokens)
                                with open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", "a") as f:
                                    standard_out = sys.stdout
                                    sys.stdout = f
                                    print(self.restore_text_from_corrected_tokens(corrected_tokens))
                                    # print(inputs["input_ids"], "\n", inputs["attention_mask"], "\n", inputs["token_type_ids"], "\n\n",\
                                    # labels["label_0"], "\n", labels["label_1"], "\n", labels["label_2"], "\n\n")
                                    sys.stdout = standard_out
                                yield (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]), \
                                    (labels["label_0"], labels["label_1"], labels["label_2"])
                    else:
                        document_lines.append(line)


def int64feature(int_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def printable_format(tokenizer, token_ids):
    tokens = tokenizer.convert_ids_to_tokens([token_id for token_id in token_ids if token_id >= 0])
    return "".join([detokenize_char(tokenizer, token) for token in tokens])


def write_examples_to_tfrecord(examples, tf_records_writer):
    random.shuffle(examples)
    for instance in examples:
        tf_records_writer.write(instance.SerializeToString())

