# a tokens tömbön lehetne futtatni azt, hogy 256-onként néha régiesít de
# régiesítve nem ugyanannyi karakter lesz
# az errorizálás is módosítja a karakterek számát 

import itertools
from collections import defaultdict
import random
import csv
import sys
import tqdm
import numpy as np
import tensorflow as tf
# from useful import ManipulateTokens
from official.nlp.bert.tokenization import _is_punctuation

class ManipulateTokens:

    def __init__(self, _tokenizer):
        self.tokenizer = _tokenizer

    def detokenize_char(self, char_token):
        if char_token.startswith("##"):
            return char_token[2:]
        if char_token in set(['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']):
            return char_token
        if _is_punctuation(char_token):
            return char_token
        return (" " + char_token)

    def corrected_tokenizer(sequence, tokenizer, seq_length):
        ids_object = tokenizer(text="a" + sequence, padding='max_length', max_length=seq_length + 1)
        for key in ids_object:
            ids_object[key] = ids_object[key][:1] + ids_object[key][2:]
        return ids_object

    def tokenize_with_starting_hash(self, sequence):
        return self.tokenizer.tokenize("a" + sequence)[1:]


    def tokenize_all_versions(self, sequence):
        tokens_versions = [
            self.tokenizer.tokenize(sequence),
            self.tokenize_with_starting_hash(sequence),
            self.tokenizer.tokenize(sequence.upper()),
            self.tokenize_with_starting_hash(sequence.upper()),
            self.tokenizer.tokenize(sequence.capitalize()),
            self.tokenize_with_starting_hash(sequence.capitalize())
        ]
        return tuple([self.correct_tokens_with_tokenized_PAD(tokens) for tokens in tokens_versions])

    def tokenize_all_versions_with_starting_hash(self, sequence):
        tokens_versions = [
            self.tokenize_with_starting_hash(sequence),
            self.tokenize_with_starting_hash(sequence.upper()),
            self.tokenize_with_starting_hash(sequence.capitalize())
        ]
        return tuple([self.correct_tokens_with_tokenized_PAD(tokens) for tokens in tokens_versions])

    def tokenize_all_versions_without_starting_hash(self, sequence):
        tokens_versions = [
            self.tokenizer.tokenize(sequence),
            self.tokenizer.tokenize(sequence.upper()),
            self.tokenizer.tokenize(sequence.capitalize()),
        ]
        return tuple([self.correct_tokens_with_tokenized_PAD(tokens) for tokens in tokens_versions])



    def correct_tokens_with_tokenized_PAD(self, tokenized_chars):
        corrected_chars = []
        i = 0
        # az "[" csak a "[PAD]"-nál szerepelhet az ocr_errors.txt-ben!
        last_pad = False
        while i < len(tokenized_chars):
            next_char = tokenized_chars[i]
            if next_char == "[":
                corrected_chars.append("[PAD]")
                i += 5
                last_pad = True
                if i < len(tokenized_chars) and tokenized_chars[i] == "|":
                    last_pad = False
                    i += 1
            elif last_pad == True:
                corrected_chars.append("##" + next_char)
                last_pad = False
                i += 1
            else:
                corrected_chars.append(next_char)
                i += 1
        return corrected_chars

    def restore_text_from_corrected_tokens(self, corrected_tokens):
        text = "".join([self.detokenize_char(token) for triple in corrected_tokens for token in triple if token != "[PAD]"])
        return text


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

    
    def correct_tokenizer(self, tokenized_chars):
        corrected_chars = []
        i = 0
        # az "[" csak a "[PAD]"-nál szerepelhet az ocr_errors.txt-ben!
        last_pad = False
        while i < len(tokenized_chars):
            next_char = tokenized_chars[i]
            if next_char == "[":
                corrected_chars.append("[PAD]")
                i += 5
                last_pad = True
            elif last_pad == True:
                corrected_chars.append("##" + next_char)
                last_pad = False
                i += 1
            else:
                corrected_chars.append(next_char)
                i += 1

        return corrected_chars

    def add_to_error_table(self, chars1, chars2, error_table):
        tokenized_chars1 = tuple(self.correct_tokenizer(self.tokenizer.tokenize(chars1)))
        tokenized_chars2 = tuple(self.correct_tokenizer(self.tokenizer.tokenize(chars2)))
        if "[UNK]" in tokenized_chars2 or "[UNK]" in tokenized_chars1:
            return
        error_table[tokenized_chars1].append(tokenized_chars2)
        if "##" + tokenized_chars1[0] in self.vocab_set and "##" + tokenized_chars2[0] in self.vocab_set:
            tokenized_chars1_hash = tuple(["##" + tokenized_chars1[0]] + list(tokenized_chars1[1:]))
            tokenized_chars2_hash = tuple(["##" + tokenized_chars2[0]] + list(tokenized_chars2[1:]))
            error_table[tokenized_chars1_hash].append(tokenized_chars2_hash)

    # def add_to_error_table(self, chars1, chars2, error_table):
    #     tokenized_chars1 = tuple(self.tokenizer.tokenize(chars1))
    #     tokenized_chars2 = tuple(self.tokenizer.tokenize(chars2))
    #     if "[UNK]" in tokenized_chars2 or "[UNK]" in tokenized_chars1:
    #         return
    #     error_table[tokenized_chars1].append(tokenized_chars2)
    #     if "##" + tokenized_chars1[0] in self.vocab_set and "##" + tokenized_chars2[0] in self.vocab_set:
    #         tokenized_chars1_hash = tuple(["##" + tokenized_chars1[0]] + list(tokenized_chars1[1:]))
    #         tokenized_chars2_hash = tuple(["##" + tokenized_chars2[0]] + list(tokenized_chars2[1:]))
    #         error_table[tokenized_chars1_hash].append(tokenized_chars2_hash)

    # def load_table_from_file(self, file_object):
    #     csv_reader = csv.reader(file_object, delimiter='\t', quotechar=None)
    #     error_rows = list(csv_reader)
    #     error_table = defaultdict(list)
    #     for possible_mistake in error_rows:
    #         for chars1 in possible_mistake:
    #             for chars2 in possible_mistake:
    #                 if chars1 == chars2:
    #                     continue
    #                 self.add_to_error_table(chars1, chars2, error_table)
    #     self.error_table = error_table

    
    def load_table_from_file(self, file_object):
        csv_reader = csv.reader(file_object, delimiter='\t', quotechar=None)
        rows = list(csv_reader)
        error_rows_unordered = [row[1:] for row in rows if row[0] == "0"]
        error_rows_ordered = [row[1:] for row in rows if row[0] == "1"]
        error_table = defaultdict(list)
        for possible_mistake in error_rows_unordered:
            for chars1 in possible_mistake:
                for chars2 in possible_mistake:
                    if chars1 == chars2:
                        continue
                    self.add_to_error_table(chars1, chars2, error_table)

        for chars1, chars2 in error_rows_ordered:
            self.add_to_error_table(chars1, chars2, error_table)

        self.error_table = error_table

    def get_error2(self, tokens_list):
        error_table_dict = dict(self.error_table)
        modified_error_table = {}
        for key, value in error_table_dict.items():
            new_key = tuple(c for c in key if c != "[PAD]")
            # if key != new_key:
            #     print("key: ", key)
            #     print("new_key: ", new_key)
            #     print("value: ", value)
            #     print("=" * 50)

            modified_error_table[new_key] = (key, value)

        key = tuple(tokens_list)
        if key in modified_error_table:
            return modified_error_table[key][0], random.choice(modified_error_table[key][1])
        return None

    
    def get_error(self, tokens_list):
        error_table_dict = dict(self.error_table)
        key = tuple(tokens_list)
        if key in error_table_dict:
            return random.choice(error_table_dict[key])
        return None

class MakeTextOld:
    def __init__(self, _tokenizer):
        self.tokenizer = _tokenizer
        self.tokens = None
        self.old_tokens = None
        self.correction_to_old_tokens = None
        self.change_table = None

    def load_change_table_from_file(self, file_object):
        csv_reader = csv.reader(file_object, delimiter='\t', quotechar=None)
        rows = list(csv_reader)
        rows_anywhere = [row[1:] for row in rows if row[0] == "0"]
        rows_middle_or_end = [row[1:] for row in rows if row[0] == "1"]
        rows_beginning = [row[1:] for row in rows if row[0] == "2"]

        change_table = defaultdict(list)
        tok = ManipulateTokens(self.tokenizer)
        for row in rows_anywhere:
            for original_tokens, old_tokens, correction_tokens in zip(*[tok.tokenize_all_versions(row[i]) for i in range(3)]):
                 change_table[tuple(original_tokens)].append((old_tokens, correction_tokens, row[3]))
        for row in rows_middle_or_end:
            for original_tokens, old_tokens, correction_tokens in zip(*[tok.tokenize_all_versions_with_starting_hash(row[i]) for i in range(3)]):
                 change_table[tuple(original_tokens)].append((old_tokens, correction_tokens, row[3]))
        for row in rows_beginning:
            for original_tokens, old_tokens, correction_tokens in zip(*[tok.tokenize_all_versions_without_starting_hash(row[i]) for i in range(3)]):
                 change_table[tuple(original_tokens)].append((old_tokens, correction_tokens, row[3]))
        self.change_table = change_table

    def get_old_version(self, tokens_list, table):
        change_table_dict = dict(table)
        key = tuple(tokens_list)
        if key in change_table_dict:
            return [item[0] for item in change_table_dict[key]]
        return None

    def get_corrected_version(self, tokens_list, table):
        change_table_dict = dict(table)
        key = tuple(tokens_list)
        if key in change_table_dict:
            return [item[1] for item in change_table_dict[key]]
        return None

    def get_frequency(self, tokens_list):
        change_table_dict = dict(self.change_table)
        key = tuple(tokens_list)
        if key in change_table_dict:
            return [item[2] for item in change_table_dict[key]]
        return 0     

    def make_tokens_old(self, tokens):
        self._old_tokens = []
        self._correction_to_old_tokens = []
        # print(self.change_table)
        token_idx = 0
        # ezeknél kell csekkolni, hogy a szó végén vannak-e
        whole_words = ["a", "é##s", "i##s", "##b##b", "v##o##l##t", "##k##é##n##t", "##b##e##l##i"] # utánuk új szó kezdődik
        prefixes = ["e##l", "m##e##g", "f##e##l", "b##e", "l##e##g"] # utánuk folytatódik még a szó
        # print(self.corr_gen.correct_tokenizer(self.tokenizer.tokenize("T[PAD][PAD]TY")))
        while token_idx < len(tokens):
            # make old
            whole_word_found = False
            prefix_found = False
            for i in range(4):
                tokens_seq = tokens[token_idx:token_idx + i + 1]
                if "".join(tokens_seq).lower() in whole_words and token_idx + i + 1 < len(tokens) and len(tokens[token_idx + i + 1]) == 1 and \
                    random.random() < float(self.get_frequency(tokens_seq)[0]):
                    whole_word_found = True
                    self._old_tokens += self.get_old_version(tokens_seq, self.change_table)[0]
                    self._correction_to_old_tokens += self.get_corrected_version(tokens_seq, self.change_table)[0]
                    token_idx += i + 1
                    break
            if whole_word_found:
                continue
            for i in range(1, 3):
                tokens_seq = tokens[token_idx:token_idx + i + 1]
                if "".join(tokens_seq).lower() in prefixes and token_idx + i + 1 < len(tokens) and len(tokens[token_idx + i + 1]) == 3 and \
                    random.random() < float(self.get_frequency(tokens_seq)[0]):
                    prefix_found = True
                    self._old_tokens += self.get_old_version(tokens_seq, self.change_table)[0]
                    self._correction_to_old_tokens += self.get_corrected_version(tokens_seq, self.change_table)[0]
                    token_idx += i + 1
                    break
            if prefix_found:
                continue
            in_table = []
            for i in reversed(range(4)):
                tokens_seq = tokens[token_idx:token_idx + i + 1]
                if self.get_old_version(tokens_seq, self.change_table):
                    in_table.append(i)
                    frequency_smaller = float(self.get_frequency(tokens_seq)[0])
                    break
            random_num = random.random()
            if in_table and  random_num <  frequency_smaller and "".join(tokens[token_idx:token_idx + in_table[0] + 1]).lower() not in whole_words + prefixes:
                i = in_table[0]
                tokens_seq = tokens[token_idx:token_idx + i + 1]
                self._old_tokens += self.get_old_version(tokens_seq, self.change_table)[0]
                self._correction_to_old_tokens += self.get_corrected_version(tokens_seq, self.change_table)[0]
                token_idx += i + 1
                continue
            if in_table and len(self.get_old_version(tokens[token_idx:token_idx + in_table[0] + 1], self.change_table)) > 1 and \
                random_num < frequency_smaller + float(self.get_frequency(tokens[token_idx:token_idx + i + 1])[1]) and \
                    "".join(tokens[token_idx:token_idx + in_table[0] + 1]).lower() not in whole_words + prefixes:
                i = in_table[0]
                tokens_seq = tokens[token_idx:token_idx + i + 1]
                self._old_tokens += self.get_old_version(tokens_seq, self.change_table)[1]
                self._correction_to_old_tokens += self.get_corrected_version(tokens_seq, self.change_table)[1]
                token_idx += i + 1
                continue

            self._old_tokens.append(tokens[token_idx])
            self._correction_to_old_tokens.append(tokens[token_idx])
            token_idx += 1
        return (self._correction_to_old_tokens, self._old_tokens)

class CorrectionDatasetGenerator:
    error_frequency = 0.15
    sparse_frequency = 0.2
    dense_frequency = 0.2
    old_frequency = 0.05
    common_extra_chars = "{}jli;|\\/(:)!1.t'"
    hyphens = "\xad-"

    def __init__(self, _tokenizer, _ocr_errors_generator, _seq_length, _old_text_generator):
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
        self.old_gen = _old_text_generator
        self._tokens = None
        self._old_tokens = None
        self._error_tokens = None
        self._correct_tokens = None

    
    def replace_with_random_token(self, token_idx):
        self._error_tokens.append(random.choice(self.vocab))
        self._correct_tokens.append([self._tokens[token_idx]])

    def delete_token(self, token_idx):
        self._correct_tokens[-1].append(self._tokens[token_idx])

    def add_extra_token(self, token_idx, old_tokens):
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
            next_token = old_tokens[token_idx]
            if next_token.startswith("##") and random.random() < 0.8:
                next_token = next_token[2:]
            self._error_tokens.append(next_token)
            self._correct_tokens.append([self._tokens[token_idx]])
            return token_idx + 1

    def add_space(self, token_idx):
        self._error_tokens.append("##" + self._old_tokens[token_idx])
        self._correct_tokens.append([self._tokens[token_idx]])

    def remove_space(self, token_idx):
        self._error_tokens.append(self._old_tokens[token_idx][2:])
        self._correct_tokens.append([self._tokens[token_idx]])

    def make_ocr_typo(self, token_idx, old_tokens, tokens):
        in_table = []
        for i in range(3):
            if self.corr_gen.get_error2(old_tokens[token_idx:token_idx + i + 1]):
                in_table.append(i)
        if in_table:
            # join_tokens = random.choice(in_table)
            join_tokens = in_table[-1]
            # a régiesített tokenek szelete, ehhez keresünk a táblázatból typot
            slice_old_tokens = old_tokens[token_idx:token_idx + join_tokens + 1]
            # a korrekció majd ez lesz
            slice_correct_tokens = tokens[token_idx:token_idx + join_tokens + 1]
            # az old tokenekhez keresünk typot
            real_correct_tokens, slice_error_tokens = self.corr_gen.get_error2(slice_old_tokens)
            if slice_old_tokens == slice_correct_tokens:
                self._correct_tokens.extend(self.corr_gen.create_correction(real_correct_tokens, slice_error_tokens))
            else:
                self._correct_tokens.extend(self.corr_gen.create_correction(slice_correct_tokens, slice_error_tokens))
            self._error_tokens.extend(slice_error_tokens)
            return token_idx + join_tokens + 1
        else:
            self._error_tokens.append(random.choice(self.vocab))
            self._correct_tokens.append([old_tokens[token_idx]])
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

    def reset_space_after_punctuation(self):
        for token_idx in range(1, len(self._error_tokens)):
            if self._error_tokens[token_idx].startswith("##") and len(self._error_tokens[token_idx - 1]) == 1 and _is_punctuation(
                    self._error_tokens[token_idx - 1]):
                self._error_tokens[token_idx] = self._error_tokens[token_idx][2:]

    def pad_to_length_3(self):
        for correct_token in self._correct_tokens:
            correct_token += [self.tokenizer.pad_token] * (3 - len(correct_token))

    def run(self, tokens, age):
        self._tokens = tokens
        self._error_tokens = []
        self._correct_tokens = []
        # print(self.tokenizer.tokenize("s[MASK]sz"))
        if age == "old":
            self._tokens, self._old_tokens = self.old_gen.make_tokens_old(self._tokens)
        else:
            self._old_tokens = self._tokens
        token_idx = 0
        while token_idx < len(self._old_tokens):
            if random.random() < self.error_frequency: # random.random() in [0, 1)
                if random.random() < 0.1:
                    self.replace_with_random_token(token_idx)
                    token_idx += 1    
                elif random.random() < 0.05 and self._correct_tokens and len(self._correct_tokens[-1]) < 3:
                    self.delete_token(token_idx)
                    token_idx += 1
                elif random.random() < 0.05:
                    token_idx = self.add_extra_token(token_idx, self._old_tokens)
                elif random.random() < 0.1 and "##" + self._old_tokens[token_idx] in self.vocab_set:
                    self.add_space(token_idx)
                    token_idx += 1
                elif random.random() < 0.1 and self._old_tokens[token_idx].startswith("##") and \
                        self._old_tokens[token_idx][2:] in self.vocab_set:
                    self.remove_space(token_idx)
                    token_idx += 1
                else:
                    token_idx = self.make_ocr_typo(token_idx, self._old_tokens, self._tokens)
            else:
                self._error_tokens.append(self._old_tokens[token_idx])
                self._correct_tokens.append([self._tokens[token_idx]])
                token_idx += 1
        self.make_sparse()
        self.make_dense()
        self.reset_space_after_punctuation()
        self.pad_to_length_3() 
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


    def generate_dataset(self, dataset_dir, thread, seed_input):
        tok = ManipulateTokens(self.tokenizer)
        NUM_OF_THREADS = 16
        input_files = sorted(tf.io.gfile.glob(dataset_dir + "/*"))
        # input_files = sorted(tf.io.gfile.glob(dataset_dir + "/AA/wiki_00_test"))
        # input_files = sorted(tf.io.gfile.glob(dataset_dir + "/AA/wiki_*"))
        # random.seed(42)
        #random.shuffle(input_files)
        # input_files = tf.io.gfile.glob(dataset_dir + "/*")

        thread_input_lengths = [1] * 11 + [2] * 5
        thread_start = (thread - 1) * thread_input_lengths[thread - 1]
        thread_input_files = input_files[thread_start:thread_start + thread_input_lengths[thread - 1]]

        print(thread_input_files)

        # thread_input_length = len(input_files) // NUM_OF_THREADS
        # if (len(input_files) % NUM_OF_THREADS):
        #     thread_input_length += 1 
        # thread_start = (thread - 1) * thread_input_length
        # thread_input_files = input_files[thread_start:thread_start + thread_input_length]
        # print(thread_input_files)
        # open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", 'w').close()
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
                            if random.random() < self.old_frequency:
                                age = "old"
                            else:
                                age = "new"
                            all_modified_tokens, all_corrected_tokens = self.run(tokens, age)
                            input_len = len(all_modified_tokens)
                            for start_index in range(0, input_len, self.seq_length - 2):
                                modified_tokens = all_modified_tokens[start_index:start_index + self.seq_length - 2]
                                corrected_tokens = all_corrected_tokens[start_index:start_index + self.seq_length - 2]
                                inputs = self.create_input(modified_tokens)
                                labels = self.create_label(corrected_tokens)
                                # with open("/home/mcsilla/machine_learning/gitrepos/err-corr/test_output.txt", "a") as f:
                                #     standard_out = sys.stdout
                                #     sys.stdout = f

                                #     corrected_tokens_without_PAD = []
                                #     for item in corrected_tokens:
                                #         corrected_tokens_without_PAD.append([])
                                #         for corr in item:
                                #             if corr != "[PAD]":
                                #                 corrected_tokens_without_PAD[-1].append(corr)
                                #             else:
                                #                 corrected_tokens_without_PAD[-1].append("")
                                #     error_text_list = []
                                #     correct_text_list = []
                                #     for err_tok, corr_tok in zip(modified_tokens, corrected_tokens):
                                #         if corr_tok[0] == '[PAD]':
                                #             error_text_list.append(err_tok[-1])
                                #             correct_text_list.append(" ")
                                #         elif corr_tok[1] == '[PAD]':
                                #             error_text_list.append(err_tok[-1])
                                #             correct_text_list.append(corr_tok[0][-1])
                                #         elif corr_tok[2] == '[PAD]':
                                #             error_text_list.extend([err_tok[-1], " "])
                                #             correct_text_list.extend([corr_tok[0][-1], corr_tok[1][-1]])
                                #         else:
                                #             error_text_list.extend([err_tok[-1], " ", " "])
                                #             correct_text_list.extend([corr_tok[0][-1], corr_tok[1][-1], corr_tok[2][-1]])

                                #     print(" ".join(error_text_list))
                                #     print("-" * 200)
                                #     print(" ".join(correct_text_list))
                                #     print("=" * 200)

                                #     # print(self.corr_gen.error_table)
                                #     # print("".join([tok.detokenize_char(token) for token in modified_tokens]))
                                #     # print(tok.restore_text_from_corrected_tokens(corrected_tokens))
                                #     # print(inputs["input_ids"], "\n", inputs["attention_mask"], "\n", inputs["token_type_ids"], "\n\n",\
                                #     # labels["label_0"], "\n", labels["label_1"], "\n", labels["label_2"], "\n\n")
                                #     # print(modified_tokens)
                                #     # print(corrected_tokens)
                                #     sys.stdout = standard_out
                                yield (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]), \
                                    (labels["label_0"], labels["label_1"], labels["label_2"])
                    else:
                        document_lines.append(line)


def int64feature(int_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def printable_format(tokenizer, token_ids):
    tok = ManipulateTokens(tokenizer)
    tokens = tokenizer.convert_ids_to_tokens([token_id for token_id in token_ids if token_id >= 0])
    return "".join([tok.detokenize_char(token) for token in tokens])


def write_examples_to_tfrecord(examples, tf_records_writer):
    random.shuffle(examples)
    for instance in examples:
        tf_records_writer.write(instance.SerializeToString())

