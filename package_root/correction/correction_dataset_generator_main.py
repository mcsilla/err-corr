import argparse
from collections import OrderedDict
import logging
import time

import tensorflow as tf
from transformers import BertTokenizerFast

from correction.correction_dataset_generator import CorrectionDatasetGenerator, generate_dataset, write_examples_to_tfrecord, int64feature

def main():
    parser = argparse.ArgumentParser() # creating an ArgumentParser object

    # input data and model directories
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    # parser.add_argument('--tokenizer_config', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, required=True)
    parser.add_argument('--dupe_factor', type=int, default=1)

    args, _ = parser.parse_known_args()

    SEQ_LENGTH = args.sequence_length
    # character_tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_config) # why is from_pretrained needed? 
    character_tokenizer = BertTokenizerFast("data/tokenizer/alphabet", do_lower_case=False) 
    dataset_generator = CorrectionDatasetGenerator(character_tokenizer)
    writer = tf.io.TFRecordWriter(args.output_file, options="GZIP")
    logging.basicConfig(level=logging.INFO)
    inst_idx = 0
    start_time = time.time()

    for repeat in range(args.dupe_factor):
        example_cache = []
        for inputs, outputs in generate_dataset(character_tokenizer, dataset_generator, args.input_file):
            inst_idx += 1
            feature = OrderedDict()
            feature['input_ids'] = int64feature(inputs[0])
            feature['attention_mask'] = int64feature(inputs[1])
            feature['token_type_ids'] = int64feature(inputs[2])
            feature['output1'] = int64feature(outputs[0])
            feature['output2'] = int64feature(outputs[1])
            feature['output3'] = int64feature(outputs[2])
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            example_cache.append(example)

            if not (inst_idx % 10000):
                logging.info(f"*** repeat: {repeat} total_instances: {inst_idx} time: {time.time() - start_time}s ***")

            if len(example_cache) >= 100000:
                write_examples_to_tfrecord(example_cache, writer)
                example_cache = []

            if inst_idx < 20:
                logging.info("*** Example ***")
                logging.info("text_with_errors: " + str(printable_format(character_tokenizer, inputs[0])))
                logging.info(f"attention_mask: {inputs[1]}")
                logging.info(f"token_type_ids: {inputs[2]}")
                logging.info("text_corrected1: " + str(printable_format(character_tokenizer, outputs[0])))
                logging.info("text_corrected2: " + str(printable_format(character_tokenizer, outputs[1])))
                logging.info("text_corrected3: " + str(printable_format(character_tokenizer, outputs[2])))

        write_examples_to_tfrecord(example_cache, writer)

    writer.close()
    runtime = time.time() - start_time
    logging.info(f"*** {inst_idx} files wrote to file in {runtime} seconds ***")

