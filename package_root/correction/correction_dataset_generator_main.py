import argparse
from collections import OrderedDict
import logging
import time
import json
from pathlib import Path
import tqdm

import tensorflow as tf
from transformers import BertTokenizerFast

import correction.correction_dataset_generator
from correction.correction_dataset_generator import MakeTextOld, ErrorTable, CorrectionDatasetGenerator, write_examples_to_tfrecord, int64feature, printable_format

def main():
    parser = argparse.ArgumentParser() # creating an ArgumentParser object

    # input data and model directories
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=int, required=True)
    parser.add_argument('--thread', type=int, required=True)

    args, _ = parser.parse_known_args()
    with open(args.config) as f:
        # print(json.load(f)["records"])
        config = json.load(f)
        files_paths = config["paths"]
        record_params = config["records"]
        seed_input = config["seed"]

    character_tokenizer = BertTokenizerFast(Path(files_paths["vocab_file"]), do_lower_case=False) 
    ocr_errors_generator = ErrorTable(character_tokenizer)
    old_text_generator = MakeTextOld(character_tokenizer)
    with open(Path(files_paths["ocr_errors"]), encoding="utf-8") as f:
        ocr_errors_generator.load_table_from_file(f)
    with open(Path(files_paths["old_table"]), encoding="utf-8") as f:
        old_text_generator.load_change_table_from_file(f)
    dataset_generator = CorrectionDatasetGenerator(character_tokenizer, ocr_errors_generator, record_params["sequence_length"], old_text_generator)
    output_dir = Path(files_paths["output_file_root"])
    tf.io.gfile.makedirs(str(output_dir))
    writer = tf.io.TFRecordWriter(str(output_dir / f"tf_record_{args.output}"), options="GZIP")
    logging.basicConfig(level=logging.INFO)
    inst_idx = 0
    start_time = time.time()

    for repeat in range(record_params["dupe_factor"]):
        example_cache = []
        for inputs, outputs in dataset_generator.generate_dataset(files_paths["dataset_dir"], args.thread, seed_input):
        # for inputs, outputs in tqdm.tqdm(dataset_generator.generate_dataset(files_paths["test_input_dir"])):
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
    #             # logging.info(f"attention_mask: {inputs[1]}")
    #             # logging.info(f"token_type_ids: {inputs[2]}")
    #             logging.info("text_corrected1: " + str(printable_format(character_tokenizer, outputs[0])))
    #             # logging.info("text_corrected2: " + str(printable_format(character_tokenizer, outputs[1])))
    #             # logging.info("text_corrected3: " + str(printable_format(character_tokenizer, outputs[2])))
    #     write_examples_to_tfrecord(example_cache, writer)

    # writer.close()
    # runtime = time.time() - start_time
    # logging.info(f"*** {inst_idx} files wrote to file in {runtime} seconds ***")

