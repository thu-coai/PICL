import multiprocessing
import os
import time
import torch
import json
import sys
import random
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
import argparse
from arguments import add_model_config_args, add_hp_args, add_runtime_args, add_data_args, add_pretraining_args, add_icl_args


class Encoder(object): 
    def __init__(self, args):
        self.args = args
        
    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_dir)

    def encode(self, line):
        line = line.replace("<@x(x!>", "\n")
        token_ids = Encoder.tokenizer.encode(line) + [Encoder.tokenizer.eos_token_id]
        
        return token_ids, len(line)


def process_data(args):
    file_name = os.path.join(args.picl_data_dir, "{}".format(args.picl_data_name))
    fin = open(file_name, "r", encoding="utf-8")
    # encoder use the tokenizer to encode data
    encoder = Encoder(args)

    pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, chunksize=10)
    proc_start = time.time()
    total_bytes_processed = 0

    # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
    train_bin_file = os.path.join(args.processed_output, f"train_lm_{args.bin_file_index}.bin")
    train_idx_file = os.path.join(args.processed_output, f"train_lm_{args.bin_file_index}.idx")

    valid_bin_file = os.path.join(args.processed_output, f"valid_lm_{args.bin_file_index}.bin")
    valid_idx_file = os.path.join(args.processed_output, f"valid_lm_{args.bin_file_index}.idx")

    train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.uint16)
    valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.uint16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # put tokenized data into binary_builder
    buffer = []
    inst_num = 0
    for lid, (input_ids, bytes_processed) in enumerate(encoded_docs):
        total_bytes_processed += bytes_processed
        if input_ids is None:
            continue
        
        buffer.extend(input_ids)
        while len(buffer) >= args.max_length:
            inst = buffer[:args.max_length]
            buffer = buffer[args.max_length:]
        
            if inst_num < args.lm_valid_num:
                valid_binary_builder.add_item(torch.IntTensor(inst))
            else:
                train_binary_builder.add_item(torch.IntTensor(inst))
            
            inst_num += 1

        if lid % 10000 == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents. {inst_num} instances.",
                f"({lid/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

    # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
    train_binary_builder.finalize(train_idx_file)
    valid_binary_builder.finalize(valid_idx_file)

    # close multiproceessing mapping
    pool.close()


def main():
    # assumes that there are 100 raw data files, named `data_1.txt` to `data_100.txt`
    parser = argparse.ArgumentParser()
    parser = add_icl_args(add_model_config_args(add_hp_args(add_runtime_args(add_data_args(add_pretraining_args(parser))))))
    args = parser.parse_args()

    os.makedirs(args.processed_output, exist_ok=True)
        
    process_data(args)


if __name__ == '__main__':
    main()