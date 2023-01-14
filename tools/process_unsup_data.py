import multiprocessing
import os
import time
import torch
import json
import sys
import random
import numpy as np
from icl_train.data_utils.indexed_dataset import make_builder
from transformers import GPT2Tokenizer
from model_center.arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args
        self.max_length_per_sample = args.max_length_per_sample
        self.max_length = args.max_length if args.max_length is not None else args.max_length_all_demos
        
        self.delimiter_id = {
            "<n>": 198,
            "<eos>": 50256,
            "2<n>": 628
        }[self.args.end_token]

    def initializer(self):
        Encoder.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_config)

    def encode(self, line):
        data = json.loads(line)
        q = data[0].replace("<@x(x!>", "\n")
        if self.args.replace_return_with_space:
            q = q.replace("\n", " ").strip()
        r = data[1]
        r = [rr.replace("<@x(x!>", "\n").strip() for rr in r]
        if self.args.replace_return_with_space:
            r = [rr.replace("\n", " ").strip() for rr in r]
        
        q_id = Encoder.tokenizer.encode(q)[:self.max_length_per_sample-1] + [self.delimiter_id]
        r_ids = [Encoder.tokenizer.encode(rr)[:self.max_length_per_sample-1] + [self.delimiter_id] for rr in r]
        
        q_id = q_id[:self.max_length]
        while len(q_id) + sum(map(len, r_ids)) >= self.max_length and len(r_ids) > 0:
            r_ids.pop()
        
        input_ids = r_ids + [q_id]
        input_ids = [x for y in input_ids for x in y]
        
        return input_ids, len(line)


def main():
    print("OK")
    # assumes that there are 100 raw data files, named `data_1.txt` to `data_100.txt`
    args = get_args()
    
    ex_prefix = ""
    ex_prefix += ("r2s" if args.replace_return_with_space else "rr")
    ex_prefix += (args.end_token)
        
    args.processed_unsup_data_path = os.path.join(args.processed_unsup_data_path, ex_prefix)

    os.makedirs(args.processed_unsup_data_path, exist_ok=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    
    for ith in range(0, 100):
        file_name = os.path.join(args.unsup_data_path, "{}_{}".format(args.unsup_data_name, ith))
        if not os.path.exists(file_name):
            print(file_name, "does not exist, stop.")
            break
        fin = open(file_name, "r", encoding="utf-8")
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, fin, chunksize=10)
        proc_start = time.time()
        total_bytes_processed = 0

        # 3. tool `indexed_dataset` compress the tokenized data into binary format `bin_file`
        # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
        train_bin_file = os.path.join(args.processed_unsup_data_path, f"train_tokenized_{ith}.bin")
        train_idx_file = os.path.join(args.processed_unsup_data_path, f"train_tokenized_{ith}.idx")
    
        valid_bin_file = os.path.join(args.processed_unsup_data_path, f"valid_tokenized_{ith}.bin")
        valid_idx_file = os.path.join(args.processed_unsup_data_path, f"valid_tokenized_{ith}.idx")
    
        train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.uint16)
        valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.uint16)

        # put tokenized data into binary_builder
        for lid, (input_ids, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if input_ids is None:
                continue
            
            if lid < args.unsup_valid_num:
                valid_binary_builder.add_item(torch.IntTensor(input_ids))
            else:
                train_binary_builder.add_item(torch.IntTensor(input_ids))
            
            if lid % 10000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"{ith} Processed {lid} documents",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        train_binary_builder.finalize(train_idx_file)
        valid_binary_builder.finalize(valid_idx_file)

        # close multiproceessing mapping
        pool.close()


if __name__ == '__main__':
    main()