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
        
    def initializer(self):
        Encoder.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_config)

    def encode(self, line):
        line = line.replace("<@x(x!>", "\n")
        token_ids = Encoder.tokenizer.encode(line) + [Encoder.tokenizer.eos_token_id]
        
        return token_ids, len(line)


def split_samples(ids, tokenizer):
    buff = []
    all_ids = []
    for x in ids:
        buff.append(x)
        if x == tokenizer.eos_token_id:
            all_ids.append(buff)
    if len(buff) != 0:
        all_ids.append(buff)

    return all_ids


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
    
    file_name = os.path.join(args.unsup_data_path, "{}".format(args.unsup_data_name))

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
    train_bin_file = os.path.join(args.processed_unsup_data_path, f"train_lm_0.bin")
    train_idx_file = os.path.join(args.processed_unsup_data_path, f"train_lm_0.idx")

    train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.uint16)

    # put tokenized data into binary_builder
    buffer = []
    inst_num = 0
    for lid, (input_ids, bytes_processed) in enumerate(encoded_docs):
        total_bytes_processed += bytes_processed
        if input_ids is None:
            continue
        
        if args.stuffed:
            buffer.extend(input_ids)
            if len(buffer) >= args.max_length:
                inst = buffer[:args.max_length]
                buffer = buffer[args.max_length:]
            
                if args.split_stuffed:
                    splited_insts = split_samples(inst, tokenizer)
                    for x in splited_insts:
                        train_binary_builder.add_item(torch.IntTensor(x))
                        inst_num += 1
                else:
                    train_binary_builder.add_item(torch.IntTensor(inst))
                    inst_num += 1
        else:
            if args.no_stuffed_no_waste:
                while len(input_ids) > args.max_length:
                    train_binary_builder.add_item(torch.IntTensor(input_ids[:args.max_length]))
                    input_ids = input_ids[:args.max_length]
                    inst_num += 1
                if len(input_ids) != 0:
                    train_binary_builder.add_item(torch.IntTensor(input_ids))
                    inst_num += 1
            else:
                train_binary_builder.add_item(torch.IntTensor(input_ids[:args.max_length]))
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

    # close multiproceessing mapping
    pool.close()


if __name__ == '__main__':
    main()