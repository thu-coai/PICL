import multiprocessing
import os
import time
import torch
import sys
import h5py
import numpy as np
from tqdm import tqdm
from icl_train.data_utils.indexed_dataset import make_builder
from icl_train.data_utils.distributed_indexed import DistributedMMapIndexedDataset
from transformers import GPT2Tokenizer
from model_center.arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args
        
    def initializer(self):
        Encoder.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_config)

    def encode(self, line):
        oid, line = line
        line = line.replace("<@x(x!>", "\n")
        if self.args.replace_return_with_space:
            line = line.replace("\n", " ").strip()
        token_ids = Encoder.tokenizer.encode(line)
        
        return oid, token_ids, len(line)


def get_np_map(map_ids, max_num):
    o2n = np.zeros(max_num, dtype=np.int32)
    n2o = np.zeros(max_num, dtype=np.int32)
    for oid, nid in map_ids:
        o2n[oid] = nid
        n2o[nid] = oid
    return o2n, n2o   


def process_data(args):    
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
        encoded_docs = pool.imap_unordered(encoder.encode, enumerate(fin), chunksize=10)
        proc_start = time.time()
        total_bytes_processed = 0

        # 3. tool `indexed_dataset` compress the tokenized data into binary format `bin_file`
        # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
        bin_file = os.path.join(args.processed_unsup_data_path, f"icl_{ith}.bin")
        idx_file = os.path.join(args.processed_unsup_data_path, f"icl_{ith}.idx")
        
        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

        # put tokenized data into binary_builder
        nid = 0
        o_docs_num = 0
        map_ids = []
        for lid, (oid, input_ids, bytes_processed) in enumerate(encoded_docs):
            o_docs_num += 1
            total_bytes_processed += bytes_processed
            if input_ids is None:
                continue
            
            binary_builder.add_item(torch.IntTensor(input_ids))
            map_ids.append((oid, nid))
            
            nid += 1
            if lid % 10000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"{ith} Processed {lid} documents. {nid} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)
        o2n, n2o = get_np_map(map_ids, o_docs_num)
        with h5py.File(os.path.join(args.processed_unsup_data_path, f"map.h5"), "w") as h5_f:
            h5_f.create_dataset("map_o2n", data=o2n, dtype=np.int32, chunks=True)
            h5_f.create_dataset("map_n2o", data=n2o, dtype=np.int32, chunks=True)
        
        # close multiproceessing mapping
        pool.close()


def get_train_valid(args):
    
    train_bin_file = os.path.join(args.processed_unsup_data_path, "train_icl_0.bin")
    train_idx_file = os.path.join(args.processed_unsup_data_path, "train_icl_0.idx")

    valid_bin_file = os.path.join(args.processed_unsup_data_path, "valid_icl_0.bin")
    valid_idx_file = os.path.join(args.processed_unsup_data_path, "valid_icl_0.idx")

    train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.int32)
    valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.int32)
    
    search_res = DistributedMMapIndexedDataset(args.unsup_data_path, "search_icl", 0, 1)
    
    for i in tqdm(range(len(search_res)), desc="Spliting train/valid"):
        data = search_res[i].astype(int).tolist()
        if i < args.unsup_valid_num:
            valid_binary_builder.add_item(torch.IntTensor(data))
        else:
            train_binary_builder.add_item(torch.IntTensor(data))
    
    train_binary_builder.finalize(train_idx_file)
    valid_binary_builder.finalize(valid_idx_file)


def main():
    # assumes that there are 100 raw data files, named `data_1.txt` to `data_100.txt`
    args = get_args()
    
    ex_prefix = ""
    ex_prefix += ("r2s" if args.replace_return_with_space else "rr")
        
    args.processed_unsup_data_path = os.path.join(args.processed_unsup_data_path, ex_prefix)

    os.makedirs(args.processed_unsup_data_path, exist_ok=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    
    # process_data(args)
    get_train_valid(args)


if __name__ == '__main__':
    main()