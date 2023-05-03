import multiprocessing
import os
import time
import torch
import sys
import h5py
import numpy as np
from tqdm import tqdm
from data_utils.indexed_dataset import make_builder
from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from transformers import AutoTokenizer

import argparse
from arguments import add_model_config_args, add_hp_args, add_runtime_args, add_data_args, add_pretraining_args, add_icl_args


class Encoder(object): 
    def __init__(self, args):
        self.args = args
        
    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_dir)

    def encode(self, line):
        oid, line = line
        line = line.replace("<@x(x!>", "\n")
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
    file_name = os.path.join(args.picl_data_dir, "{}".format(args.picl_data_name))
    fin = open(file_name, "r", encoding="utf-8")
    # encoder use the tokenizer to encode data
    encoder = Encoder(args)

    pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, enumerate(fin), chunksize=10)
    proc_start = time.time()
    total_bytes_processed = 0

    # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
    bin_file = os.path.join(args.processed_output, f"picl_0.bin")
    idx_file = os.path.join(args.processed_output, f"picl_0.idx")
    
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
            print(f"Processed {lid} documents. {nid} instances.",
                f"({lid/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

    # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
    binary_builder.finalize(idx_file)
    o2n, n2o = get_np_map(map_ids, o_docs_num)
    with h5py.File(os.path.join(args.processed_output, f"map.h5"), "w") as h5_f:
        h5_f.create_dataset("map_o2n", data=o2n, dtype=np.int32, chunks=True)
        h5_f.create_dataset("map_n2o", data=n2o, dtype=np.int32, chunks=True)
    
    # close multiproceessing mapping
    pool.close()


def main():
    # assumes that there are 100 raw data files, named `data_1.txt` to `data_100.txt
    parser = argparse.ArgumentParser()
    parser = add_icl_args(add_model_config_args(add_hp_args(add_runtime_args(add_data_args(add_pretraining_args(parser))))))
    args = parser.parse_args()

    os.makedirs(args.processed_output, exist_ok=True)
    
    process_data(args)


if __name__ == '__main__':
    main()