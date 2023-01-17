import os
import torch
import numpy as np
from tqdm import tqdm

from data_utils.indexed_dataset import make_builder
from data_utils.distributed_indexed import DistributedMMapIndexedDataset

import argparse
from arguments import add_model_config_args, add_hp_args, add_runtime_args, add_data_args, add_pretraining_args, add_icl_args


def get_train_valid(args):
    
    train_bin_file = os.path.join(args.processed_output, "train_icl_0.bin")
    train_idx_file = os.path.join(args.processed_output, "train_icl_0.idx")

    valid_bin_file = os.path.join(args.processed_output, "valid_icl_0.bin")
    valid_idx_file = os.path.join(args.processed_output, "valid_icl_0.idx")

    train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.int32)
    valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.int32)
    
    filtered_res = DistributedMMapIndexedDataset(args.picl_idx_data_dir, args.picl_data_name, 0, 1)
    
    for i in tqdm(range(len(filtered_res)), desc="Spliting train/valid"):
        data = filtered_res[i].astype(int).tolist()
        if i < args.picl_valid_num:
            valid_binary_builder.add_item(torch.IntTensor(data))
        else:
            train_binary_builder.add_item(torch.IntTensor(data))
    
    train_binary_builder.finalize(train_idx_file)
    valid_binary_builder.finalize(valid_idx_file)


def main():
    parser = argparse.ArgumentParser()
    parser = add_icl_args(add_model_config_args(add_hp_args(add_runtime_args(add_data_args(add_pretraining_args(parser))))))
    args = parser.parse_args()

    args.processed_output = os.path.join(args.processed_output, args.picl_data_prefix.replace("/", "_"))
    os.makedirs(args.processed_output, exist_ok=True)
    
    get_train_valid(args)


if __name__ == "__main__":
    main()