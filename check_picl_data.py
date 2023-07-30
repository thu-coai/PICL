from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from transformers import GPT2Tokenizer
import sys
import random
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("checkpoints/gpt2-large")
    
    path_icl_ctx = "pretrain_data/80M_128/gpt2"
    path_icl_idx = "pretrain_data/picl/80M_128_TRAIN_p1_en1_hn4_s42_lr5e-05-bs64-G1_4000.pt_L2_filtered_0.0"

    icl_ctx = DistributedMMapIndexedDataset(path_icl_ctx, "icl", 0, 1) # for new idx
    icl_idx = DistributedMMapIndexedDataset(path_icl_idx, f"train_icl", 0, 1) # for origin idx

    with h5py.File(os.path.join(path_icl_ctx, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    while True:
        index = int(input("Input Paragraph Index >>>"))

        icl_indices = icl_idx[index].astype(int)
        
        data = [icl_ctx[int(icl_idx_map[i])].tolist() for i in icl_indices]
            
        q_id = data[0][:256-1] + [198]
        r_ids = [rr[:256-1] + [198] for rr in data[1:]]
        
        print("#" * 10 + "  Query  " + "#" * 10)
        print(tokenizer.decode(q_id))
        for i, rr in enumerate(r_ids[1:17]):
            print("#" * 10 + f"  Retrieved Paragraph #{i+1}  " + "#" * 10)
            print(tokenizer.decode(rr))
            print()


if __name__ == "__main__":
    main()