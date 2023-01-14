from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from transformers import GPT2Tokenizer
import sys
import random
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


def check():
    base_path = "/home/lidong1/CodeRepo/icl_train/unsup_data/lm_data/1024/no_stuffed/rr<n>/"

    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")

    ctx = DistributedMMapIndexedDataset(base_path, f"valid_lm", 0, 1)

    while True:
        idx = int(input("Idx: "))

        d = ctx[idx].astype(int)
        
        text = tokenizer.decode(d)

        print(text)
        
        # all_ids = text.strip().split("\n")
        
        # test_sample = all_ids[-1]

        # demos = all_ids[:-1]
        
        # print("##### Test Sample #####")
        # print(test_sample.replace("<@x(x!>", "\n"))
        # print("#" * 50)
        # print()

        # for i, demo in enumerate(demos):
        #     print(f"##### DEMO #{i}: #####")
        #     print(demo.replace("<@x(x!>", "\n"))
        #     print()

        # print("#" * 50)
    

def len_dist():
    base_path = "/home/lidong1/CodeRepo/icl_train/unsup_data/lm_data/1024/no_stuffed/rr<n>/"

    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")

    ctx = DistributedMMapIndexedDataset(base_path, f"valid_lm", 0, 1)
    
    all_length = []
    
    for i in tqdm(range(len(ctx))):
        all_length.append(len(ctx[i]))
        
    plt.hist(all_length, 100)
    
    plt.savefig("len_dist.png")


def check_map():
    
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/mnt/yuxian/unsup_data/general/full_256/SEARCH1/r2s"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"valid_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    index = 9000

    icl_indices = icl_idx[index].astype(int)
    
    print(icl_indices)
    
    # idx after unordered preprocess may be in random order icl_idx_map: from origin idx to new idx
    data = [icl_ctx[int(icl_idx_map[i])].tolist() for i in icl_indices]
        
    q_id = data[0][:256-1] + [198]
    r_ids = [rr[:256-1] + [198] for rr in data[1:]]

    # while len(q_id) + sum(map(len, r_ids)) >= 1024 and len(r_ids) > 0:
    #     r_ids.pop()
        
    # r_ids = r_ids[:16]
    # random.shuffle(r_ids)
    
    print(tokenizer.decode(q_id))
    print("#" * 10)
    for rr in r_ids:
        print(tokenizer.decode(rr))
        print()


def check_h5_o2n():
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/10K_256/SEARCH1/r2s"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
    
    index = 9900
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]
        
    with open("/home/lidong1/dpr-simple/results/general/10K_256/SEARCH1/raw_0") as f:
        lines = f.readlines()
        
    print(lines[index])
    print()
    print(tokenizer.decode(icl_ctx[int(icl_idx_map[index])]))


def check_h5_n2o():
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/10K_256/SEARCH1/r2s"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
    
    index = 9900
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]
        
    with open("/home/lidong1/dpr-simple/results/general/10K_256/SEARCH1/raw_0") as f:
        lines = f.readlines()
        
    print(lines[index])
    print()
    print(tokenizer.decode(icl_ctx[int(icl_idx_map[index])]))


def main():
    # check()
    # len_dist()
    check_map()
    # check_h5_o2n()
    # with open("/home/lidong1/dpr-simple/results/general/full_256/SEARCH1/raw_0") as f:
    #     for i, line in enumerate(f):
    #         if i == 936:
    #             print(line)
    #             print()
            
    #         if i == 1101186:
    #             print(line)
            
    #         if i > 1101188:
    #             break
    # ctx = DistributedMMapIndexedDataset("/home/lidong1/dpr-simple/results/general/full_256/SEARCH1", "search_icl", 0, 1)
    
    # print(ctx[100000])

if __name__ == "__main__":
    main()