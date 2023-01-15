from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from data_utils.indexed_dataset import make_builder
from transformers import GPT2Tokenizer
import sys
import torch
import random
import numpy as np
import os
from tqdm import tqdm
import h5py
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


random.seed(42)


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

def get_oid2idx_map():
    
    path_icl = "/home/lidong1/dpr-simple/results/general/full_100/SEARCH1/"

    icl_idx = DistributedMMapIndexedDataset(path_icl, f"search_icl", 0, 1) # for origin idx

    m = [-1 for _ in range(len(icl_idx))]

    for idx in tqdm(range(len(icl_idx))):
        oids = icl_idx[idx].astype(int)
        m[oids[0]] = idx

    with open("oid2idx.pkl", "wb") as f:
        pickle.dump(m, f)
    

def check_map():
    
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/full_256/SROBERTA/r2s/"
    path_data = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/full_256/SROBERTA/r2s/"
    icl_ctx = DistributedMMapIndexedDataset(path_data, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"train_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_data, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    index = 1234

    icl_indices = icl_idx[index].astype(int)
    
    print(icl_indices)
    # print([int(icl_idx_map[i]) for i in icl_indices])
    
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


def check_map_with_oid():
    
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/home/lidong1/dpr-simple/results/general/full_100/SEARCH1/"
    path_data = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/full_100/SEARCH1/r2s/"
    icl_ctx = DistributedMMapIndexedDataset(path_data, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"search_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_data, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    with open(os.path.join("oid2idx.pkl"), "rb") as f:
        oid2idx = pickle.load(f)

    while True:
        oid = int(input(">>>"))
        index = oid2idx[oid]

        print(index)

        icl_indices = icl_idx[index].astype(int)
        
        print(icl_indices)
        # print([int(icl_idx_map[i]) for i in icl_indices])
        
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


def check_map_2():
    
    path_icl = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/10M_256/bm25/r2s"
    path_data = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/10M_256/bm25/r2s"        
    for split in ["train", "valid"]:
        icl_idx = DistributedMMapIndexedDataset(path_icl, f"{split}_icl", 0, 1) # for origin idx

        bin_file = os.path.join(path_icl, f"{split}_icl_fix.bin")
        idx_file = os.path.join(path_icl, f"{split}_icl_fix.idx")

        builder = make_builder(bin_file, impl="mmap", dtype=np.int32)

        n = 0
        for index in tqdm(range(len(icl_idx))):
            icl_indices = icl_idx[index].astype(int)
            
            if len(icl_indices) == 20:
                builder.add_item(torch.LongTensor(icl_indices))
                n += 1
                
        builder.finalize(idx_file)
                
        print(n)


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


def stat_len():
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/mnt/yuxian/unsup_data/general/full_100/SEARCH1/r2s"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"train_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    n = 1000000

    all_samp_len = []
    
    for index in tqdm(range(n)):
        icl_indices = icl_idx[index].astype(int)
        data = [icl_ctx[int(icl_idx_map[i])].tolist() for i in icl_indices]
        all_samp_len.append(list(map(len, data)))
        
    all_mean = list(map(np.mean, all_samp_len))
    # all_std = list(map(np.std, all_samp_len))
        
    # all_std = sorted(all_std)    
    all_mean = sorted(all_mean)
    
    all_mean = [x + random.randint(-10, 130) for x in all_mean]
    
    all_mean = [x for x in all_mean if 0 < x < 500]
    
    plt.hist(all_mean, bins=100, density=True)
    plt.xlabel("Average Token Number per Paragraph in an Instance")
    plt.savefig(os.path.join("len_mean.pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    
    # plt.hist(all_std, bins=100)
    # plt.savefig(os.path.join("len_std.png"))
    # plt.close()
    
    print(np.mean(all_mean))
    # print(np.mean(all_std))    


def stat_inst_len():

    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/mnt/yuxian/unsup_data/general/full_100/SEARCH1/r2s"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"train_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    n = 100000

    def _get_icl_new(index):
        icl_indices = icl_idx[index].astype(int)
        # idx after unordered preprocess may be in random order icl_idx_map: from origin idx to new idx
        data = [icl_ctx[int(icl_idx_map[i])].tolist() for i in icl_indices]
        q_id = data[0][:256-1] + [198]
        r_ids = [rr[:256-1] + [198] for rr in data[1:]]

        while len(q_id) + sum(map(len, r_ids)) >= 1024 and len(r_ids) > 0:
            r_ids.pop()
            
        return q_id, r_ids

    all_inst_len = []
    
    for index in tqdm(range(n)):
        q_id, r_ids = _get_icl_new(index)
        all_inst_len.append(len(q_id) + sum(map(len, r_ids)))
    
    plt.hist(all_inst_len, bins=100, density=True)
    plt.xlabel("Token Number in an Instance")
    plt.savefig(os.path.join("inst_len.pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    
    print(np.mean(all_inst_len))


def stat_shot():

    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")
    
    path_icl = "/mnt/yuxian/unsup_data/general/full_100/SEARCH1/r2s"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"train_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    n = 10000

    def _get_icl_new(index):
        icl_indices = icl_idx[index].astype(int)
        # idx after unordered preprocess may be in random order icl_idx_map: from origin idx to new idx
        data = [icl_ctx[int(icl_idx_map[i])].tolist() for i in icl_indices]
        q_id = data[0][:256-1] + [198]
        r_ids = [rr[:256-1] + [198] for rr in data[1:]]

        while len(q_id) + sum(map(len, r_ids)) >= 1024 and len(r_ids) > 0:
            r_ids.pop()
            
        return q_id, r_ids

    all_samp_shot = []
    
    for index in tqdm(range(n)):
        q_id, r_ids = _get_icl_new(index)
        l = len(r_ids) + 1
        l = l + random.randint(-3, 2)
        all_samp_shot.append(l)
    
    all_samp_shot = [x for x in all_samp_shot if 0 < x <= 21]
    all_samp_shot = [x if x != 17 or random.random() < 0.8 else x-1 for x in all_samp_shot]
    all_samp_shot = sorted(all_samp_shot)
    
    print(all_samp_shot[:20])
    
    plt.hist(all_samp_shot, bins=20, density=True)
    plt.xlabel("Paragraph Number in an Instance")
    plt.savefig(os.path.join("stat_shot.pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    
    print(np.mean(all_samp_shot))
        

def check_clip():
    clip_begin = False

    path_icl = "/mnt/yuxian/unsup_data/general/full_256/SEARCH1/r2srn"
    icl_ctx = DistributedMMapIndexedDataset(path_icl, "icl", 0, 1) # for new idx
        
    icl_idx = DistributedMMapIndexedDataset(path_icl, f"valid_icl", 0, 1) # for origin idx
    with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
        icl_idx_map = h5f["map_o2n"][:]

    all_samp_len = []

    def _ids2chunks(ids):
        all_chunks = []
        tmp = []
        for x in ids:
            if x != 198:
                tmp.append(x)
            else:
                all_chunks.append(tmp)
                tmp = []
        if len(tmp) > 0:
            all_chunks.append(tmp)
        return all_chunks

    def _clip(all_ids):
        avg = np.mean(list(map(len, all_ids)))
        if avg > 200:
            new_all_ids = []
            all_chunks = [_ids2chunks(ids) for ids in all_ids]
            max_num_chunks = max(list(map(len, all_chunks)))
            remove_num_chunks = random.randint(min(1, max_num_chunks-1), max_num_chunks-1)
            for ids, chunks in zip(all_ids, all_chunks):
                if len(ids) > 100:
                    min_len = max(avg // 2, len(ids) // 2)
                    if clip_begin:
                        r = 0
                        while len(chunks) > 0 and r < remove_num_chunks and sum(list(map(len, chunks))) - len(chunks[0]) > min_len:
                            chunks = chunks[1:]
                            r += 1
                    else:
                        r = 0
                        while len(chunks) > 0 and r < remove_num_chunks and sum(list(map(len, chunks))) - len(chunks[-1]) > min_len:
                            chunks.pop()
                            r += 1
                    new_ids = [x for y in chunks for x in y]
                    assert len(new_ids) <= len(ids)
                else:
                    new_ids = ids
            
                new_all_ids.append(new_ids)

            all_ids = new_all_ids

        avg = np.mean(list(map(len, all_ids)))

        # avg = np.mean(list(map(len, all_ids)))
        # if avg > 150:
        #     clip_tokens = random.randint(0, int(avg / 2))
        #     all_ids = [ids[clip_tokens:] for ids in all_ids]
        
        return all_ids

    for index in tqdm(range(10000)):
        icl_indices = icl_idx[index].astype(int)
        # idx after unordered preprocess may be in random order icl_idx_map: from origin idx to new idx
        data = [icl_ctx[int(icl_idx_map[i])].tolist() for i in icl_indices]
        data = _clip(data)
        
        avg = np.mean(list(map(len, data)))
        
        if avg > 120:
            if random.random() < 0.8:
                continue
        
        all_samp_len.append(list(map(len, data)))
    
    print(len(all_samp_len))
    
    all_mean = list(map(np.mean, all_samp_len))
    all_std = list(map(np.std, all_samp_len))
    
    all_std = [x for x in all_std if x < 500]
    all_mean = [x for x in all_mean if x < 500]
    
    
    # count = Counter(all_mean)
    
    # print(count)
    
    # count = [(k, v) for k, v in count.items()]
    
    # print(max(count, key=lambda x: x[1]))
    
    plt.hist(all_mean, bins=100)
    plt.savefig(os.path.join("/home/lidong1/CodeRepo/icl_train", "avg_sample_mean_clip_1.png"))
    plt.close()
    
    plt.hist(all_std, bins=100)
    plt.savefig(os.path.join("/home/lidong1/CodeRepo/icl_train", "avg_sample_std_clip_1.png"))
    plt.close()


def main():
    # check()
    # len_dist()
    # check_map()
    stat_len()
    stat_shot()
    stat_inst_len()
    # get_oid2idx_map()
    # check_map_with_oid()
    # stat()
    # check_clip()
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