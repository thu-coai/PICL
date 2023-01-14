import h5py
import os
import torch
import numpy as np
from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from data_utils.indexed_dataset import make_builder
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from tqdm import tqdm


def plot():
    base_path = "/home/guyuxian/CodeRepo/icl_train/unsup_data/general/para_corpus_10K_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/l2_h5/res_-1/256_1024_-1/r2s<n>/"

    ctx = DistributedMMapIndexedDataset(base_path, f"eval_filter", 0, 1)

    name = "score_icl_meta_icl_train_HR_vanilla_test_target_chunk-1_pos0_<n>_r2s_trim_shot16_lr1e-05-bs1-G1-N8_len256-1024-None_gpt2-large_10-10-42_30000_"

    with h5py.File(os.path.join(base_path, "filter_res", f"{name}.h5"), "r") as f:
        scores = f["score"][:]
        
    assert len(scores) == len(ctx)

    all_filter_scores = []

    for idx in range(len(ctx)):
        ids = ctx[idx].astype(int).tolist()
        s = scores[idx]
        avg_s = np.mean(s[:len(ids)-1])
        # print(avg_s)
        all_filter_scores.append(avg_s)
        
    plt.hist(all_filter_scores, bins=40)
    plt.savefig(os.path.join(base_path, f"{name}.png"))    


def diff():
    base_path = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/10M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/l2_h5/res_-1/256_1024_-1/r2s<n>10000/"

    ctx = DistributedMMapIndexedDataset(base_path, f"eval_filter", 0, 1)

    name_base = "score_zs_gpt2-large"
    name_ours = "score_icl_gpt2-large"

    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")

    with h5py.File(os.path.join(base_path, "filter_res", f"{name_base}.h5"), "r") as f:
        scores_base = f["score"][:]

    with h5py.File(os.path.join(base_path, "filter_res", f"{name_ours}.h5"), "r") as f:
        scores_ours = f["score"][:]

    assert len(scores_base) == len(ctx)
    assert len(scores_ours) == len(ctx)

    n = 0
    for idx in tqdm(range(len(ctx))):
        ids = ctx[idx].astype(int).tolist()
        s_base = scores_base[idx]
        s_ours = scores_ours[idx]
        avg_s_base = np.mean(s_base[:len(ids)-1])
        avg_s_ours = np.mean(s_ours[:len(ids)-1])

        if avg_s_ours < avg_s_base:
            # print(tokenizer.decode(ids))
            # print(avg_s_ours, avg_s_base)
            n += 1
            # exit(0)

    print(n)
    
    
def do_filter():
    
    threshold = 0.0
    unsup_valid_num = 10000
    
    base_path = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/100M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/50M/l2_h5/res_-1/256_1024_-1/r2s<n>/10M/"
    save_path = "/home/lidong1/unsup_data_do_filter/general/100M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/50M/l2_h5/res_-1/256_1024_-1/r2s<n>/10M/"
    os.makedirs(os.path.join(save_path, f"filtered_{threshold}"), exist_ok=True)

    print(base_path)
    ctx = DistributedMMapIndexedDataset(base_path, f"eval_filter", 0, 1)

    name_base = "score_zs_gpt2-large"
    name_ours = "score_icl_gpt2-large"

    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large/")

    print("Loading scores")
    with h5py.File(os.path.join(base_path, "filter_res", f"{name_base}.h5"), "r") as f:
        scores_base = f["score"][:]

    with h5py.File(os.path.join(base_path, "filter_res", f"{name_ours}.h5"), "r") as f:
        scores_ours = f["score"][:]
    print("Score load end")

    assert len(scores_base) == len(ctx)
    assert len(scores_ours) == len(ctx)

    train_bin_file = os.path.join(save_path, f"filtered_{threshold}", "train_tokenized_0.bin")
    train_idx_file = os.path.join(save_path, f"filtered_{threshold}", "train_tokenized_0.idx")

    valid_bin_file = os.path.join(save_path, f"filtered_{threshold}", "valid_tokenized_0.bin")
    valid_idx_file = os.path.join(save_path, f"filtered_{threshold}", "valid_tokenized_0.idx")

    train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.uint16)
    valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.uint16)

    n = 0
    for idx in tqdm(range(len(ctx))):
        ids = ctx[idx].astype(int).tolist()
        s_base = scores_base[idx]
        s_ours = scores_ours[idx]
        avg_s_base = np.mean(s_base[:len(ids)-1])
        avg_s_ours = np.mean(s_ours[:len(ids)-1])
        
        if avg_s_ours - avg_s_base < -threshold:
            # print(tokenizer.decode(ids))
            # print(avg_s_ours, avg_s_base)
            if n < unsup_valid_num:
                valid_binary_builder.add_item(torch.IntTensor(ids))
            else:
                train_binary_builder.add_item(torch.IntTensor(ids))

            n += 1
            # exit(0)

    print(n, len(ctx))
    train_binary_builder.finalize(train_idx_file)
    valid_binary_builder.finalize(valid_idx_file)


def position_wise_score():
    base_path = "/home/guyuxian/CodeRepo/icl_train/unsup_data/general/para_corpus_10K_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/l2_h5/res_-1/256_1024_-1/r2s<n>/"

    ctx = DistributedMMapIndexedDataset(base_path, f"eval_filter", 0, 1)

    name = "score_icl_meta_icl_train_HR_vanilla_all_full_chunk-1_pos0_<n>_r2s_trim_shot16_lr1e-05-bs1-G1-N8_len256-1024-None_gpt2-large_10-10-42_30000"
    # name = "score_icl_meta_icl_train_HR_vanilla_test_target_chunk-1_pos0_<n>_r2s_trim_shot16_lr1e-05-bs1-G1-N8_len256-1024-None_gpt2-large_10-10-42_30000_"
    # name = "score_icl_gpt2-large"

    with h5py.File(os.path.join(base_path, "filter_res", f"{name}.h5"), "r") as f:
        scores = f["score"][:]

    assert len(scores) == len(ctx)

    all_idxs = [0, 1, 2]

    for idx in all_idxs:
        ids = ctx[idx].astype(int).tolist()
        s = scores[idx]
        s = s[:len(ids)-1]

        plt.figure(figsize=(30, 10))
        plt.plot(s)
        delim_pos = []
        for p, x in enumerate(ids):
            if x == 198:
                delim_pos.append(p)
        plt.scatter(x=delim_pos, y=[-1] * len(delim_pos))
        plt.savefig(os.path.join(base_path, f"{name}_pos_wise_{idx}.png"))
        plt.close()


def main():
    
    # plot()
    do_filter()
    # position_wise_score()


if __name__ == "__main__":
    main()