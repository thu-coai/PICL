import matplotlib.pyplot as plt
import re
import os
import numpy as np

p = r"train \| epoch   0 \| Iter:.*/.* \| global iter:.*/.* \| loss: (.*) \| lr: .* \|.*"

base_paths = [
    "/home/lidong1/CodeRepo/icl_train/results/pretrain/general_100M_256_256_roberta-base_HR_pos1_easy_neg1_hard_neg1_seed42_concate32_bs64_32_lr0.00005_G1_SEED_4000.pt_50M_l2_h5_res_-1_256_1024_-1_r2s<n>_10M/vanilla/all_target/chunk-1/pos0_<n>/shot16/lr1e-06-bs2-G16-N16-wm0/len256-1024-None/gpt2-large/10-10-42/",
    "/home/lidong1/CodeRepo/icl_train/results/pretrain/general_10M_256_256_roberta-bHR_en1_hn1_cat32_bs64_lr0.00005_G1_4000.pt_r2s<n>_/lm_data_1024_stuffed_rr<n>_1.0/vanilla/all_target/chunk-1/pos0_<n>/shot16/lr1e-06-bs2-G16-N16-wm1000/len256-1024-None/gpt2-large/10-10-42/",
    "/home/lidong1/CodeRepo/icl_train/results/pretrain/general_10M_256_256_roberta-bHR_en1_hn1_cat32_bs64_lr0.00005_G1_4000.pt_r2s<n>_/lm_data_1024_stuffed_rr<n>_1.0_only/vanilla/all_target/chunk-1/pos0_<n>/shot16/lr1e-06-bs2-G16-N16-wm0/len256-1024-None/gpt2-large/10-10-42/",
    "/home/lidong1/CodeRepo/icl_train/results/pretrain/general_10M_256_rand_icl_256_1024_-1_r2s<n>_/vanilla/all_target/chunk-1/pos0_<n>/shot16/lr1e-06-bs2-G16-N16-wm0/len256-1024-None/gpt2-large/10-10-42/",
]

labels = [
    "Retrieve + LM + warmup",
    "Retrieve + LM",
    "LM Only",
    "Rand + LM"
]

for base_path, label in zip(base_paths, labels):
    with open(os.path.join(base_path, "log.txt")) as f:
        lines = f.readlines()
        
    losses = []
        
    for line in lines:
        x = re.match(p, line)
        if x is not None:
            l = x.group(1)
            losses.append(float(l))

    losses = losses[:20000]

    step = 10
    avg_losses = [np.mean(losses[i:i+step]) for i in range(0, len(losses), step)]
    steps = [i * step * 5 for i in range(len(avg_losses))]

    print(len(losses))
    print(len(avg_losses))
        
    plt.plot(steps, avg_losses, label=label)

plt.legend()
plt.savefig(os.path.join("loss.png"))