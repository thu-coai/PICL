import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

base_dir = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/gpt2-large/10-10-42"

attention = torch.load(os.path.join(base_dir, "attentions.pt"), map_location="cpu")
model_batch = torch.load(os.path.join(base_dir, "model_batch.pt"), map_location="cpu")
no_model_batch = torch.load(os.path.join(base_dir, "no_model_batch.pt"), map_location="cpu")

print(attention[0].size())
input_ids = model_batch["input_ids"]
position_ids = model_batch["position_ids"]
demo_ids = no_model_batch["all_demo_ids"][0]
print([x.tolist() for x in demo_ids])
demo_lens = [len(x) for x in demo_ids]

sum_demo_lens = sum(demo_lens)

# print(input_ids[0].tolist())

attn_mask = model_batch["attention_mask"]

plt.figure(figsize=(40, 40))
plt.xticks([])
plt.yticks([])
plt.axis("off")

first_pred_token_pos = -4

all_attn = []

for layer in tqdm(range(36)):
    for head in range(20):
        ax = plt.subplot2grid((36, 20), (layer, head))
        pos = ax.get_position()
        if layer == 0:
            plt.figtext(pos.x0 + 0.005, pos.y0 + 0.02, str(head))
        if head == 0:
            plt.figtext(pos.x0 - 0.02, pos.y0 + 0.005, str(layer))
        for idx in range(8):
            attn = attention[layer][idx][head][first_pred_token_pos]
            attn = torch.masked_select(attn, attn_mask[idx][first_pred_token_pos].bool())
            attn = attn.float()
            
            attn[0] = 0
            
            lens = demo_lens + [len(attn) - sum(demo_lens)]
            lens = torch.tensor(lens, dtype=torch.long)
            
            attn = torch.cumsum(attn, dim=0)
            pos = torch.cumsum(lens, dim=0) - 1
            attn = torch.index_select(attn, dim=0, index=pos)
            attn = torch.diff(attn, dim=0, prepend=torch.zeros(1))
            
            plt.plot(attn, label=str(idx))
            
            all_attn.append(attn)


plt.legend()
plt.savefig(os.path.join(base_dir, "attn_pattern.png"))
plt.close()
avg_attn = sum(all_attn) / len(all_attn)

plt.plot(avg_attn)
plt.savefig(os.path.join(base_dir, "attn_pattern_avg.png"))
plt.close()