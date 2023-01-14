import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

base_dir = "/home/guyuxian/CodeRepo/icl_train/results/meta_icl/infer/sst2/balance_eval/many_bag/test_target/add_bos/remove_inner_bos/chunk-1/pos3/shot64/len256-None-163840/attn_dtype-float/gpt2-xl/10-10-42"

attention = torch.load(os.path.join(base_dir, "attentions.pt"), map_location="cpu")
model_batch = torch.load(os.path.join(base_dir, "model_batch.pt"), map_location="cpu")
no_model_batch = torch.load(os.path.join(base_dir, "no_model_batch.pt"), map_location="cpu")

print(attention[0].size())
input_ids = model_batch["input_ids"]
position_ids = model_batch["position_ids"]
demo_ids = no_model_batch["all_demo_ids"][0]
print([x.tolist() for x in demo_ids])
demo_lens = [len(x)-1 for x in demo_ids]
print(demo_lens)
sum_demo_lens = sum(demo_lens)

attn_mask = model_batch["attention_mask"]

input_ids = input_ids.tolist()

input_ids = [ids[1:] for ids in input_ids]
input_ids = [ids[:ids.index(50256) if 50256 in ids else -4] for ids in input_ids]

print(input_ids)
print(list(map(len, input_ids)))


plt.figure(figsize=(60, 40))
plt.xticks([])
plt.yticks([])
plt.axis("off")

first_pred_token_pos = -4

for layer in tqdm(range(48)):
    for head in range(25):
        ax = plt.subplot2grid((48, 25), (layer, head))
        pos = ax.get_position()
        if layer == 0:
            plt.figtext(pos.x0 + 0.005, pos.y0 + 0.02, str(head))
        if head == 0:
            plt.figtext(pos.x0 - 0.02, pos.y0 + 0.005, str(layer))
        for idx in range(8):
            attn = attention[layer][idx][head][first_pred_token_pos]
            attn = torch.masked_select(attn, attn_mask[idx][first_pred_token_pos].bool())
            attn = attn.float()
            
            lens = [1] + demo_lens + [len(attn)-1 - sum(demo_lens)]
            lens = torch.tensor(lens, dtype=torch.long)
            
            attn = torch.cumsum(attn, dim=0)
            pos = torch.cumsum(lens, dim=0) - 1
            attn = torch.index_select(attn, dim=0, index=pos)
            attn = torch.diff(attn, dim=0, prepend=torch.zeros(1))
            
            # attn[0] = 0

            plt.plot(attn, label=str(idx))

# print(Counter(max_attn_idxs))
# print(Counter(max_len_idxs))

plt.legend()
plt.savefig(os.path.join(base_dir, "attn_pattern.png"))
