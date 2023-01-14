import json
import random
import os

with open("/home/guyuxian/data_hf/wsc/cache/validation.jsonl") as f:
    lines = f.readlines()
    
lines = [json.loads(line) for line in lines]

# p, n = 0, 0
# for line in lines:
#     if line["label"] == 0:
#         n += 1
#     else:
#         p += 1
        
# print(p, n)

pos_set = []
neg_set = []

for line in lines:
    if line["label"] == 1:
        pos_set.append(line)
    else:
        neg_set.append(line)
        
min_len = min(len(pos_set), len(neg_set))

pos_set = pos_set[:min_len]
neg_set = neg_set[:min_len]

all_set = pos_set + neg_set

random.seed(42)

random.shuffle(all_set)

os.makedirs("/home/guyuxian/data_hf/wsc_balance/cache/", exist_ok=True)
with open("/home/guyuxian/data_hf/wsc_balance/cache/validation.jsonl", "w") as f:
    for d in all_set:
        f.write(json.dumps(d) + "\n")
