import json
import os
from icl_train.data_utils.data_config import DATA_CONFIG
# with open("/home/lidong1/dpr-simple/results_1/general/100M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/50M/l2_h5/res_-1_0") as f:
#     lines = f.readlines()
#     lines = [json.loads(line) for line in lines]


# idx = 106

# test_sample, demos = lines[idx]

# print("##### Test Sample #####")
# print(test_sample.replace("<@x(x!>", "\n"))
# print("#" * 50)
# print()

# for i, demo in enumerate(demos):
#     print(f"##### DEMO #{i}: #####")
#     print(demo.replace("<@x(x!>", "\n"))
#     print()

# print("#" * 50)

# # print("Scores")
# # print(scores)

data_name = "quartz"

with open(os.path.join("/home/lidong1/CodeRepo/", DATA_CONFIG[data_name].data_dir, "train.jsonl")) as f:
    lines = f.readlines()

print(json.loads(lines[20]))
    