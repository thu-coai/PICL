import json
import random
import os
from tqdm import tqdm
from promptsource.templates import TemplateCollection
from data_utils.data_config import DATA_CONFIG, DATA_GROUP_CONFIG

collection = TemplateCollection()

# with open("/home/lidong1/dpr-simple/results/general/10M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/l2_h5/example_256.jsonl") as f:
#     lines = f.readlines()
#     lines = [json.loads(line) for line in lines]


# with open("unsup_256.jsonl", "w") as f:

#     all_data = []

#     for idx in range(10):
#         test_sample, demos, scores = lines[idx]

#         test_sample = test_sample.replace("<@x(x!>", "\n")

#         demos = [demo.replace("<@x(x!>", "\n") for demo in demos]
            
#         all_data.append([test_sample, demos])
        
#     f.write(json.dumps(all_data, indent=4))

dnn = "art_o"

item = DATA_CONFIG[dnn]
random.seed(10)

d = item.data_dir
name, sub_name = item.name
prompt_name = "Concatenation"
prompt = collection.get_dataset(name, sub_name)
with open(f"/home/lidong1/CodeRepo/{d}/train.jsonl") as f:
    lines = f.readlines()
    train_lines = [json.loads(line) for line in lines]

with open(f"/home/lidong1/CodeRepo/{d}/validation.jsonl") as f:
    lines = f.readlines()
    valid_lines = [json.loads(line) for line in lines]

for prompt_name in prompt.all_template_names:    
    pn = prompt_name.replace(" ", "_")
    dn = name + "-" + (sub_name or "")
    out_d = os.path.join("small_filter_data", dn)
    os.makedirs(out_d, exist_ok=True)
    with open(os.path.join(out_d, f"{pn}.jsonl"), "w") as f:

        all_data = []

        for idx in tqdm(range(100)):
            test_sample = " ".join(prompt[prompt_name].apply(valid_lines[idx]))
            demos = random.sample(train_lines, k=16)
            demos = [" ".join(prompt[prompt_name].apply(demo))
                    for demo in demos]
            all_data.append([test_sample, demos])
            
        f.write(json.dumps(all_data, indent=4))
