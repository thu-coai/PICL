import os
import json

base_dir = "/home/lidong1/CodeRepo/"

for s_split, t_split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
    with open(os.path.join(base_dir, "data", "sst5", f"{s_split}.jsonl")) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    
    os.makedirs(os.path.join(base_dir, "data", "sst5", "cache"), exist_ok=True)
    with open(os.path.join(base_dir, "data", "sst5", "cache", f"{t_split}.jsonl"), "w") as f:
        for line in lines:
            new_line = {
                "sentence": line["text"],
                "label": line["label"]
            }
            f.write(json.dumps(new_line) + "\n")
        