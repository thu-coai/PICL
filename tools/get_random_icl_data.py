import random
import os
import json
from tqdm import tqdm

random.seed(42)

with open("/home/lidong1/dpr-simple/pretrain_data/general/10M_256/raw_dup.txt") as f:
    lines = f.readlines()

os.makedirs("/home/lidong1/dpr-simple/results_1/general/10M_256/", exist_ok=True)

fo = open("/home/lidong1/dpr-simple/results_1/general/10M_256/rand_icl.txt", "w")
for line in tqdm(lines):
    demos = random.sample(lines, k=20)
    fo.write(json.dumps([
        line.strip(),
        [demo.strip() for demo in demos]
    ]) + "\n")
