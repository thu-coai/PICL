import re
import json
from collections import defaultdict

file_names = [
    "/home/guyuxian/CodeRepo/icl_train/results_cache/infer/sst2/many_add_bos_remove_inner_bos_3/chunk1024_end<eos>/1024/gpt2-xl/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_cache/infer/sst2/many_add_bos_remove_inner_bos_3/chunk1024_end<eos>/1024/gpt2-xl/20-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_cache/infer/sst2/many_add_bos_remove_inner_bos_3/chunk1024_end<eos>/1024/gpt2-xl/30-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_cache/infer/sst2/many_add_bos_remove_inner_bos_3/chunk1024_end<eos>/1024/gpt2-xl/40-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_cache/infer/sst2/many_add_bos_remove_inner_bos_3/chunk1024_end<eos>/1024/gpt2-xl/50-10/share-balance/log.txt",
]

for name in file_names:
    # print(name)
    with open(name) as f:
        lines = f.readlines()
        
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    p = re.compile(r"test \| name: (.*) \| avg res: (.*) \| avg loss: (.*) \| res: ({.*}) \| loss: ({.*})")

    res = []

    for line in lines:
        m = p.match(line)
        if m is not None:
            # print(m.group(1))
            # print(m.group(2))
            # print(m.group(3))
            all_res = json.loads(m.group(4).replace("\'", "\""))
            print(all_res.keys())
            v = list(all_res.values())[0]
            res.append(str(round(v.get("accuracy", v.get("bleu", 0)), 2)))
    
    print("\t".join(res))
