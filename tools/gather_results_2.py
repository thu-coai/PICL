import re
import json
from collections import defaultdict

file_names = [
    "/home/guyuxian/CodeRepo/icl_train/results_new/infer/sst2-rte-cb-copa-wsc-wic/few_unordered_2/16/MCQA-EXQA-CBQA-TC-PARA/few_unordered_2/test_target/16/gpt2-large/10-10/lr0.00005_bs2_G128_ml1024/800/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_new/infer/sst2-rte-cb-copa-wsc-wic/few_unordered_2/16/MCQA-EXQA-CBQA-TC-PARA/few_unordered_2/test_target/16/gpt2-large/20-10/lr0.00005_bs2_G128_ml1024/800/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_new/infer/sst2-rte-cb-copa-wsc-wic/few_unordered_2/16/MCQA-EXQA-CBQA-TC-PARA/few_unordered_2/test_target/16/gpt2-large/30-10/lr0.00005_bs2_G128_ml1024/800/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_new/infer/sst2-rte-cb-copa-wsc-wic/few_unordered_2/16/MCQA-EXQA-CBQA-TC-PARA/few_unordered_2/test_target/16/gpt2-large/40-10/lr0.00005_bs2_G128_ml1024/800/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results_new/infer/sst2-rte-cb-copa-wsc-wic/few_unordered_2/16/MCQA-EXQA-CBQA-TC-PARA/few_unordered_2/test_target/16/gpt2-large/50-10/lr0.00005_bs2_G128_ml1024/800/10-10/share-balance/log.txt",
]

res = defaultdict(list)

for name in file_names:
    print(name)
    with open(name) as f:
        lines = f.readlines()
        
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    p = re.compile(r"test \| name: (.*) \| avg res: (.*) \| avg loss: (.*) \| res: ({.*}) \| loss: ({.*})")

    for line in lines:
        m = p.match(line)
        if m is not None:
            # print(m.group(1))
            # print(m.group(2))
            # print(m.group(3))
            all_res = json.loads(m.group(4).replace("\'", "\""))
            
            res[m.group(1)].append((
                "\t".join([k for k in all_res.keys()]),
                "\t".join([str(round(v.get("accuracy", v.get("bleu", 0)), 2)) for v in all_res.values()])
            ))

for n in res:
    print(n)
    for k, _ in res[n]:
        print(k)
    for _, v in res[n]:
        print(v)
