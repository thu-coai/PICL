import re
import json

file_names = [
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/10-20/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/10-30/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/10-40/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/10-50/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/10-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/20-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/30-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/40-10/share-balance/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/infer/sst2-rte-cb-copa-wsc-wic/8/MCQA-EXQA-CBQA-TC-PARA/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G4_ml512/800/50-10/share-balance/log.txt",
]

for name in file_names:
    print(name)
    with open(name) as f:
        lines = f.readlines()
        
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    p = re.compile(r"test \| name: (.*) \| avg res: (.*) \| avg loss: (.*) \| res: ({.*}) \| loss: ({.*})")

    for line in lines:
        m = p.match(line)
        if m is not None:
            print(m.group(1))
            print(m.group(2))
            print(m.group(3))
            all_res = json.loads(m.group(4).replace("\'", "\""))
            
            print("\t".join([k for k in all_res.keys()]))
            print("\t".join([str(round(v.get("accuracy", v.get("bleu", 0)), 2)) for v in all_res.values()]))
            print("\n")
        
            
        