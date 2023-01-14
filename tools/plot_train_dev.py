import matplotlib.pyplot as plt
import re
import numpy as np

names = [
    "/home/guyuxian/CodeRepo/icl_train/results/train/MCQA-EXQA-CBQA-TC-PARA/test_target/16_rand/gpt2-large/bmt/10-10/lr0.00005_bs32_G8_ml1024/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/train_unordered/MCQA-EXQA-CBQA-TC-PARA/8_rand/home/guyuxian/checkpoints/gpt2-large/bmt/10-10/lr0.00005_bs16_G8_ml256/log.txt",
    "/home/guyuxian/CodeRepo/icl_train/results/train_unordered_cpos_test_target/MCQA-EXQA-CBQA-TC-PARA/8_rand/home/guyuxian/checkpoints/gpt2-large/bmt/10-10/lr0.00005_bs16_G8_ml256/log.txt"
]
labels = [
    "Meta Train",
    "Many-shot Meta Train (Pos From 0)",
    "Many-shot Meta Train (Pos From end)"
]

show_name = "trec"

for name, label in zip(names, labels):
    with open(name) as f:
        lines = f.readlines()
        
    res = []
    p = re.compile(r"dev \| name: (.*) \| avg res: (.*) \| avg loss: .*")
    for line in lines:
        m = p.match(line)
        if m is not None:
            if m.group(1) == show_name:
                res.append(float(m.group(2)))
    
    res = res[:10]

    print(len(res))

    plt.plot(res, label=label)

plt.legend()
plt.savefig(f"icl_origin_{show_name}.png")