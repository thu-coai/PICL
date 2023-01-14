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

for name, label in zip(names, labels):
    with open(name) as f:
        lines = f.readlines()
        
    losses = []
    p = re.compile(r"train \| epoch (.*) \| Iter: (.*) \| global iter: (.*) \| loss: (.*) \| lr: (.*)")
    for line in lines:
        m = p.match(line)
        if m is not None:
            losses.append(float(m.group(4)))
    losses = losses[:3000]

    step = 8

    X = [i for i in range(0, len(losses), step)]
    Y = [np.mean(losses[i:i+step]) for i in X]

    print(len(losses), len(Y))

    plt.plot(X, Y, label=label)

plt.legend()
plt.savefig("icl_origin.png")
