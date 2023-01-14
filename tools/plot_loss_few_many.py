import matplotlib.pyplot as plt
import re

with open("/home/guyuxian/CodeRepo/icl_train/train_many_attn_32.log") as f:
    log_many_32 = f.readlines()

with open("/home/guyuxian/CodeRepo/icl_train/train_many_check_attn_32.log") as f:
    log_many = f.readlines()
    
with open("/home/guyuxian/CodeRepo/icl_train/train_few_check_attn_32.log") as f:
    log_few = f.readlines()
    
p = r"train \|.*loss: (.*) \| lr: 5.0000e-05 \|.* step time: 0.000"

losses_many_32, losses_many, losses_few = [], [], []

for line in log_many_32:
    m = re.match(p, line)
    if m is not None:
        loss = float(m.group(1))
        losses_many_32.append(loss)

for line in log_many:
    m = re.match(p, line)
    if m is not None:
        loss = float(m.group(1))
        losses_many.append(loss)

for line in log_few:
    m = re.match(p, line)
    if m is not None:
        loss = float(m.group(1))
        losses_few.append(loss)


tot = min(len(losses_many_32), len(losses_few), len(losses_many))

print(len(losses_many_32))
print(len(losses_many))
print(len(losses_few))

start = 0

losses_many = losses_many[start:tot]
losses_few = losses_few[start:tot]
losses_many_32 = losses_many_32[start:tot]
        
steps = [4 * i for i in range(start, tot)]

plt.plot(steps, losses_few, label="loss vanilla")
plt.plot(steps, losses_many, label="loss ours")
plt.plot(steps, losses_many_32, label="loss ours attn 32")

plt.legend()

plt.savefig("compare_many_few_attn_fp32.png")
