import matplotlib.pyplot as plt

with open("mem_add_icl.txt") as f:
    mems = [int(x.strip()) for x in f.readlines()]
    
plt.plot(mems[:3500])

plt.savefig("mem_add_icl.png")