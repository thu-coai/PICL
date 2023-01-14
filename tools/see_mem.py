import os
import re
import matplotlib.pyplot as plt

fout = open("mem_add_icl.txt", "w")

for i in range(1000000000000):
    out = os.popen("top -n 1")
    info = out.readlines()
    out.close()
    mem = info[3]
    m = re.match(r".*free,(.*)used.*", mem)
    tmp = m.group(1).split("+")[0]
    tmp = tmp.split(" ")[1]
    use = int(tmp)
    fout.write(str(use) + "\n")
    print(use)
    
fout.close()