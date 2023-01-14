import os
from data_utils.data_config import DATA_CONFIG, DATA_GROUP_CONFIG

names = [DATA_CONFIG[n].name for n in DATA_GROUP_CONFIG["HR"]]

print(names)

for name in names:
    n, sn = name
    if sn is None:
        sn = ""
    
    d = os.path.join("/home/guyuxian/promptsource/promptsource/templates", n + "_origin", sn)
    od = os.path.join("/home/guyuxian/MetaICL/preprocess/promptsource/promptsource/templates/", n, sn)
    os.makedirs(d, exist_ok="True")
    os.system("cp {} {}".format(os.path.join(od, "templates.yaml"), d))