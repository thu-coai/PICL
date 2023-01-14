from data_utils.data_config import DATA_CONFIG, DATA_GROUP_CONFIG
import os

for dn in DATA_GROUP_CONFIG["HR"]:
    print(dn)
    print(DATA_CONFIG[dn].data_dir)
    os.system("cd {}; rm *.pkl".format(DATA_CONFIG[dn].data_dir))
    