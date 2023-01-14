from data_utils.data_config import DATA_CONFIG
import sys
import os

if sys.argv[1] == "all":
    for name in DATA_CONFIG:
        if name == "sst2":
            continue
        data_dir = DATA_CONFIG[name]["data_dir"]
        os.system("rm {}".format(os.path.join(data_dir, "*.pkl")))
else:
    for name in sys.argv[5:]:
        data_dir = DATA_CONFIG[name]["data_dir"]
        print(name)
        os.system("rm {}".format(os.path.join(data_dir, "icl_cache_train_1_{}_{}.pkl".format(sys.argv[1], sys.argv[4]))))
        os.system("rm {}".format(os.path.join(data_dir, "icl_cache_valid_1_{}_{}.pkl".format(sys.argv[2], sys.argv[4]))))
        os.system("rm {}".format(os.path.join(data_dir, "icl_cache_test_1_{}_{}.pkl".format(sys.argv[3], sys.argv[4]))))
