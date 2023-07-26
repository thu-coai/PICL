BASE_PATH=${1}

INPUT_PATH_1="${BASE_PATH}/pretrain_data/full_doc/gpt2/train_lm_1"
INPUT_PATH_2="${BASE_PATH}/pretrain_data/full_doc/gpt2/train_lm_2"

OUTPUT_PATH="${BASE_PATH}/pretrain_data/full_doc/gpt2/train_lm_0"

PYTHONPATH=${BASE_PATH} python3 tools/merge_bin_files.py $INPUT_PATH_1 $INPUT_PATH_2 $OUTPUT_PATH