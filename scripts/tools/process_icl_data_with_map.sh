
BASE_PATH="/home/lidong1/CodeRepo"
DATA_PATH=${1-"general/full_256/SEARCH1"}
DATA_NAME="raw"

MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256
MAX_LENGTH_ALL_DEMOS=-1

PYTHONPATH=${BASE_PATH} CODE_BASE=HF python3 ${BASE_PATH}/icl_train/tools/process_icl_data_with_map.py \
    --unsup-data-path /home/lidong1/dpr-simple/results/${DATA_PATH} \
    --unsup-data-name ${DATA_NAME} \
    --processed-unsup-data-path ${BASE_PATH}/icl_train/unsup_data_1/${DATA_PATH}/ \
    --model-config ${BASE_PATH}/icl_train/results/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE} \
    --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS} \
    --data-process-workers 64 \
    --no-extend-save-path \
    --replace-return-with-space
