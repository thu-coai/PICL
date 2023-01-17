BASE_PATH=${1}
DATA_PREFIX=${2-"100K_128"}
DATA_NAME="processed"

MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_picl_data.py \
    --picl-data-dir ${BASE_PATH}/pretrain_data/raw/${DATA_PREFIX} \
    --picl-data-name ${DATA_NAME} \
    --processed-output ${BASE_PATH}/pretrain_data/raw/${DATA_PREFIX}/ \
    --model-dir ${BASE_PATH}/results/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE} \
    --data-process-workers 64 \
