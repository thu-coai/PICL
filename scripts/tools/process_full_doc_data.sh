BASE_PATH=${1}

MAX_LENGTH=1024
DATA_NAME="100K_128"

PYTHONPATH=${BASE_PATH} CODE_BASE=HF python3 ${BASE_PATH}/tools/process_full_doc_data.py \
    --picl-data-dir ${BASE_PATH}/pretrain_data/raw/${DATA_NAME} \
    --picl-data-name processed \
    --processed-output ${BASE_PATH}/pretrain_data/full_doc/${DATA_NAME} \
    --model-dir ${BASE_PATH}/results/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --num-workers 64 \