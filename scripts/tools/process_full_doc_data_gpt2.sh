BASE_PATH=${1}

MAX_LENGTH=1024
MODEL_TYPE="gpt2"

PYTHONPATH=${BASE_PATH} CODE_BASE=HF python3 ${BASE_PATH}/tools/process_full_doc_data.py \
    --picl-data-dir ${BASE_PATH}/pretrain_data/ \
    --picl-data-name merge.txt \
    --processed-output ${BASE_PATH}/pretrain_data/full_doc/${MODEL_TYPE}/ \
    --model-dir ${BASE_PATH}/checkpoints/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --data-process-workers 64 \