
BASE_PATH="/home/guyuxian/CodeRepo"

MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256
MAX_LENGTH_ALL_DEMOS=-1

PYTHONPATH=${BASE_PATH} CODE_BASE=HF python3 ${BASE_PATH}/icl_train/tools/process_lm_data_for_filter.py \
    --unsup-data-path /home/guyuxian/dpr-simple/pretrain_data/ \
    --unsup-data-name merge \
    --processed-unsup-data-path ${BASE_PATH}/icl_train/unsup_data/lm_data/${MAX_LENGTH}/no_stuffed/ \
    --model-config ${BASE_PATH}/icl_train/results/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE} \
    --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS} \
    --data-process-workers 64 \
    --no-extend-save-path \
    --filter-num 10000
    # --replace-return-with-space
    # --end-token "<eos>"