BASE_PATH="/home/lidong1/CodeRepo"

MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256
MAX_LENGTH_ALL_DEMOS=-1

PYTHONPATH=${BASE_PATH} CODE_BASE=HF python3 ${BASE_PATH}/icl_train/tools/process_lm_data.py \
    --unsup-data-path /home/lidong1/dpr-simple/pretrain_data/ \
    --unsup-data-name merge_full \
    --processed-unsup-data-path /home/lidong1/unsup_data_1/lm_data/full/stuffed/ \
    --model-config ${BASE_PATH}/icl_train/results/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE} \
    --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS} \
    --data-process-workers 64 \
    --no-extend-save-path \
    --stuffed
    # --end-token "<eos>"