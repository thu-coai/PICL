BASE_PATH=${1}
DATA_PREFIX=${2-"100K_128_TRAIN_p1_en1_hn4_s42_lr5e-05-bs64-G1_4212.pt_L2/filtered_0.0"}
DATA_NAME="filtered"

MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/split_picl_train_valid.py \
    --picl-idx-data-dir ${BASE_PATH}/pretrain_data/filter_results/${DATA_PREFIX} \
    --picl-data-name ${DATA_NAME} \
    --picl-data-prefix ${DATA_PREFIX} \
    --processed-output ${BASE_PATH}/pretrain_data/picl/ \
    --model-dir ${BASE_PATH}/results/gpt2-large \
    --max-length ${MAX_LENGTH} \
    --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE} \
    --data-process-workers 64 \
    --picl-valid-num 1000
