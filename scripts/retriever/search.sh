#! /bin/bash

WORKING_DIR=${1}

MODEL_DIR="${WORKING_DIR}/checkpoints/roberta-base/"
METRIC_TYPE="L2"
DATA_NAME=${2-"100K_128/TRAIN_p1_en1_hn4_s42_lr5e-05-bs64-G1_4212.pt"}
EMBED_DIR="${WORKING_DIR}/pretrain_data/retrieval_results/${DATA_NAME}"
SAVE_DIR="${WORKING_DIR}/pretrain_data/retrieval_results/"
MAX_NUM=-1
K=20
BATCH_SIZE=1024


OPTS=""
OPTS+=" --model-dir ${MODEL_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --embed-dir ${EMBED_DIR}"
OPTS+=" --save ${SAVE_DIR}"
OPTS+=" --data-num ${MAX_NUM}"
OPTS+=" --search-k ${K}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --metric-type ${METRIC_TYPE}"
OPTS+=" --do-search"


CMD="python3 ${WORKING_DIR}/retriever.py ${OPTS} $@"

echo ${CMD}
mkdir -p ${SAVE_DIR}
${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
