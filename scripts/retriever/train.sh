#! /bin/bash

WORKING_DIR=${1}

NUM_GPUS_PER_WORKER=${2-1} # number of gpus used on one node

MASTER_PORT=${3-2010}

# model
MODEL_NAME="roberta-base"
MODEL_DIR="${WORKING_DIR}/checkpoints/${MODEL_NAME}/"
# data
DATA_NAME="${4-TRAIN/p1_en1_hn4_s42}"
DATA_DIR="${WORKING_DIR}/retriever_data/${DATA_NAME}/merge"
# hp
BATCH_SIZE=64
LR=0.00005
GRAD_ACC=1
DEV_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
EPOCHS=10
# runtime
SAVE_PATH="${WORKING_DIR}/results/retriever/"


OPTS=""
# model
OPTS+=" --model-dir ${MODEL_DIR}"
OPTS+=" --ckpt-name ${MODEL_NAME}"
OPTS+=" --share-model"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --data-process-workers 32"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --max-length 256"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
# runtime
OPTS+=" --do-train"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-interval 1"
OPTS+=" --save-log-interval 10"
# seed
OPTS+=" --seed 10"


CMD="python3 ${WORKING_DIR}/retriever.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD}
