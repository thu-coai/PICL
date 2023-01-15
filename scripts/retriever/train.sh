#! /bin/bash

WORKING_DIR=${1}

NUM_GPUS_PER_WORKER=${2-1} # number of gpus used on one node

MASTER_PORT=${3-2010}

# model
MODEL_NAME="roberta-base"
MODEL_DIR="${WORKING_DIR}/checkpoints/${MODEL_NAME}/"
# data
DATA_TAG="${4-HR/pos1_easy_neg1_hard_neg1_seed42_concate32}"
DATA_DIR="${WORKING_DIR}/dpr_data/${DATA_TAG}/merge"
# hp
BATCH_SIZE=64
LR=0.00005
GRAD_ACC=1
DEV_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
EPOCHS=10
# runtime
SAVE_PATH="${WORKING_DIR}/results/retriever/${DATA_TAG}/bs${BATCH_SIZE}_${EVAL_BATCH_SIZE}_lr${LR}_G${GRAD_ACC}_SEED${SEED}"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --model_dir ${MODEL_DIR}"
OPTS+=" --model_name ${MODEL_NAME}"
OPTS+=" --share_model"
# data
OPTS+=" --data_dir ${DATA_DIR}"
OPTS+=" --load_data_workers 32"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch_size ${BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps ${GRAD_ACC}"
OPTS+=" --eval_batch_size ${EVAL_BATCH_SIZE}"
OPTS+=" --epochs ${EPOCHS}"
# runtime
OPTS+=" --do-train"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"


CMD="python3 ${WORKING_DIR}/retriever.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD}
