#! /bin/bash

WORKING_DIR=${1}

NUM_GPUS_PER_WORKER=${2-4} # number of gpus used on one node

MASTER_PORT=${3-2012}

# model
MODEL_DIR="${WORKING_DIR}/checkpoints/roberta-base/"
# data
DATA_DIR="${WORKING_DIR}/pretrain_data/"
DATA_NAME=${5-"100K_128"}
# hp
BATCH_SIZE=128
SEED=10
MAX_LEN=256
# runtime
CKPT=${4-"TRAIN_p1_en1_hn4_s42/lr5e-05-bs64-G1/4212.pt"}
LOAD_PATH="${WORKING_DIR}/results/retriever/${CKPT}"
SAVE_PATH="${WORKING_DIR}/pretrain_data/retrieval_results/"


OPTS=""
# model
OPTS+=" --model-dir ${MODEL_DIR}"
OPTS+=" --ckpt-name ${CKPT}"
OPTS+=" --share-model"
OPTS+=" --data-names ${DATA_NAME}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-process-workers 32"
# hp
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --max-length ${MAX_LEN}"
# runtime
OPTS+=" --do-infer"
OPTS+=" --load ${LOAD_PATH}"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"


CMD="torchrun --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/retriever.py ${OPTS}"

export TF_CPP_MIN_LOG_LEVEL=3
echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD}
