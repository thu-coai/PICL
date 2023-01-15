#! /bin/bash

WORKING_DIR=${1-"/home/guyuxian/dpr-simple"}

NUM_GPUS_PER_WORKER=${2-2} # number of gpus used on one node

MASTER_PORT=${3-2012}

# model
MODEL_NAME="roberta-base"
MODEL_DIR="${WORKING_DIR}/checkpoints/${MODEL_NAME}/"
# data
DATA_DIR="${WORKING_DIR}/pretrain_data/"
DATA_NAME=${5-"general/full_256"}
SEARCH=${6-SEARCH1}
# hp
BATCH_SIZE=128
SEED=10
MAX_LEN=256
# runtime
CKPT=${4-"HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt"}
LOAD_PATH="${WORKING_DIR}/results/retriever/${CKPT}"
SAVE_PATH="${WORKING_DIR}/results/retriever/${DATA_NAME}/${SEARCH}/"


OPTS=""
# model
OPTS+=" --model_dir ${MODEL_DIR}"
OPTS+=" --model_name ${MODEL_NAME}"
OPTS+=" --share_model"
OPTS+=" --data_name ${DATA_NAME}"
# data
OPTS+=" --data_dir ${DATA_DIR}"
OPTS+=" --load_data_workers 32"
OPTS+=" --search_chunk 1000000000"
# hp
OPTS+=" --batch_size ${BATCH_SIZE}"
OPTS+=" --max_len ${MAX_LEN}"
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
