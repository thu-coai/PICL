#! /bin/bash

WORKING_DIR=${1-"/home/guyuxian/dpr-simple"}

FILE_TYPE="h5"
METRIC_TYPE="l2"
DATA_KEY_1=${2-"general/full_256/SROBERTA/"}
EMBED_DIR="${WORKING_DIR}/results/retriever/${DATA_KEY_1}/"
SAVE_DIR="${WORKING_DIR}/results/retriever/${DATA_KEY_1}/"
MAX_NUM=-1
K=20
BATCH_SIZE=1024


OPTS=""
OPTS+=" --embed_dir ${EMBED_DIR}"
OPTS+=" --save ${SAVE_DIR}"
OPTS+=" --max_num ${MAX_NUM}"
OPTS+=" --search_k ${K}"
OPTS+=" --search_batch_size ${BATCH_SIZE}"
OPTS+=" --file_type ${FILE_TYPE}"
OPTS+=" --metric_type ${METRIC_TYPE}"
OPTS+=" --do-search"


CMD="python3 ${WORKING_DIR}/retriever.py ${OPTS} $@"

echo ${CMD}
mkdir -p ${SAVE_DIR}
${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
