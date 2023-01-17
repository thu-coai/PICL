BASE_PATH="/home/lidong1/yuxian/PICL/"

# Process general corpus. Split documents into paragraphs.
bash scripts/tools/process_corpus.sh

# Process full document data.
bash scripts/tools/process_full_doc_data.sh ${BASE_PATH}

# Tokenize paragraphs in corpus
bash scripts/tools/process_picl_data.sh ${BASE_PATH}

# Process training data for retriever. Construct hard negatives.
python3 tools/process_retriever_train_data.py --save retriever_data --data-names TRAIN

# Train the retriever.
bash scripts/retriever/train.sh ${BASE_PATH}

# Get encoded paragraphs.
bash scripts/retriever/infer.sh ${BASE_PATH}

# Search for paragraphs that share the same intrinsic tasks.
bash scripts/retriever/search.sh ${BASE_PATH}

# Filter
bash scripts/filter/filter.sh ${BASE_PATH}

# Split train/valid
bash scripts/tools/split_picl_train_valid.sh ${BASE_PATH}

# Pretrain
bash scripts/pretrain/pretrain_picl.sh ${BASE_PATH}

# Evaluation
bash scripts/eval/eval_cls.sh ${BASE_PATH}
bash scripts/eval/eval_inst.sh ${BASE_PATH}
