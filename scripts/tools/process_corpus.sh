python3 tools/process_corpus.py \
    --model-dir checkpoints/roberta-base \
    --raw-input pretrain_data/raw/merge.txt \
    --processed-output pretrain_data/raw/ \
    --data-num 100000 \
    --max-length 128 \
    --log-interval 10000
