python3 tools/process_corpus.py \
    --model-dir checkpoints/roberta-base \
    --raw-input pretrain_data/merge.txt \
    --processed-output pretrain_data/ \
    --data-num 100000 \
    --max-length 128 \
    --log-interval 10000 \
    --data-process-workers 32
