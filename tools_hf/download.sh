DATA=app_reviews
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/sealuzh/user_quality/master/csv_files/reviews.csv -P /home/guyuxian/data_hf/${DATA}/

DATA=commensense_qa
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl -P /home/guyuxian/data_hf/${DATA}/
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -P /home/guyuxian/data_hf/${DATA}/
wget https://s3.amazonaws.com/commensenseqa/test_rand_split.jsonl -P /home/guyuxian/data_hf/${DATA}/

DATA=dbpedia_14
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz -P /home/guyuxian/data_hf/${DATA}/

DATA=dream
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/nlpdata/dream/master/data/train.jsonl -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/nlpdata/dream/master/data/dev.jsonl -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/nlpdata/dream/master/data/test.jsonl -P /home/guyuxian/data_hf/${DATA}/

DATA=duorc
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/duorc/duorc/master/dataset/SelfRC_train.json -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/duorc/duorc/master/dataset/SelfRC_dev.json -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/duorc/duorc/master/dataset/SelfRC_test.json -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/duorc/duorc/master/dataset/ParaphraseRC_train.json -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/duorc/duorc/master/dataset/ParaphraseRC_dev.json -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/duorc/duorc/master/dataset/ParaphraseRC_test.json -P /home/guyuxian/data_hf/${DATA}/

DATA=hotpot_qa
mkdir /home/guyuxian/data_hf/${DATA}/
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -P /home/guyuxian/data_hf/${DATA}/

DATA=imdb
mkdir /home/guyuxian/data_hf/${DATA}/
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P /home/guyuxian/data_hf/${DATA}/

DATA=qasc
mkdir /home/guyuxian/data_hf/${DATA}/
wget http://data.allenai.org/downloads/qasc/qasc_dataset.tar.gz -P /home/guyuxian/data_hf/${DATA}/

DATA=quail
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/text-machine-lab/quail/master/quail_v1.3/xml/randomized/quail_1.3_challenge_randomized.xml -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/text-machine-lab/quail/master/quail_v1.3/xml/randomized/quail_1.3_dev_randomized.xml -P /home/guyuxian/data_hf/${DATA}/
wget https://raw.githubusercontent.com/text-machine-lab/quail/master/quail_v1.3/xml/randomized/quail_1.3_train_randomized.xml -P /home/guyuxian/data_hf/${DATA}/

DATA=quarel
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://s3-us-west-2.amazonaws.com/ai2-website/data/quarel-dataset-v1-nov2018.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=quartz
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://s3-us-west-2.amazonaws.com/ai2-website/data/quartz-dataset-v1-aug2019.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=quoref
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=ropes
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://ropes-dataset.s3-us-west-2.amazonaws.com/train_and_dev/ropes-train-dev-v1.0.tar.gz -P /home/guyuxian/data_hf/${DATA}/
wget https://ropes-dataset.s3-us-west-2.amazonaws.com/test/ropes-test-questions-v1.0.tar.gz -P /home/guyuxian/data_hf/${DATA}/

DATA=rotten_tomatoes
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz -P /home/guyuxian/data_hf/${DATA}/

DATA=sciq
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://s3-us-west-2.amazonaws.com/ai2-website/data/SciQ.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=social_i_qa
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=trec
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label -P /home/guyuxian/data_hf/${DATA}/
wget https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label -P /home/guyuxian/data_hf/${DATA}/

DATA=wiki_bio
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://huggingface.co/datasets/wiki_bio/resolve/main/data/wikipedia-biography-dataset.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=wiki_qa
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://download.microsoft.com/download/E/5/f/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip -P /home/guyuxian/data_hf/${DATA}/

DATA=wiqa
mkdir /home/guyuxian/data_hf/${DATA}/
wget https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/wiqa-dataset-v2-october-2019.zip -P /home/guyuxian/data_hf/${DATA}/
