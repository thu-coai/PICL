# PICL: Pre-Training to Learn in Context

## 1 Install
```bash
# setup environment with conda
conda create -n picl python=3.8
# install basic packages
pip3 install -r requirements
conda install faiss-gpu -c pytorch
# install transformers & promptsource
pip3 install -e transformers
pip3 install -e promptsource
# install apex
git clone https://github.com/NVIDIA/apex apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## 2 Prepare Corpus
Download [OpenWebText](https://huggingface.co/datasets/openwebtext), [Wikicorpus](https://huggingface.co/datasets/wikicorpus), and [Bookcorpus](https://huggingface.co/datasets/bookcorpus). Run `tools/prepare_raw_data.py` to get all full_documents and merge them:
```bash
python3 tools/prepare_raw_data.py /PATH/TO/openwebtext pretrain_data/raw/openwebtext.txt
python3 tools/prepare_raw_data.py /PATH/TO/wikicorpus pretrain_data/raw/wikicorpus.txt
python3 tools/prepare_raw_data.py /PATH/TO/bookcorpus pretrain_data/raw/bookcorpus.txt
cat pretrain_data/raw/openwebtext.txt pretrain_data/raw/wikicorpus.txt pretrain_data/raw/bookcorpus.txt > pretrain_data/raw/merge_no_shuf.txt
shuf -o pretrain_data/raw/merge_no_shuf.txt pretrain_data/raw/merge.txt
```
The "\n" tokens in full documents are replace by a special token "<@x(x!>" such that each document occupy a single line in the file.

## 3 Run
Run the entire pipeline in a toy setting (corpus size = 100K) with
```
bash pipeline.sh
```
Step-by-step runing. `${BASE_PATH}` is the path of the directory of this project:
### 3.1 Corpus Processing
+ Split full documents into paragraphs.
    ```bash
    bash scripts/tools/process_corpus.sh
    ```
+ Process full document data.
    ```bash
    bash scripts/tools/process_full_doc_data.sh ${BASE_PATH}
    ```
+ Tokenize paragraphs in corpus.
    ```bash
    bash scripts/tools/process_picl_data.sh ${BASE_PATH}
    ```
### 3.2 Retrival
+ Process training data for retriever. Construct hard negatives.
    ```bash
    python3 tools/process_retriever_train_data.py --save retriever_data --data-names TRAIN
    ```
+ Train the retriever.
    ```bash
    bash scripts/retriever/train.sh ${BASE_PATH}
    ```
+ Get encoded paragraphs.
    ```bash
    bash scripts/retriever/infer.sh ${BASE_PATH}
    ```
+ Search for paragraphs that share the same intrinsic tasks.
    ```bash
    bash scripts/retriever/search.sh ${BASE_PATH}
    ```

### 3.3 Filter
+ Filter out non-informative samples.
    ```bash
    bash scripts/filter/filter.sh ${BASE_PATH}
    ```

### 3.4 Pre-train
+ Pre-train the LM with PICL.
    ```bash
    bash scripts/pretrain/pretrain_picl.sh ${BASE_PATH}
    ```

### 3.5 Evaluation
+ Evaluate the trained model on text classification datasets and super-natural instructions.
    ```bash
    bash scripts/eval/eval_cls.sh ${BASE_PATH}
    bash scripts/eval/eval_inst.sh ${BASE_PATH}
    ```