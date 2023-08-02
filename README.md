# Pre-Training to Learn in Context

This repository contains the code of our ACL2023 paper:

> [Pre-Training to Learn in Context](https://arxiv.org/pdf/2305.09137.pdf)

In this work, we propose PICL (**P**re-training for **I**n-**C**ontext **L**earning), a framework to enhance the language models' in-context learning ability by pre-training the model on a large collection of ``intrinsic tasks'' in the general plain-text corpus using the simple language modeling objective. PICL encourages the model to infer and perform tasks by conditioning on the contexts while maintaining task generalization of pre-trained models. 

![PICL](figures/method.png "PICL Framework")

## 1 Install
```bash
# setup environment with conda
conda create -n picl python=3.8
# install basic packages
pip3 install -r requirements.txt
conda install faiss-gpu -c pytorch
# install transformers & promptsource
pip3 install -e transformers
pip3 install -e promptsource
```

## 2 Prepare Plain-Text Corpus
Download [OpenWebText](https://huggingface.co/datasets/openwebtext), [Wikicorpus](https://huggingface.co/datasets/wikicorpus), and [Bookcorpus](https://huggingface.co/datasets/bookcorpus). Run `tools/prepare_raw_data.py` to get all full_documents and merge them:
```bash
python3 tools/prepare_raw_data.py /PATH/TO/openwebtext pretrain_data/raw/openwebtext.txt
python3 tools/prepare_raw_data.py /PATH/TO/wikicorpus pretrain_data/raw/wikicorpus.txt
python3 tools/prepare_raw_data.py /PATH/TO/bookcorpus pretrain_data/raw/bookcorpus.txt
cat pretrain_data/raw/openwebtext.txt pretrain_data/raw/wikicorpus.txt pretrain_data/raw/bookcorpus.txt > pretrain_data/raw/merge_no_shuf.txt
shuf -o pretrain_data/raw/merge_no_shuf.txt pretrain_data/raw/merge.txt
```
The "\n" tokens in full documents are replace by a special token "<@x(x!>" such that each document occupy a single line in the file.

## 3 Run the Pipeline
Run the entire pipeline in a toy setting (corpus size = 100K) with
```
bash pipeline.sh
```
`${BASE_PATH}` is the path of the directory of this project. 

The details of each step in the pipeline are shown in the following sections.

## 4 Construct PICL Data

We release the constructed PICL data in this [link](https://huggingface.co/t1101675/PICL/tree/main/pretrain_data).

You can check the same-intrinsic-task paragraphs by running `python3 check_picl_data.py` and then entering an interger index to pick a query and the retrieved paragraphs:
<details><summary><b>Latex Equation Translation</b></summary>

```
Input Paragraph Index >>>11156                                                         
##########  Query  ##########
ω p = I s ω s I p cos ⁡ ( α ) {\displaystyle {\boldsymbol {\omega }}_{\mathrm {p} }={\frac {{\boldsymbol {I}}_{\mathrm {s} }{\boldsymbol {\omega }}_{\mathrm {s} }}{{\boldsymbo
l {I}}_{\mathrm {p} }\cos({\boldsymbol {\alpha }})}}}

##########  Retrieved Paragraph #1  ##########
τ b ∗ = τ b ( ρ s − ρ f ) ( g ) ( D ) {\displaystyle \tau _{b}*={\frac {\tau _{b}}{(\rho _{s}-\rho _{f})(g)(D)}}}


##########  Retrieved Paragraph #2  ##########
M H ≤ ℏ c 3 8 π G k B T u {\displaystyle M_{\mathrm {H} }\leq {\frac {\hbar c^{3}}{8\pi Gk_{\mathrm {B} }T_{\mathrm {u} }}}}

...
```
</details>


<details><summary><b>Question Ansering</b></summary>

```
##########  Query  ##########
Question: Where would a gnarly off-road racer like Tanner Foust meet up with a frightened five-year-old child with leukemia? Answer: In a hospital, of course!


##########  Retrieved Paragraph #1  ##########
Question: What do a siren, an in-wall light switch, a sleep sensing iPhone dock, and a flood detector have in common? Answer: They are all SmartThings!


##########  Retrieved Paragraph #2  ##########
Question: Where do you find a one legged dog? Answer: Where you left it.
...
```
</details>

Here are some indices for interesting paragraphs. Try it out!

<details><summary><b>Indices</b></summary>

```
0
8
109
1000
4645
5384
9473
11156
11969
12231
17838
17849
28844
28845
37577
40119
59996
85034
90096
97616
```
</details>

You can also constructe the PICL data from scratch following the instructions below.

### 4.1 Preprocessing and Toknization
Tokenize and store full documents and paragraphs into binary files.
+ Split full documents into paragraphs.
    ```bash
    bash scripts/tools/process_corpus.sh
    ```
+ Process full document data. The scripts will generate `.bin` and `.idx` files.
    ```bash
    bash scripts/tools/process_full_doc_data_gpt2.sh ${BASE_PATH}
    ```
+ Tokenize paragraphs in corpus. The scripts will generate `.bin` and `.idx` files.
    ```bash
    bash scripts/tools/process_picl_data_gpt2.sh ${BASE_PATH}
    ```

#### NOTE
Since the corpus is large, the `.bin` file of full document data will be about 29G and the paragraph data will be about 13G. The data processing may take long, and there may be unexpected problems that stuck the process (like running out of CPU memories). To handle this issue, you can split the `merge.txt` file to multiple files like:
```bash
split -C 1000M merge.txt
```
And then, you can process the split files one by one (by setting different `picl-data-name` and `bin-file-index` in `process_full_doc_data_gpt2.sh`), each takes less time and has less risk of running into problems. Assume that you have generated two (`.bin`, `.idx`) pairs:
```
train_lm_1.bin
train_lm_1.idx
train_lm_2.bin
train_lm_2.idx
```
You can finally merge them by runing
```bash
bash scripts/tools/merge_bin_files.sh ${BASE_PATH}
```
which will merge the two pairs into `train_lm_0.bin` and `train_lm_0.idx`.

### 4.2 Retrival
+ Process training data for retriever. Construct hard negatives. The raw data and the preprocessed data can be downloaded from this [link](https://huggingface.co/t1101675/PICL/tree/main/retriever_data). To do data processing, put the raw datasets under `data/` and run the following command:
    ```bash
    python3 tools/process_retriever_train_data.py --save retriever_data --data-names TRAIN
    ```
+ Train the retriever. The `train.jsonl` and `valid.jsonl` data should be put in `retriever_data/TRAIN/p1_en1_hn1_s42/merge`. The trained retriever can be downloaded from this [link](https://huggingface.co/t1101675/PICL/tree/main/results/retriever).
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

### 4.2 Filter
+ Filter out non-informative samples.
    ```bash
    bash scripts/filter/filter.sh ${BASE_PATH}
    ```

## 5 Pre-train
+ Pre-train the LM with PICL. The pre-trained models can be downloaded from this [link](https://huggingface.co/t1101675/PICL/tree/main/results/picl).
    ```bash
    bash scripts/pretrain/pretrain_picl_gpt2_large.sh ${BASE_PATH}
    ```

## 6 Evaluation
+ Evaluate the trained model on text classification datasets and super-natural instructions. The evaluation data can be downloaded from this [link](https://huggingface.co/t1101675/PICL/tree/main/data).
    ```bash
    bash scripts/eval/eval_cls.sh ${BASE_PATH}
    bash scripts/eval/eval_inst.sh ${BASE_PATH}
    ```

## 7 Citation
```
@inproceedings{gu2023picl,
  title={Pre-Training to Learn in Context},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle={Proceedings of ACL},
  year={2023}
}
```
