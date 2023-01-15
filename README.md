# PICL: Pre-Training to Learn in Context

## 1 Install
```
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

## 2 Run