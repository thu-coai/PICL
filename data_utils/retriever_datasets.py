import json
import pickle
import os
import sys
import h5py
import time
import multiprocessing
from tqdm import tqdm
import numpy as np
from itertools import chain

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import RobertaTokenizer

from data_utils.indexed_dataset import make_builder
from data_utils.distributed_indexed import DistributedMMapIndexedDataset


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.max_len = args.max_length

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = RobertaTokenizer.from_pretrained(self.args.model_dir)

    def encode_train(self, line_str):
        line = json.loads(line_str)
        query = Encoder.tokenizer.encode(line["query"], max_length=self.max_len, truncation=True)
        pos_context = Encoder.tokenizer.encode(line["pos_context"], max_length=self.max_len, truncation=True)
        easy_neg_contexts = [Encoder.tokenizer.encode(neg_ctx, max_length=self.max_len, truncation=True) for neg_ctx in line["easy_neg_contexts"]]
        hard_neg_contexts = [Encoder.tokenizer.encode(neg_ctx, max_length=self.max_len, truncation=True) for neg_ctx in line["hard_neg_contexts"]]
        
        d = {
            "query": query,
            "pos_context": pos_context,
            "easy_neg_contexts": easy_neg_contexts,
            "hard_neg_contexts": hard_neg_contexts,
            "label": line["label"],
            "easy_neg_labels": line["easy_neg_labels"],
            "hard_neg_labels": line["hard_neg_labels"],
        }

        return d, line_str, len(line_str)
    
    def encode_infer(self, line):
        oid, line = line
        line = line.strip()
        line = line.replace("<@x(x!>", "\n")
        d = self.tokenizer.encode(line.strip(), max_length=self.max_len, truncation=True)
        line = line.replace("\n", "<@x(x!>")
        return oid, d, len(line)


class RetrieverDataset(Dataset):
    def __init__(self, args, split, path, tokenizer: RobertaTokenizer):
        super().__init__()

        self.args = args
        self.split = split
        self.max_len = args.max_length
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        
        cache_dir = os.path.join(args.data_dir, split, str(self.max_len), args.ckpt_name)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            if args.data_process_workers <= 0:
                self.data = self.load_data(path)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.data, f)
            else:
                self.data, data_reordered = self.load_data_parallel(path, args.data_process_workers)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.data, f)
                    
                with open(cache_path.replace(".pkl", "_reordered.jsonl"), "w") as f:
                    for line in data_reordered:
                        f.write(line)

    def load_data_parallel(self, path, workers):
        data = []
        data_reordered = []
        startup_start = time.time()
        fin = open(path, "r", encoding='utf-8')
        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode_train, fin, 10)
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for lid, (d, line, bytes_processed) in enumerate(encoded_docs, start=1):

            total_bytes_processed += bytes_processed
            if d is None:
                continue

            data.append(d)
            data_reordered.append(line)

            if lid % 10000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        pool.close()
        fin.close()
        
        return data, data_reordered

    def load_data(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"Processing {self.split} data"):
            line = json.loads(line)
            query = self.tokenizer.encode(line["query"], max_length=self.max_len, truncation=True)
            pos_context = self.tokenizer.encode(line["pos_context"], max_length=self.max_len, truncation=True)
            easy_neg_contexts = [self.tokenizer.encode(neg_ctx, max_length=self.max_len, truncation=True) for neg_ctx in line["easy_neg_contexts"]]
            hard_neg_contexts = [self.tokenizer.encode(neg_ctx, max_length=self.max_len, truncation=True) for neg_ctx in line["hard_neg_contexts"]]

            data.append({
                "query": query,
                "pos_context": pos_context,
                "easy_neg_contexts": easy_neg_contexts,
                "hard_neg_contexts": hard_neg_contexts,
                "label": line["label"],
                "easy_neg_labels": line["easy_neg_labels"],
                "hard_neg_labels": line["hard_neg_labels"],
            })
            
        return data  

    def __len__(self):
        return len(self.data)

    def show_example(self):
        print('data example:')
        tmp = [self.__getitem__(i) for i in range(4)]
        for t in tmp:
            print(t)
            print("**QUERY:**", self.tokenizer.decode(t["query"]))
            print("**POS CTX:**", self.tokenizer.decode(t["pos_context"]))
            print("**EASY NEG:**", self.tokenizer.decode(t["easy_neg_contexts"][0]))
            if len(t["hard_neg_contexts"]) > 0:
                print("**HARD NEG:**", self.tokenizer.decode(t["hard_neg_contexts"][0]))

        print(self.collate(tmp))
        

    def __getitem__(self, idx):
        return self.data[idx]
    
    def to_device(self, model_batch, device):
        for k in model_batch["query_inputs"]:
            model_batch["query_inputs"][k] = model_batch["query_inputs"][k].to(device)
        for k in model_batch["context_inputs"]:
            model_batch["context_inputs"][k] = model_batch["context_inputs"][k].to(device)
        model_batch["pos_ctx_indices"] = model_batch["pos_ctx_indices"].to(device)
        model_batch["pos_ctx_indices_mask"] = model_batch["pos_ctx_indices_mask"].to(device)

        return model_batch
    
    def collate(self, samples):
        bs = len(samples)
        model_batch = {
            "query_inputs": {
                "input_ids": [],
                "attention_mask": [],
            },
            "context_inputs": {
                "input_ids": [],
                "attention_mask": [],
            },
            "pos_ctx_indices": [],
            "pos_ctx_indices_mask": []
        }
        
        context_labels = list(chain(*[[s["label"]] + s["easy_neg_labels"] + s["hard_neg_labels"] for s in samples]))
        
        for s in samples:
            model_batch["query_inputs"]["input_ids"].append(s["query"] + [self.pad_id] * (self.max_len - len(s["query"])))
            model_batch["query_inputs"]["attention_mask"].append([1] * len(s["query"]) + [0] * (self.max_len - len(s["query"])))
            model_batch["pos_ctx_indices"].append(len(model_batch["context_inputs"]["input_ids"]))
            contexts = [s["pos_context"]] + s["easy_neg_contexts"] + s["hard_neg_contexts"]
            for ctx in contexts:
                model_batch["context_inputs"]["input_ids"].append(ctx + [self.pad_id] * (self.max_len - len(ctx)))
                model_batch["context_inputs"]["attention_mask"].append([1] * len(ctx) + [0] * (self.max_len - len(ctx)))
            model_batch["pos_ctx_indices_mask"].append([int(s["label"] == context_label) for context_label in context_labels])

        for k in model_batch["query_inputs"]:
            model_batch["query_inputs"][k] = torch.tensor(model_batch["query_inputs"][k], dtype=torch.long)
        for k in model_batch["context_inputs"]:
            model_batch["context_inputs"][k] = torch.tensor(model_batch["context_inputs"][k], dtype=torch.long)
        model_batch["pos_ctx_indices"] = torch.tensor(model_batch["pos_ctx_indices"], dtype=torch.long)
        model_batch["pos_ctx_indices_mask"] = torch.tensor(model_batch["pos_ctx_indices_mask"], dtype=torch.float32)

        return model_batch


class RetrieverInferDataset(Dataset):
    def __init__(self, args, split, path, tokenizer: RobertaTokenizer):
        super().__init__()

        self.args = args
        self.split = split
        self.max_len = args.max_length
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id

        self.cache_dir = os.path.join(args.save, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_ok_path = os.path.join(self.cache_dir, "OK")
                
        if not os.path.exists(cache_ok_path):
            if dist.get_rank() == 0:
                if args.data_process_workers <= 0:
                    self.load_data(path, self.cache_dir)
                else:
                    print("Process at rank 0")
                    self.load_data_parallel(path, self.cache_dir, args.data_process_workers)
                with open(cache_ok_path, "w") as f:
                    f.write("OK")
            dist.barrier()
            # reload data to ensure the data are the same in all processes
        
        self.ctx = DistributedMMapIndexedDataset(self.cache_dir, f"data", dist.get_rank(), dist.get_world_size())

    def get_np_map(self, map_ids, max_num):
        o2n = np.zeros(max_num, dtype=np.int32)
        n2o = np.zeros(max_num, dtype=np.int32)
        for oid, nid in map_ids:
            o2n[oid] = nid
            n2o[nid] = oid
        return o2n, n2o    

    def load_data_parallel(self, path, cache_dir, workers):
        startup_start = time.time()
        fin = open(path, "r", encoding='utf-8')
        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode_infer, enumerate(fin), 10)
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        data_bin_file = os.path.join(cache_dir, f"data_0.bin")
        data_idx_file = os.path.join(cache_dir, f"data_0.idx")
        data_binary_builder = make_builder(data_bin_file, impl="mmap", dtype=np.uint16)

        nid = 0
        o_docs_num = 0
        map_ids = []
        for lid, (oid, d, bytes_processed) in enumerate(encoded_docs):
            o_docs_num += 1
            total_bytes_processed += bytes_processed
            if d is None:
                continue
                            
            data_binary_builder.add_item(torch.IntTensor(d))
            
            map_ids.append((oid, nid))
            
            nid += 1
            if lid % 10000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents",
                      f"({lid/elapsed} docs/s, {mbs} MB/s).",
                      file=sys.stderr)
                        
        data_binary_builder.finalize(data_idx_file)
        o2n, n2o = self.get_np_map(map_ids, o_docs_num)
        with h5py.File(os.path.join(cache_dir, f"map.h5"), "w") as h5_f:
            h5_f.create_dataset("map_o2n", data=o2n, dtype=np.int32, chunks=True)
            h5_f.create_dataset("map_n2o", data=n2o, dtype=np.int32, chunks=True)

        pool.close()
        fin.close()

    def load_data(self, path):
        raise NotImplementedError

    def __len__(self):
        assert self.ctx is not None
        return len(self.ctx)

    def show_example(self):
        if dist.get_rank() == 0:
            print('data example:')
            for i in range(4):
                tmp = self.__getitem__(i)
                print(tmp)
                print((self.tokenizer.decode(tmp)).replace("\n", "<n>"))
                print("#" * 50)

    def __getitem__(self, idx):
        data = self.ctx[idx]
        input_ids = data.astype(int)
        return input_ids.tolist()
    
    def to_device(self, model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)

        return model_batch
    
    def collate(self, samples):
        bs = len(samples)
        model_batch = {
            "input_ids": [],
            "attention_mask": [],
        }
        
        for s in samples:
            model_batch["input_ids"].append(s + [self.pad_id] * (self.max_len - len(s)))
            model_batch["attention_mask"].append([1] * len(s) + [0] * (self.max_len - len(s)))

        for k in model_batch:
            model_batch[k] = torch.tensor(model_batch[k], dtype=torch.long)
        
        return model_batch

    def set_embeds_path(self, path):
        self.embeds_path = path
    
    def set_h5(self):
        with h5py.File(self.embeds_path, "w") as f:
            f.create_dataset("embeds", data=np.zeros((0, 768), dtype=np.float32), maxshape=(None, None), chunks=True)
        
    def dump_h5(self, embeddings):
        assert os.path.exists(self.embeds_path)
        with h5py.File(self.embeds_path, "a") as f:
            d = f["embeds"]
            d.resize(d.shape[0] + embeddings.shape[0], axis=0)
            d[-len(embeddings):] = embeddings
            
    def sum_h5(self):
        with h5py.File(self.embeds_path, "r+") as f:
            s_origin = f["embeds"].shape
            f["embeds"].resize(self.__len__(), axis=0)
            s_resize = f["embeds"].shape
            print(f"Dumped to {self.embeds_path}, origin size = {s_origin}, resize = {s_resize}")
