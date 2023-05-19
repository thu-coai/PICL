"""Processing data"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import multiprocessing
import time
import torch
import random
import numpy as np
import random
from transformers import RobertaTokenizer
from numerize import numerize

import argparse
from arguments import add_data_args, add_runtime_args, add_model_config_args, add_hp_args

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = RobertaTokenizer.from_pretrained(self.args.model_dir)

    def encode(self, doc):
        if len(doc) < 20:
            return None, len(doc)

        paras = doc.split("<@x(x!>")
        valid_paras = []
        if self.args.max_length > 0:
            tmp = []
            tmp_tokens_count = 0
            for para in paras:
                para = para.strip()
                if len(para) == 0:
                    continue

                tokens = Encoder.tokenizer.encode(para)
                
                if tmp_tokens_count + len(tokens) < self.args.max_length:
                    tmp.append(para)
                    tmp_tokens_count += len(tokens)
                else:
                    if len(tmp) == 0 or tmp[-1][-1] in [":", "?", "：", "？"]:
                        tmp.append(para)
                        tmp_tokens_count += len(tokens)
                    else:
                        valid_paras.append(tmp)
                        tmp = [para]
                        tmp_tokens_count = len(tokens)
            if len(tmp) != 0:
                valid_paras.append(tmp)
        else:
            valid_paras = [[x.strip()] for x in paras]               
        
        valid_paras = ["<@x(x!>".join(para) for para in valid_paras]            

        return valid_paras, len(doc)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser = add_hp_args(add_model_config_args(add_data_args(add_runtime_args(parser))))
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.raw_input)
    # output_path = os.path.join(args.output_path, f"full_{args.max_length}{r2s}")
    output_path = os.path.join(args.processed_output, f"{numerize.numerize(args.data_num)}_{args.max_length}")
    os.makedirs(output_path, exist_ok=True)
    
    fin = open(args.raw_input, 'r', encoding="utf-8")
    fout = open(os.path.join(output_path, "paragraphs_dup"), "w", encoding="utf-8")

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.data_process_workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    sid = 0
    for lid, (paras, bytes_processed) in enumerate(encoded_docs, start=1):
        
        total_bytes_processed += bytes_processed
        if paras is None:
            continue

        for para in paras:
            fout.write((para + "\n"))
            sid += 1
            
        if lid % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents",
                  f"({lid/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
            
        if args.data_num > 0 and sid >= args.data_num:
            break

    pool.close()
    fout.close()
    fin.close()
    
    print("Shuffling")
    os.system("sort {} | uniq | shuf > {}".format(os.path.join(output_path, "paragraphs_dup"), os.path.join(output_path, "paragraphs")))

if __name__ == '__main__':
    main()