# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
from numerize.numerize import numerize


def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-config', type=str, 
                       help='model configuration file')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--n-gpu", type=int, default=1)
    
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None,
                       help='Path to the project base directory.')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--balance-eval", action="store_true")
    group.add_argument("--trim", action="store_true")
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--train-lm-num", type=int, default=-1)
    group.add_argument("--dev-lm-num", type=int, default=-1)
    group.add_argument('--dataset-name', type=str, default=None,
                       help='Name of the dataset')
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-name', type=str, default=None,
                       help='Output filename to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--inspect-iters', type=int, default=1000,
                       help='number of inspecting')
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=-1,
                       help='total number of iterations to train over all training runs')
    group.add_argument("--gpt-max-length", type=int, default=1024)
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--max-encoder-length', type=int, default=512,
                       help='max length of encoder input')
    group.add_argument('--max-decoder-length', type=int, default=256,
                       help='max length of decoder input')
    group.add_argument('--start-step', type=int, default=0,
                       help='step to start or continue training')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--reset-seed-each-data", action="store_true")

    group.add_argument('--epochs', type=int, default=10,
                       help='total number of epochs to train over all training runs')
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    group.add_argument("--no-extend-save-path", action="store_true")
    

    # Learning rate.
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')

    group.add_argument('--warmup-iters', type=int, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--output-attentions", action="store_true")
    
    return parser


def add_icl_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('train', 'in context learning configurations')
    group.add_argument("--shot", type=int, default=16)
    group.add_argument("--flan-sample-max", type=int, default=3000)
    group.add_argument("--norm-option-loss", action="store_true")
    group.add_argument("--icl-pool-num", type=int, default=10000)
    group.add_argument("--eval-test", action="store_true", help="For debugging two evaluate fucntions")
    group.add_argument("--icl-share-demo", action="store_true")
    group.add_argument("--icl-bag-of-inst", action="store_true")
    group.add_argument("--icl-balance", action="store_true")
    group.add_argument("--max-length-per-sample", type=int, default=256)
    group.add_argument("--max-length-all-demos", type=int, default=-1)
    group.add_argument("--icl-inner-batch-size", type=int, default=32)
    group.add_argument("--rand-shot-num", action="store_true")
    group.add_argument("--train-prompts", type=str, default=None)
    group.add_argument("--eval-prompts", type=str, default=None)
    group.add_argument("--pos-type", type=int, default=0)
    group.add_argument("--remove-inner-bos", action="store_true")
    group.add_argument("--icl-sup", type=str, default="test_target")
    group.add_argument("--sup-start-pos", type=int, default=None)
    group.add_argument("--sup-start-ratio", type=float, default=None)
    group.add_argument("--add-bos", action="store_true")
    group.add_argument("--end-token", type=str, default="<n>")
    group.add_argument("--icl-demo-len", type=int, default=None)
    group.add_argument("--type", type=str, default=None)
    group.add_argument("--icl-many-in-model", action="store_true")
    group.add_argument("--attn-scale", action="store_true")
    group.add_argument("--chunk-len", type=int, default=-1)
    group.add_argument("--lm-ratio", type=float, default=None)
    group.add_argument("--end-lm-ratio", type=float, default=None)
    group.add_argument("--lm-data-dir", type=str, default=None)
    group.add_argument("--lm-data-prefix", type=str, default=None)
    group.add_argument("--lm-data-name", type=str, default="lm")
    group.add_argument("--lm-only", action="store_true")
    group.add_argument("--eval-mixed", action="store_true")
    group.add_argument("--optim-batch", action="store_true")
    group.add_argument("--pretrain-type", type=str)
    group.add_argument("--ni-ref-file", default=None)
    group.add_argument("--icl-qa-ratio", type=float, default=None)
    
    return parser


def add_unsup_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('preprocess', 'unsupervised data processing')
    group.add_argument("--unsup-data-path", type=str, default=None)
    group.add_argument("--unsup-data-name", type=str, default=None)
    group.add_argument("--unsup-data-prefix", type=str, default=None)
    group.add_argument("--unsup-data-max-num", type=int, default=-1)
    group.add_argument("--processed-unsup-data-path", type=str, default=None)
    group.add_argument("--unsup-data-process-workers", type=int, default=64)
    group.add_argument("--unsup-valid-num", type=int, default=10000)
    group.add_argument("--replace-return-with-space", action="store_true")
    group.add_argument("--retain-rn", action="store_true")
    group.add_argument("--stuffed", action="store_true")
    group.add_argument("--split-stuffed", action="store_true")
    group.add_argument("--no-stuffed-no-waste", action="store_true")
    group.add_argument("--filter-num", type=int, default=-1)
    group.add_argument("--filter-ratio", type=float, default=1.0)
    group.add_argument("--score-small", action="store_true")
    group.add_argument("--score-icl", action="store_true")
    group.add_argument("--score-zero", action="store_true")
    group.add_argument("--unsup-fast", action="store_true")
    group.add_argument("--unsup-data-clip-begin", action="store_true")
    
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_icl_args(parser)
    parser = add_unsup_data_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if args.max_length_all_demos < 0:
       args.max_length_all_demos = None
    
    if args.max_length < 0:
        args.max_length = None 
    
    assert args.max_length_per_sample is not None, "max_length_per_sample should not be None"
    assert int(args.max_length_all_demos is None) + int(args.max_length is None) == 1, "one and only one of max_length_all_demos and max_length should be None"
    
    if not args.no_extend_save_path:
        r2s = "_r2s" if args.replace_return_with_space else ""
        trim = "_trim" if args.trim else ""
        save_path = os.path.join(
            args.save,
            (args.pretrain_type or "") + (f"_qa{args.icl_qa_ratio}" if args.icl_qa_ratio is not None else ""),
            args.data_names or "",
            (args.unsup_data_prefix or "").replace("/", "_") + ("" if args.train_num < 0 else numerize(args.train_num)),
            (args.lm_data_prefix or "").replace("/", "_") + str(args.lm_ratio or "") + str(args.end_lm_ratio or "") + ("_only" if args.lm_only else ""),
            "balance_eval" if args.balance_eval else "",
            args.type,
            args.icl_sup + str(args.sup_start_pos or "") + str(args.sup_start_ratio or ""),
            "add_bos" if args.add_bos else "",
            "remove_inner_bos" if args.remove_inner_bos else "",
            "attn_scale" if args.attn_scale else "",
            f"chunk{args.chunk_len}",
            f"pos{args.pos_type}_{args.end_token}{r2s}{trim}",
            f"shot{args.shot}",
            f"lr{args.lr}-bs{args.batch_size}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-wm{args.warmup_iters}" if args.do_train else "",
            f"len{args.max_length_per_sample}-{args.max_length}-{args.max_length_all_demos}",
            f"attn_dtype-{args.attn_dtype}" if args.attn_dtype is not None else "",
            (args.ckpt_name or "").replace("/", "_"),
            f"{args.seed}-{args.seed_order}-{args.seed_data}",
        )
        
        save_path = save_path.replace("roberta-base_HR_pos1_easy_neg1_hard_neg1_seed42_concate32_bs64_32_lr0.00005_G1_SEED_4000.pt_l2_h5_res_-1", "roberta-bHR_en1_hn1_cat32_bs64_lr0.00005_G1_4000.pt")
        save_path = save_path.replace("roberta-base_HR_pos1_easy_neg1_hard_neg1_seed42_concate32_bs64_32_lr0.00005_G1_SEED_4000.pt", "SEARCH1")
        save_path = save_path.replace("roberta-bHR_en1_hn1_cat32_bs64_lr0.00005_G1_4000.pt_256_1024_-1", "SEARCH1")
        args.save = save_path
    
    return args
