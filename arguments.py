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
    group.add_argument('--model-dir', type=str, help='model configuration file')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--model-type", type=str, default="gpt2")
    
    return parser

def add_hp_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('hyper-parameters', 'hyper-parameters configuration')
    group.add_argument('--lr', type=float, default=1.0e-4, help='initial learning rate')
    group.add_argument('--lr-decay-iters', type=int, default=None, help='number of iterations to decay LR over')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'], help='learning rate decay function')

    group.add_argument('--batch-size', type=int, default=32, help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32, help='Data Loader batch size')
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--gradient-checkpointing", action="store_true")
    
    group.add_argument('--epochs', type=int, default=10, help='total number of epochs to train over all training runs')
    group.add_argument('--warmup-iters', type=int, default=0.01, help='percentage of data to warmup')
    group.add_argument('--train-iters', type=int, default=-1, help='total number of iterations to train over all training runs')
    
    group.add_argument('--clip-grad', type=float, default=1.0, help='gradient clipping')
    group.add_argument('--weight-decay', type=float, default=1.0e-2, help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536, help='loss scale')

    group.add_argument("--gpt-max-length", type=int, default=1024)
    group.add_argument('--max-length', type=int, default=1024, help='max length of input')
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('runtime', 'runtime configuration')
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument("--do-infer", action="store_true")
    group.add_argument('--load', type=str, default=None, help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None, help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--save-log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=1000, help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None, help='local rank passed from distributed launcher')
    group.add_argument("--num-workers", type=int, default=1)
    
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('data', 'data configuration')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--balance-eval", action="store_true")
    group.add_argument("--data-num", type=int)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--train-prompts", type=str, default=None)
    group.add_argument("--eval-prompts", type=str, default=None)
    
    group.add_argument("--raw-input", type=str, default=None)
    group.add_argument("--processed-output", type=str, default=None)
    
    return parser


def add_rng_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('random', 'random configuration')
    group.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--reset-seed-each-data", action="store_true")

    return parser


def add_pretraining_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('pre-train', 'pre-training configurations')
    
    group.add_argument("--pretrain-type", type=str)

    group.add_argument("--picl-data-dir", type=str, default=None)
    group.add_argument("--picl-idx-data-dir", type=str, default=None)
    group.add_argument("--picl-data-name", type=str, default=None)
    group.add_argument("--picl-data-prefix", type=str, default=None)
    group.add_argument("--picl-train-num", type=int, default=-1)
    group.add_argument("--picl-valid-num", type=int, default=10000)
    
    group.add_argument("--lm-train-num", type=int, default=-1)
    group.add_argument("--lm-valid-num", type=int, default=10000)
    group.add_argument("--lm-ratio", type=float, default=None)
    group.add_argument("--lm-data-dir", type=str, default=None)
    group.add_argument("--lm-data-name", type=str, default="lm")
    group.add_argument("--lm-data-prefix", type=str, default=None)
    
    return parser


def add_icl_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('in context learning', 'in context learning configurations')
    group.add_argument("--shot", type=int, default=16)
    group.add_argument("--flan-sample-max", type=int, default=3000)
    group.add_argument("--norm-option-loss", action="store_true")
    group.add_argument("--icl-pool-num", type=int, default=10000)
    group.add_argument("--icl-share-demo", action="store_true")
    group.add_argument("--icl-balance", action="store_true")
    group.add_argument("--max-length-per-sample", type=int, default=256)
    group.add_argument("--max-length-all-demos", type=int, default=-1)
    group.add_argument("--rand-shot-num", action="store_true")
    group.add_argument("--icl-sup", type=str, default="all_target")
    group.add_argument("--sni-ref-file", default=None)
    
    return parser


def add_filter_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('filter', 'in context learning configurations')
    group.add_argument("--do-filter", action="store_true")
    group.add_argument("--filter-threshold", type=float, default=0.0)
    group.add_argument("--filter-num", type=int, default=-1)
    group.add_argument("--filter-ratio", type=float, default=1.0)
    group.add_argument("--score-small", action="store_true")
    group.add_argument("--score-icl", action="store_true")
    group.add_argument("--score-zero", action="store_true")

    return parser


def add_retriever_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('retriever', 'retriever configurations')

    # runtime
    group.add_argument("--do-search", action="store_true")
    
    # data args
    group.add_argument("--ret-pos-num", type=int, default=1)
    group.add_argument("--ret-easy-neg-num", type=int, default=1)
    group.add_argument("--ret-hard-neg-num", type=int, default=4)
    group.add_argument("--ret-train-num-per-prompt", type=int, default=1000)
    group.add_argument("--ret-eval-num-per-prompt", type=int, default=100)
    group.add_argument("--ret-source-split", type=str, default="train")

    # model args
    group.add_argument("--share-model", action="store_true")
    group.add_argument("--pool-type", default="cls")
    
    # search args
    group.add_argument("--embed-dir", type=str, default=None)
    group.add_argument("--metric-type", type=str, default="l2")
    group.add_argument("--search-k", type=int, default=10)

    return parser


############## get args ################

def get_picl_pretrain_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_hp_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_rng_args(parser)
    parser = add_pretraining_args(parser)
    parser = add_icl_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            
    save_path = os.path.join(
        args.save,
        args.pretrain_type,
        args.picl_data_prefix.replace("/", "_") + ("" if args.train_num < 0 else numerize(args.train_num)),
        (args.lm_data_prefix or "").replace("/", "_") + str(args.lm_ratio or ""),
        "balance_eval" if args.balance_eval else "",
        f"shot{args.shot}",
        f"lr{args.lr}-bs{args.batch_size}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-wm{args.warmup_iters}",
        f"len{args.max_length_per_sample}-{args.max_length}",
        args.ckpt_name,
        f"{args.seed}-{args.seed_order}-{args.seed_data}",
    )
    
    args.save = save_path
    
    return args


def get_picl_eval_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_hp_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_rng_args(parser)
    parser = add_icl_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            
    save_path = os.path.join(
        args.save,
        args.data_names,
        f"shot{args.shot}",
        f"len{args.max_length_per_sample}-{args.max_length}",
        (args.ckpt_name or "").replace("/", "_"),
        f"{args.seed}-{args.seed_order}-{args.seed_data}",
    )
    
    args.save = save_path
    
    return args


def get_retrieval_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_hp_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_rng_args(parser)
    parser = add_retriever_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    return args


def get_filter_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_hp_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_rng_args(parser)
    parser = add_pretraining_args(parser)
    parser = add_icl_args(parser)
    parser = add_filter_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    save_path = os.path.join(
        args.save,
        args.picl_data_prefix.replace("/", "_")
    )
    
    args.save = save_path

    return args