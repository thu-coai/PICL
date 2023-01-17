import os
import h5py
import json
import time
import numpy as np
import random
import deepspeed
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn

from arguments import get_filter_args

from data_utils.pretrain_datasets import ICLPretrainDataset
from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from data_utils.indexed_dataset import make_builder
from transformers import AutoModelForCausalLM, GPT2Tokenizer

from utils import set_random_seed, print_args
from utils import print_rank, save_rank, get_rank
from tqdm import tqdm

num_threads = 4
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
torch.set_num_threads(num_threads)


def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)

    return tokenizer


def get_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    return model


def setup_model(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)

    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=None,
        args=args,
        lr_scheduler=None,
        config_params=ds_config
    )

    return model


def prepare_dataset(args, tokenizer, rank, world_size):
    rng_sample = random.Random(args.seed)
    dataset = ICLPretrainDataset(
        args,
        tokenizer,
        args.picl_data_dir,
        path_icl_idx=args.picl_idx_data_dir,
        split="search",
        num=args.filter_num, 
        shot=args.shot,
        mode="icl",
        rng_sample=rng_sample)
    
    return dataset


def score(args, tokenizer, model, dataset: ICLPretrainDataset, device, mode="icl"):

    print_rank("Scoring Mode:", mode)
    
    collate_fn = dataset.collate if mode in ["icl", "lm"] else dataset.collate_zs

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model.eval()
    all_avg_loss = []
    step = 0

    ckpt_name = (args.ckpt_name).replace("/", "_")
    score_file_name = os.path.join(args.save, f"score_{mode}_{ckpt_name}.h5")

    if dist.get_rank() == 0:
        dataset.set_h5(score_file_name, "score")

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch).logits
            loss = loss_func(
                logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            loss = loss.view(no_model_batch["loss_mask"].size())
            loss = loss * no_model_batch["loss_mask"]

            avg_loss = torch.sum(loss, dim=-1) / torch.sum(no_model_batch["loss_mask"], dim=-1)
            all_avg_loss.extend(avg_loss.cpu().tolist())

            gathered_losses = [torch.zeros_like(
                loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss)
            gathered_losses = torch.stack(
                gathered_losses, dim=1).view(-1, loss.size(-1)).cpu().half().numpy()

            if dist.get_rank() == 0:
                dataset.dump_h5(score_file_name, "score", gathered_losses)

            step += 1

    if dist.get_rank() == 0:
        dataset.sum_h5(score_file_name, "score")

    return np.mean(all_avg_loss)


def filter(args):
    print_rank("Filtering")

    threshold = args.filter_threshold
    
    ctx = DistributedMMapIndexedDataset(args.picl_idx_data_dir, f"search_icl", 0, 1)

    name_base = "score_zs_gpt2-large"
    name_ours = "score_icl_gpt2-large"

    print("Loading scores")
    with h5py.File(os.path.join(args.save, f"{name_base}.h5"), "r") as f:
        scores_base = f["score"][:]

    with h5py.File(os.path.join(args.save, f"{name_ours}.h5"), "r") as f:
        scores_ours = f["score"][:]
    print("Score load end")

    print((len(scores_base), len(ctx)))
    print((len(scores_ours), len(ctx)))

    os.makedirs(os.path.join(args.save, f"filtered_{threshold}"), exist_ok=True)
    bin_file = os.path.join(args.save, f"filtered_{threshold}", "filtered_0.bin")
    idx_file = os.path.join(args.save, f"filtered_{threshold}", "filtered_0.idx")

    binary_builder = make_builder(bin_file, impl="mmap", dtype=np.int32)

    n = 0
    for idx in tqdm(range(len(scores_base))):
        ids = ctx[idx].astype(int).tolist()
        s_base = scores_base[idx]
        s_ours = scores_ours[idx]
        
        mask = (s_base > 0.0001)
                
        avg_s_base = np.sum(s_base * mask) / np.sum(mask)
        avg_s_ours = np.sum(s_ours * mask) / np.sum(mask)
        
        if avg_s_ours - avg_s_base < -threshold:
            # print(avg_s_ours, avg_s_base)
            binary_builder.add_item(torch.IntTensor(ids))

            n += 1

    print(n, len(scores_base))
    binary_builder.finalize(idx_file)


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


def initialize():
    # get arguments
    args = get_filter_args()
    # init bmt
    init_distributed(args)
    set_random_seed(args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def gather(item, device):
    t = torch.tensor(item, device=device)
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return gathered
    

def main():
    torch.backends.cudnn.enabled = False
    
    args = initialize()
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["zero_optimization"]["stage"] = 0
    
    tokenizer = get_tokenizer(args)
    
    dataset = prepare_dataset(
        args,
        tokenizer,
        dist.get_rank(), dist.get_world_size(),
    )
    
    model = setup_model(args, ds_config, device, set_optim=args.do_train)
    if args.score_icl:
        avg_icl_loss = score(args, tokenizer, model, dataset, device, "icl")
        all_avg_icl_loss = gather(avg_icl_loss, device)
        all_avg_icl_loss = [x.item() for x in all_avg_icl_loss]
        if dist.get_rank() == 0:
            print("All ICL Loss", all_avg_icl_loss, "Avg.", sum(all_avg_icl_loss) / len(all_avg_icl_loss))

    if args.score_zero:
        avg_zs_loss = score(args, tokenizer, model, dataset, device, "zs")
        all_avg_zs_loss = gather(avg_zs_loss, device)
        all_avg_zs_loss = [x.item() for x in all_avg_zs_loss]
        if dist.get_rank() == 0:
            print("All ZS Loss", all_avg_zs_loss, "Avg.", sum(all_avg_zs_loss) / len(all_avg_zs_loss))

    if args.do_filter:
        if get_rank() == 0:
            filter(args)

if __name__ == "__main__":
    main()