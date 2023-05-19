import os
import time
import random
import faiss
import h5py
import numpy as np
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW


from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer

from arguments import get_retrieval_args

from data_utils.indexed_dataset import make_builder

from data_utils.retriever_datasets import RetrieverDataset, RetrieverInferDataset
from modeling.retriever_modeling import RetrieverModel, get_optimizer_params


def save_log(args, log_str):
    with open(os.path.join(args.save, "log.txt"), "a") as f:
        f.write(log_str + "\n")


def compute_rank_metrics(pred_scores, target_labels, ks):
    # Compute total un_normalized avg_ranks, mrr
    values, indices = torch.sort(pred_scores, dim=1, descending=True)
    rank = 0
    mrr = 0.0
    score = {k:0 for k in ks}
    for i, idx in enumerate(target_labels):
        gold_idx = torch.nonzero(indices[i] == idx, as_tuple=False)
        rank += gold_idx.item() + 1
        for k in ks:
            score[k] += (gold_idx.item() < k)
        mrr += 1 / (gold_idx.item() + 1)
    return rank, mrr, score


def train(args, tokenizer, model, optimizer, scheduler, train_dataset, dev_dataset, train_dataloader, dev_dataloader, device):
    
    total_loss_print = 0.0
    total_loss_save = 0.0
    total_steps = 0
    all_steps = args.epochs * len(train_dataset) // (args.gradient_accumulation_steps * args.batch_size)
    for e in range(args.epochs):
        model.train()
        for model_batch in train_dataloader:
            torch.cuda.synchronize()
            st_time = time.time()
            model_batch = train_dataset.to_device(model_batch, device)
            output = model(**model_batch)
            loss = output["loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            total_loss_print += loss.item()
            total_loss_save += loss.item()
            total_steps += 1

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            def get_log_str(log_loss):
                return "Train | Epoch {:3d} | Step {:6d}/{:6d} | lr: {:.4e} | loss: {:.4f} | single step time: {:.3f}".format(
                    e,
                    total_steps,
                    all_steps,
                    scheduler.get_last_lr()[0],
                    log_loss,
                    elapsed_time
                )
            
            if total_steps % args.log_interval == 0:
                log_str = get_log_str(total_loss_print / args.log_interval)
                print(log_str)
                print(args.save)
                total_loss_print = 0
            
            if total_steps % args.save_log_interval == 0:
                log_str = get_log_str(total_loss_save / args.save_log_interval)
                save_log(args, log_str)
                total_loss_save = 0

            if total_steps % args.eval_interval == 0:
                dev_res = evaluate(args, model, tokenizer, dev_dataset, dev_dataloader, device)
                
                print("dev_res: ", dev_res)
                save_log(args, "dev_res: " + str(dev_res))

            if total_steps % args.save_interval == 0:
                save_path = os.path.join(args.save, "{}.pt".format(total_steps))
                print("save to", save_path)
                torch.save(model.state_dict(), save_path)


def evaluate(args, model, tokenizer, eval_dataset, eval_dataloader, device):
    model.eval()
    total_loss = 0.0
    step = 0
    ks = [1, 5, 10]

    total_avg_rank, total_ctx_count, total_count = 0, 0, 0
    total_mrr = 0
    total_score = {k:0 for k in ks}

    for model_batch in tqdm(eval_dataloader, desc="Evaluating"):
        model_batch = eval_dataset.to_device(model_batch, device)
        with torch.no_grad():
            output = model(**model_batch)
            
        loss = output["loss"]
        total_loss += loss.item()
        
        sim_scores = output["sim_scores"]
        query_repr = output["query_repr"]
        context_repr = output["context_repr"]
        
        rank, mrr, score = compute_rank_metrics(sim_scores, model_batch["pos_ctx_indices"], ks)
        total_avg_rank += rank
        total_mrr += mrr
        for k in ks:
            total_score[k] += score[k]
        total_ctx_count += context_repr.size(0)
        total_count += query_repr.size(0)

        step += 1
        
    eval_loss = total_loss / step
    total_ctx_count = total_ctx_count / step
    
    eval_res = {
        "loss": eval_loss,
        "avg_rank": total_avg_rank / total_count,
        "mrr": total_mrr / total_count,
        **{f"acc@{k}": total_score[k] / total_count for k in ks},
        "ctx_count": total_ctx_count
    }
    
    return eval_res


def inference(args, tokenizer, model: RetrieverModel, dataset: RetrieverInferDataset, dataloader, device):
    model.eval()

    if dist.get_rank() == 0:
        print("data_size = ", len(dataset))
        dataset.set_embeds_path(os.path.join(args.save, f"embeds.h5"))
        dataset.set_h5()

    for itr, model_batch in enumerate(tqdm(dataloader, desc="Infering", disable=(dist.get_rank()!=0))):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)

        with torch.no_grad():
            outputs = model.context_encoder(**model_batch)
        repr = outputs[1]
        
        gathered_repr = [torch.zeros_like(repr) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_repr, repr.contiguous())
        repr = torch.stack(gathered_repr, dim=1).view(-1, repr.size(-1)).cpu().numpy()
    
        if dist.get_rank() == 0:
            dataset.dump_h5(repr)
    
    if dist.get_rank() == 0:
        dataset.sum_h5()


def search(args, device):
    torch.set_grad_enabled(False)
    
    faiss.omp_set_num_threads(64)
    dim = 768

    embed_path = os.path.join(args.embed_dir, f"embeds.h5")
    
    map_path = os.path.join(args.embed_dir, "cache", "map.h5")

    print(os.path.exists(embed_path))
    print(os.path.exists(map_path))

    with h5py.File(map_path) as map_f:
        map_n2o = map_f["map_n2o"][:]
    
    print("Load text data end")
    
    with h5py.File(embed_path, "r") as f:
        embeds = f["embeds"][:]
    print("Load embeds end")
    if args.metric_type == "IP":
        cpu_index = faiss.IndexFlatIP(dim)
    elif args.metric_type == "L2":
        cpu_index = faiss.IndexFlatL2(dim)
    else:
        raise NotImplementedError
    
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = False

    gpu_index_flat = faiss.index_cpu_to_all_gpus( # build the index
        cpu_index,
        co=co
    )
    
    print("Begin add embeds")
    gpu_index_flat.add(embeds)
    print(gpu_index_flat.ntotal)
    
    search_bin_file = os.path.join(args.save, "search_icl_0.bin")
    search_idx_file = os.path.join(args.save, "search_icl_0.idx")
    search_binary_builder = make_builder(search_bin_file, impl="mmap", dtype=np.int32)

    with h5py.File(os.path.join(args.save, "scores.h5"), "w") as f:
        f.create_dataset("scores", data=np.zeros((0, 20), dtype=np.float32), maxshape=(None, None), chunks=True)

    print("Searching")
    num = args.data_num if args.data_num > 0 else len(embeds)
    bs = args.batch_size
    for st in tqdm(range(0, num, bs)):
        ed = min(st+bs, num)
        query = embeds[st:ed]
        scores, retrieved_indices = gpu_index_flat.search(query, 20)
        assert len(query) == len(scores)
        assert len(query) == len(retrieved_indices)
        
        for k, ri in enumerate(retrieved_indices):
            if args.metric_type == "l2":
                l = [int(map_n2o[k+st])] + [int(map_n2o[rr]) for rr in ri[1:]]
            else:
                if k+st == ri[0]:
                    l = [int(map_n2o[k+st])] + [int(map_n2o[rr]) for rr in ri[1:]]
                else:
                    l = [int(map_n2o[k+st])] + [int(map_n2o[rr]) for rr in ri]
                    l = l[:20]
            search_binary_builder.add_item(torch.IntTensor(l))
            
        with h5py.File(os.path.join(args.save, "scores.h5"), "a") as f:
            d = f["scores"]
            d.resize(d.shape[0] + scores.shape[0], axis=0)
            d[-len(scores):] = scores
        
    search_binary_builder.finalize(search_idx_file)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def change_save_path(args):
    if args.do_train:
        save_path = os.path.join(
            args.save,
            args.data_names.replace("/", "_"),
            f"lr{args.lr}-bs{args.batch_size}-G{args.gradient_accumulation_steps}",
        )
    elif args.do_infer:
        save_path = os.path.join(
            args.save,
            args.data_names.replace("/", "_"),
            args.ckpt_name.replace("/", "_")
        )
    else: # args.do_search
        save_path = os.path.join(
            args.save,
            args.data_names,
            args.metric_type
        )
        
    args.save = save_path
    
    return args    


def main():

    args = get_retrieval_args()
    args = change_save_path(args)
    
    if args.do_infer:
        init_distributed(args)
    
    set_random_seed(args.seed)
    
    device = torch.cuda.current_device()
    os.makedirs(args.save, exist_ok=True)

    print(args.do_search)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
    if args.do_train:
        train_dataset = RetrieverDataset(args, "train", os.path.join(args.data_dir, "train.jsonl"), tokenizer)
        valid_dataset = RetrieverDataset(args, "valid", os.path.join(args.data_dir, "valid.jsonl"), tokenizer)

        train_dataset.show_example()
        valid_dataset.show_example()

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=train_dataset.collate)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size,
                                    shuffle=False,
                                    collate_fn=valid_dataset.collate)
        print('train_size = ', len(train_dataset))
        print('valid_size = ', len(valid_dataset))
        
        model = RetrieverModel(args.model_dir, len(tokenizer), args.share_model)
        model = model.to(device)

        optimizer = AdamW(params=get_optimizer_params(args, model), lr=args.lr, eps=1e-8)
        total_steps = (len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps)) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(args.warmup_iters * total_steps),
                                                    num_training_steps=total_steps)

        if args.eval_interval == -1:
            args.eval_interval = len(train_dataset) // (args.gradient_accumulation_steps * args.batch_size)
        if args.save_interval == -1:
            args.save_interval = len(train_dataset) // (args.gradient_accumulation_steps * args.batch_size)

        train(args, tokenizer, model, optimizer, scheduler, train_dataset, valid_dataset, train_dataloader, valid_dataloader, device)
    
    if args.do_infer:            
        infer_dataset = RetrieverInferDataset(args, "infer", os.path.join(args.data_dir, args.data_names, "paragraphs"), tokenizer)
        
        model = RetrieverModel(args.model_dir, len(tokenizer), args.share_model, args.pool_type)
        if args.load is not None:
            model.load_state_dict(torch.load(args.load, map_location="cpu"))
        model = model.to(device)

        infer_dataset.show_example()

        infer_data_sampler = DistributedSampler(infer_dataset, shuffle=False, drop_last=False)
        infer_dataloader = DataLoader(infer_dataset, sampler=infer_data_sampler, batch_size=args.batch_size,
                                    collate_fn=infer_dataset.collate)
        inference(args, tokenizer, model, infer_dataset, infer_dataloader, device)
        
    if args.do_search:
        search(args, device)

if __name__ == "__main__":
    main()