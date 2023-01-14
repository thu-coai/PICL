import torch
import torch.distributed as dist
import os
import argparse
from time import sleep
from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer



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

    dist.init_process_group(backend="nccl")


def main():
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # init_distributed(args)

    # device = torch.cuda.current_device()

    # t = torch.ones(1000, 1000).to(device)

    # print("start")
    # while True:
    #     t = t * t
    #     sleep(0.001)
    
    # tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/data/yuxian/checkpoints/opt-6.7B")
    
    # print(tokenizer.encode("I love you. Me too."))

    import torch

    # the fast tokenizer currently does not work correctly
    tokenizer = AutoTokenizer.from_pretrained("/home/lidong1/data/yuxian/checkpoints/opt-2.7B", use_fast=False)

    prompt = "Hello, I'm am conscious and"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained("/home/lidong1/data/yuxian/checkpoints/opt-2.7B", torch_dtype=torch.float16)

    generated_ids = model.generate(input_ids)

    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # ["Hello, I'm am conscious and aware of my surroundings. I'm not sure what you mean"]       
        
if __name__ == "__main__":
    main()