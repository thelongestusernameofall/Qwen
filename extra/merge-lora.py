#!/usr/bin/env python
# coding=utf-8
import argparse

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def merge(lora_path: str, out_path: str, shard_size: str = "2048MB"):
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,  # path to the output directory
        trust_remote_code=True
    )

    tokenizer.save_pretrained(out_path)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_path, max_shard_size=shard_size, safe_serialization=True)


if __name__ == '__main__':
    argsparser = argparse.ArgumentParser("merge lora model")
    argsparser.add_argument("-l", "--lora_path", type=str, required=True, help="lora model path")
    argsparser.add_argument("-o", "--out_path", type=str, required=True, help="output path")
    argsparser.add_argument("-s", "--shard_size", type=str, default="2048MB", help="shard size")

    args = argsparser.parse_args()

    merge(args.lora_path, args.out_path, args.shard_size)
