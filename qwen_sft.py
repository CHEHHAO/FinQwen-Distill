import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from utils.utils import find_files,formatting_prompts_func,print_trainable_parameters


# model_name = "Qwen/Qwen2.5-7B-Instruct"

model_path = "/data/mengao/models/models--Qwen--Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True)


