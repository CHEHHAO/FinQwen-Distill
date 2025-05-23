import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer    
from utils.distill_uitls import DistillConfig, DistillTrainer


WANDB = False
model_path = "Qwen/Qwen3-0.6B"
output_dir = "output"
teacher_model_path = "Qwen/Qwen-7B"


model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True)
teacher_model = 