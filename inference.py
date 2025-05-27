import os 
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm  
# from utils.utils import format_to_sft


# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'


def format_to_sft(example):
    """
    Formats a single example into the SFT prompt-target pair for Qwen2.5 NER fine-tuning.

    Args:
        example (dict): A dict with keys:
            - title (str): the title of the text.
            - text (str): the input text to label.
            - entities (List[str]): list of company name entities in the text.

    Returns:
        str: A string containing the full chat-style prompt and the JSON array of entities.
    """
    # System prompt defining the NER task
    SYSTEM_PROMPT = (
        "You are a named-entity recognition assistant. Identify all company names "
        "in the following title and text, and output a JSON array of strings, "
        "each string being one company name."
    )

    # ChatML instruction delimiters
    inst_start = "<|im_start|>[INST] <<SYS>>\n"
    inst_end = "<</SYS>>\nTitle: \"{title}\"\nText: \"{text}\"\n[/INST]\n"
    inst_close = "<|im_end|>"

    # Build the JSON array of entities
    entities = example.get("entities", [])
    output_json = json.dumps(entities, ensure_ascii=False)

    # Compose final SFT example including title and text
    prompt = (
        f"{inst_start}"
        f"{SYSTEM_PROMPT}\n"
        f"{inst_end.format(title=example['title'], text=example['text'])}"
        # f"{output_json}\n"
        f"{inst_close}"
    )
    return prompt


# Example usage:
if __name__ == "__main__":
#     example = {
#         "title": "Microsoft Acquires GitHub",
#         "text": "Microsoft announced today that it has acquired GitHub, and will integrate GitHub Copilot into its Azure cloud platform.",
#         "entities": ["Microsoft", "GitHub", "GitHub Copilot", "Azure"]
#     }
#     print(format_to_sft(example))

# exit()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def batch_inference(
    model_path: str,
    examples_path: str,
    output_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8
):
    # 1. 加载模型与分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 2. 使用 text-generation 管道
    gen_pipe = pipeline(
        "text-generation",
        model=model.eval(),
        tokenizer=tokenizer,
        device_map='auto'
    )

    # 3. 加载示例
    with open(examples_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    predictions = []
    total = len(examples)

    # 4. 批量推理，添加进度条
    for i in tqdm(range(0, total, batch_size), desc="Batch Inference"):
        batch = examples[i:i+batch_size]
        prompts = [format_to_sft(ex) for ex in batch]

        batch_outputs = gen_pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

        # 处理输出: gen_pipe 对每个 prompt 返回一个列表，取第一个结果
        for prompt, outputs in zip(prompts, batch_outputs):
            # outputs is a list of generation dicts
            out = outputs[0] if isinstance(outputs, list) else outputs
            full = out['generated_text']
            gen_text = full[len(prompt):].strip()
            try:
                entities = json.loads(gen_text)
            except json.JSONDecodeError:
                print(f"Warning: JSON parse failed for output: {gen_text}")
                entities = []
            predictions.append(entities)

    # 5. 保存到文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(predictions, fout, ensure_ascii=False, indent=2)
    print(f"Batch inference done: {len(predictions)} examples saved to {output_path}")


if __name__ == '__main__':
    MODEL_PATH = "/data/mengao/models/Qwen/Qwen2.5-7B-Instruct"
    EXAMPLES_PATH = "data/clean_data/sft_test.json"
    OUTPUT_PATH = "data/generated/responses.json"
    batch_inference(
        MODEL_PATH,
        EXAMPLES_PATH,
        OUTPUT_PATH,
        batch_size=4,
        max_new_tokens=512,
        temperature=0.7,
        top_k=20,
        top_p=0.8
    ) 