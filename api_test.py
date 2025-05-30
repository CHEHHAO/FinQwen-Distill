# inference 
# from openai import OpenAI


# client = OpenAI(api_key="0",base_url="http://127.0.0.1:8000/v1")
# message = [
#     {
#     "role": "system",
#     "content": (
#     "You are a named-entity recognition assistant. Identify all company names in the following title and text, "
#     "and output an array of strings, each string being one company name."
#     )
#     },
#     {
#     "role": "user",
#     "content": f"Title: \"Microsoft Acquires GitHub\"\nText: \"Microsoft announced today that it has acquired GitHub, and will integrate GitHub Copilot into its Azure cloud platform.\""
#     }
# ]
# # message = [{"role": "user", "content":"who are you?"}]
# result = client.chat.completions.create(messages=message, model="Qwen/Qwen2.5-7B-Instruct")
# print(result.choices[0].message.content)


import os
import json
import time
from tqdm import tqdm # 进度条
from openai import OpenAI
from openai import APIConnectionError, OpenAIError

# 配置 OpenAI-compatible API
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = os.environ.get("API_PORT", "8000")
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "0"), base_url=f"http://{API_HOST}:{API_PORT}/v1")

# 输入/输出文件
INPUT_FILE = "/home/svu/idfv22/dev/FinQwen-Distill/data/clean_data/sft_test.json"
OUTPUT_FILE = "/home/svu/idfv22/dev/FinQwen-Distill/data/generated/fin_r1_responses.json"
# OUTPUT_FILE = "/home/svu/idfv22/dev/FinQwen-Distill/data/generated/baseline.json"

# 最大重试次数 & 延迟
MAX_RETRIES = 3
RETRY_DELAY = 1 # 秒
SLEEP_BETWEEN = 0.1 # 每条间延迟

# 加载所有数据
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    records = json.load(f)

results = []

# 批量推理
for record in tqdm(records, desc="Processing records", unit="rec"):
    system = record.get("system_prompt", "")
    text = record.get("prompt", "")
    messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": f"Text: \"{text}\""}
    ]

# 重试调用
    entities = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=256,
            )
            content = resp.choices[0].message.content.strip()
            companies = json.loads(content)
            # Remove duplicates using a set
            companies = list(set(companies))
            companies.sort()
            # 将列表转换为格式化的 JSON 字符串
            entities = json.dumps(companies, indent=2)
            break
        except APIConnectionError as e:
            print(f"Connection error on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY)
        except OpenAIError as e:
            print(f"API error: {e}")
            break
        except json.JSONDecodeError:
            print(f"Invalid JSON response: {content}")
            break

    record["predicted_entities"] = str(entities)
    results.append(record)
    time.sleep(SLEEP_BETWEEN)

# 写出结果
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Batch processing completed: {len(results)} records saved to {OUTPUT_FILE}")

