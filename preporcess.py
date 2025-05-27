import json
import logging
from sklearn.model_selection import train_test_split

# 配置日志，将日志输出到 convert.log 文件
logging.basicConfig(
    filename='logs/convert.log',
    filemode='a',  # 追加模式
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# 假设原始数据保存在 data.json（包含一个或多个 JSON 对象）
# 每个对象具有字段：Title, Content, Companies_list

def load_raw_data(json_file_path):
    """
    从 JSON 文件中加载原始数据，返回列表。
    支持整个文件是一个 list 或者每行一个 JSON 对象。
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 按行读取多行 JSON 对象
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    return data


def convert_to_examples(raw_list):
    """
    将原始列表转换为 SFT 例子字典格式。

    Args:
        raw_list (List[dict]): 原始 JSON 对象列表，字段包括 Title, Content, Companies_list。

    Returns:
        List[dict]: 转换后的例子列表，每个例子包含 title, text, entities。
    """
    system_prompt = """You are a named-entity recognition assistant. Identify all company names in the following title and text, and output an array of strings, each string being one company name."""
    examples = []
    for item in raw_list:
        
        title = item.get('Title', '')
        text = item.get('Content', '')
        entities = item.get('Companies_list', [])
        entities_str = ', '.join(entities)  # 将列表转换为逗号分隔的字符串
        # 去除字符串左右空白，并保持实体唯一
        entities = list({e.strip() for e in entities if isinstance(e, str)})
        examples.append({
            "system_prompt": system_prompt,
            "prompt": f"Title: \"{title}\"\nText: \"{text}\"",
            "response": entities_str,
        })

    return examples


if __name__ == '__main__':


    raw_data_path = 'data/raw_data/AIDF_AlternativeData.news_with_companies.json'
    output_dir = 'data/clean_data'
    raw = load_raw_data(raw_data_path)
    examples = convert_to_examples(raw)

    # 划分为训练集、验证集、测试集
    # 先分割出测试集（占比10%），再从剩余中分割验证集（占比10%）
    train_val, test_set = train_test_split(examples, test_size=0.1, random_state=42)
    val_size = 0.1 / 0.9
    train_set, val_set = train_test_split(train_val, test_size=val_size, random_state=42)

    # 保存各划分文件
    splits = {
        'train': train_set,
        'validation': val_set,
        'test': test_set
    }
    for split_name, data in splits.items():
        path = f"{output_dir}/sft_{split_name}.json"
        with open(path, 'w', encoding='utf-8') as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(data)} examples to {path}.")
        print(f"Saved {len(data)} examples to {path}.")
