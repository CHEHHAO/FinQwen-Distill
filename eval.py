import json
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

# 配置日志，用于记录评估结果
logging.basicConfig(
    filename='logs/eval.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

sft_data_path = 'data/clean_data/sft_data.json'
sft_response_path = 'data/clean_data/responses.json'

with open(sft_data_path, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

with open('responses.json', 'r', encoding='utf-8') as f:
    pred_data = json.load(f)

assert len(gold_data) == len(pred_data), "数量不一致：金标准与预测结果条数应相同"

# 2. 计算每条对象的 precision, recall, f1
precisions = []
recalls = []
f1s = []

for idx, (gold, pred) in enumerate(zip(gold_data, pred_data)):
    gold_set = set(gold.get('entities', []))
    pred_set = set(pred)
    true_positive = len(gold_set & pred_set)
    # 避免除零错误
    p = true_positive / len(pred_set) if pred_set else 0.0
    r = true_positive / len(gold_set) if gold_set else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)
    logging.info(
        f"Example {idx}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f} | Gold={gold_set} Pred={pred_set}"
    )

# 3. 计算总体 macro 平均
macro_p = sum(precisions) / len(precisions)
macro_r = sum(recalls) / len(recalls)
macro_f1 = sum(f1s) / len(f1s)

# 打印并记录总体结果
summary = (
    f"Macro Precision: {macro_p:.3f}\n"
    f"Macro Recall:    {macro_r:.3f}\n"
    f"Macro F1:        {macro_f1:.3f}"
)
print(summary)
logging.info("Overall Evaluation Metrics:\n" + summary)

# 可选：保存带指标的详细结果
with open('eval_results.json', 'w', encoding='utf-8') as outf:
    results = []
    for idx, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
        results.append({'index': idx, 'precision': p, 'recall': r, 'f1': f})
    json.dump({'summary': {
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1
    }, 'details': results}, outf, ensure_ascii=False, indent=2)
