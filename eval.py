import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 配置日志，用于记录评估结果
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = '/home/svu/idfv22/dev/FinQwen-Distill/logs'
logging.basicConfig(
    filename=f'{log_dir}/eval_{current_time}.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

sft_data_path = '/home/svu/idfv22/dev/FinQwen-Distill/data/clean_data/sft_test.json'
# sft_response_path = '/home/svu/idfv22/dev/FinQwen-Distill/data/generated/baseline.json'
# sft_response_path = "/home/svu/idfv22/dev/FinQwen-Distill/data/generated/qwen_2.5_responses.json"
sft_response_path = "/home/svu/idfv22/dev/FinQwen-Distill/data/generated/fin_r1_responses.json"



def plot_metrics(metrics, save_path):
    """绘制评估指标图表"""
    plt.figure(figsize=(12, 6))

    # 绘制span-level指标
    plt.subplot(1, 2, 1)
    span_metrics = metrics['span_level']['metrics']
    x = np.arange(len(span_metrics['precision']))
    width = 0.25
    plt.bar(x - width, span_metrics['precision'], width, label='Precision')
    plt.bar(x, span_metrics['recall'], width, label='Recall')
    plt.bar(x + width, span_metrics['f1'], width, label='F1')
    plt.title('Span-level Metrics')
    plt.xlabel('Examples')
    plt.ylabel('Score')
    plt.legend()

    # 绘制token-level指标
    plt.subplot(1, 2, 2)
    token_metrics = metrics['token_level']['metrics']
    x = np.arange(len(token_metrics['precision']))
    plt.bar(x - width, token_metrics['precision'], width, label='Precision')
    plt.bar(x, token_metrics['recall'], width, label='Recall')
    plt.bar(x + width, token_metrics['f1'], width, label='F1')
    plt.title('Token-level Metrics')
    plt.xlabel('Examples')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_ner(gold_data, pred_data):
    """
    对比实体集合进行评估，返回span级和token级的precision/recall/f1指标。
    不再使用seqeval的classification_report，避免长度不匹配错误。
    """
    span_metrics = {'precision': [], 'recall': [], 'f1': []}
    token_metrics = {'precision': [], 'recall': [], 'f1': []}

    for gold, pred in zip(gold_data, pred_data):
        # 加载预测结果实体列表

        gold_list = json.loads(gold['response'])
        pred_list = json.loads(pred['predicted_entities'])

        # gold_list = gold.get('predicted_entities', [])

        gold_set = set(gold_list)
        pred_set = set(pred_list)

        # 处理空-空样本
        if not gold_set and not pred_set:
            p, r, f = 1.0, 1.0, 1.0
        else:
            tp = len(gold_set & pred_set)
            p = tp / len(pred_set) if pred_set else 0.0
            r = tp / len(gold_set) if gold_set else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
        span_metrics['precision'].append(p)
        span_metrics['recall'].append(r)
        span_metrics['f1'].append(f)

        # Token-level评估
        gold_tokens = set()
        pred_tokens = set()
        for ent in gold_set:
            gold_tokens.update(ent.split())
        for ent in pred_set:
            pred_tokens.update(ent.split())

        if not gold_tokens and not pred_tokens:
            p_t, r_t, f_t = 1.0, 1.0, 1.0
        else:
            tp_t = len(gold_tokens & pred_tokens)
            p_t = tp_t / len(pred_tokens) if pred_tokens else 0.0
            r_t = tp_t / len(gold_tokens) if gold_tokens else 0.0
            f_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) else 0.0
        token_metrics['precision'].append(p_t)
        token_metrics['recall'].append(r_t)
        token_metrics['f1'].append(f_t)

        logging.info(f"Span: p={p:.3f}, r={r:.3f}, f={f:.3f}; Token: p_t={p_t:.3f}, r_t={r_t:.3f}, f_t={f_t:.3f}")

    # 计算宏平均
    span_macro = {k: sum(v) / len(v) for k, v in span_metrics.items()}
    token_macro = {k: sum(v) / len(v) for k, v in token_metrics.items()}

    return {
        'span_level': {'metrics': span_metrics, 'macro': span_macro},
        'token_level': {'metrics': token_metrics, 'macro': token_macro}
    }


def main():
    with open(sft_data_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    with open(sft_response_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    assert len(gold_data) == len(pred_data), "金标准与预测结果条数不一致"

    results = evaluate_ner(gold_data, pred_data)
    span_m = results['span_level']['macro']
    token_m = results['token_level']['macro']

    summary = (
        f"Span-level 宏平均: Precision={span_m['precision']:.3f}, Recall={span_m['recall']:.3f}, F1={span_m['f1']:.3f}\n"
        f"Token-level 宏平均: Precision={token_m['precision']:.3f}, Recall={token_m['recall']:.3f}, F1={token_m['f1']:.3f}"
    )
    print(summary)
    logging.info("Overall: " + summary)

    # 保存结果
    results_file = f'/home/svu/idfv22/dev/FinQwen-Distill/results/eval_results_{current_time}.json'
    with open(results_file, 'w', encoding='utf-8') as outf:
        json.dump(results, outf, ensure_ascii=False, indent=2)

    # 可视化
    plot_metrics(results, f'/home/svu/idfv22/dev/FinQwen-Distill/results/eval_metrics_{current_time}.png')

if __name__ == "__main__":
    main()
