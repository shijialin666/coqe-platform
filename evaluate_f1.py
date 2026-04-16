# -*- coding: utf-8 -*-
import json
import argparse

def compute_f1_scores(pred_pt, gold_pt):
    """
    与 eval_utils.py 完全一致的 F1 计算逻辑
    输入: List[List[List[str]]]  # 三层列表
    """
    if len(pred_pt) != len(gold_pt):
        raise ValueError("预测与标准答案数量不匹配")

    n_tp = n_gold = n_pred = 0

    for i in range(len(pred_pt)):
        # 转换为 tuple 并过滤空五元组（与原代码一致）
        def filter_empty(triples):
            result = []
            for t in triples:
                # t 是 [sub, obj, ap, op, pr]
                if any(t):  # 如果有任何一个字段非空
                    result.append(tuple(t))
            return set(result)

        gold_set = filter_empty(gold_pt[i])
        pred_set = filter_empty(pred_pt[i])

        n_gold += len(gold_set)
        n_pred += len(pred_set)
        n_tp += len(gold_set & pred_set)

    precision = n_tp / n_pred if n_pred > 0 else 0.0
    recall = n_tp / n_gold if n_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to results JSON file")
    args = parser.parse_args()

    # 加载 JSON
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 target 和 prediction（保持原始格式：List[List[List[str]]]）
    all_labels = []
    all_preds = []

    for example in data["examples"]:
        # 直接使用 JSON 中的列表（已经是 [[str, str, ...], ...] 格式）
        all_labels.append(example["target"])
        all_preds.append(example["prediction"])

    # 计算 F1
    scores = compute_f1_scores(all_preds, all_labels)

    # 输出结果
    print(f"Dataset: {data.get('dataset', 'unknown')}")
    print(f"Computed F1: {scores['f1']:.6f}")
    print(f"Precision: {scores['precision']:.6f}")
    print(f"Recall: {scores['recall']:.6f}")
    
    # 验证是否与 JSON 中记录的 F1 一致
    if "f1" in data:
        print(f"JSON recorded F1: {data['f1']:.6f}")
        

if __name__ == "__main__":
    main()