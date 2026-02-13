#!/usr/bin/env python3
"""
QA Results Evaluation Script
评估 QA-result 目录中的所有方法结果，计算 EM 和 F1 分数
"""

import os
import json
import re
import string
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from unidecode import unidecode


# ============================================================================
# 评估指标模块 (参考 utils.py)
# ============================================================================

def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def replace_dash_with_space(text):
        return " ".join(text.split("-"))

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join([ch for ch in text if ch not in exclude])

    def lower(text):
        if isinstance(text, (int, float)):
            text = str(text)
        return unidecode(text.lower())

    return white_space_fix(remove_articles(remove_punc(replace_dash_with_space(lower(s)))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """计算 token 级别的 F1 分数"""
    ZERO_METRIC = (0, 0, 0)

    if prediction in ['yes', 'no', 'noanswer'] and prediction != ground_truth:
        return ZERO_METRIC
    if ground_truth in ['yes', 'no', 'noanswer'] and prediction != ground_truth:
        return ZERO_METRIC

    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def get_metrics(preds: List[str], gt_answer: List[str]) -> Dict[str, float]:
    """计算 EM 和 F1 指标"""
    if isinstance(gt_answer, str):
        gt_answer = [gt_answer]
    if isinstance(preds, str):
        preds = [preds]

    if len(preds) == 0 and len(gt_answer) != 0:
        return {'em': 0, 'f1': 0}
    if len(preds) != 0 and len(gt_answer) == 0:
        return {'em': 0, 'f1': 0}

    em = 0
    f1 = 0

    for pred in preds:
        pred = normalize_answer(pred)
        if pred == "":
            if gt_answer[0] == "":
                return {'em': 1, 'f1': 1.0}
            else:
                return {'em': 0, 'f1': 0}

        for gt in gt_answer:
            gt = normalize_answer(gt)
            em = max(em, int(pred == gt))
            f1 = max(f1, f1_score(pred, gt)[0])
            if em:
                return {'em': 1, 'f1': 1.0}

    return {'em': em, 'f1': f1}


# ============================================================================
# 答案提取模块
# ============================================================================

def extract_bold_entity_from_first_sentence(text: str) -> str:
    """从第一句话中提取加粗实体（**实体名**）"""
    if not text:
        return ""

    # 找到第一句话（以./?/!结束）
    first_sentence_match = re.split(r'[.!?]', text)
    first_sentence = first_sentence_match[0] if first_sentence_match else text

    # 提取加粗实体 **XXX**
    bold_pattern = r'\*\*([^*]+)\*\*'
    matches = re.findall(bold_pattern, first_sentence)

    if matches:
        return matches[0].strip()

    # 如果没有加粗实体，返回第一句话
    return first_sentence.strip()


def extract_answer_from_dyg_rag(answer_text: str) -> str:
    """从 DyG-RAG 的 answer 字段中提取答案"""
    if not answer_text:
        return ""

    # 查找 **Answer:** 后的内容
    answer_pattern = r'\*\*Answer:\*\*\s*(.*?)(?:\*\*Justification:\*\*|$)'
    answer_match = re.search(answer_pattern, answer_text, re.DOTALL | re.IGNORECASE)

    if answer_match:
        answer_section = answer_match.group(1).strip()
        # 从 Answer 部分提取第一个加粗实体
        return extract_bold_entity_from_first_sentence(answer_section)

    # 如果没有找到 Answer: 标记，尝试提取第一个加粗实体
    bold_pattern = r'\*\*([^*]+)\*\*'
    matches = re.findall(bold_pattern, answer_text)
    if matches:
        # 过滤掉 "Answer:", "Justification:" 等标记
        for match in matches:
            if match.lower() not in ['answer:', 'justification:', 'answer', 'justification']:
                return match.strip()

    return ""


def extract_answer_by_method(method_name: str, result_data: Dict) -> str:
    """根据方法名提取答案"""
    if method_name == "DyG-RAG":
        # DyG-RAG: 从 answer 字段的 Answer: 部分提取
        answer_text = result_data.get('answer', '')
        return extract_answer_from_dyg_rag(answer_text)
    else:
        # 其他方法: 从 output 的第一句话提取加粗实体
        output_text = result_data.get('output', '')
        return extract_bold_entity_from_first_sentence(output_text)


# ============================================================================
# 数据加载模块
# ============================================================================

def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 格式文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_json(file_path: str) -> Dict:
    """加载 JSON 格式文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_method_results(method_path: str, method_name: str) -> List[Dict]:
    """加载某个方法的结果文件"""
    results_file = os.path.join(method_path, 'results.json')

    if not os.path.exists(results_file):
        print(f"⚠️  警告: 未找到结果文件 {results_file}")
        return []

    # 尝试作为 JSON 加载
    try:
        data = load_json(results_file)
        if isinstance(data, dict) and 'results' in data:
            # DyG-RAG 格式
            return data['results']
        elif isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试作为 JSONL 加载
    try:
        return load_jsonl(results_file)
    except Exception as e:
        print(f"❌ 错误: 无法加载 {results_file}: {e}")
        return []


def load_ground_truth(dataset_path: str) -> Dict[str, Dict]:
    """加载 ground truth 数据集，返回 question -> {targets, level} 的映射"""
    gt_mapping = {}

    test_file = os.path.join(dataset_path, 'test_processed.json')
    if not os.path.exists(test_file):
        print(f"⚠️  警告: 未找到 ground truth 文件 {test_file}")
        return gt_mapping

    data = load_json(test_file)

    if 'datas' in data:
        for doc in data['datas']:
            if 'questions_list' in doc:
                for q_data in doc['questions_list']:
                    question = q_data.get('question', '')
                    targets = q_data.get('targets', [])
                    level = q_data.get('level', 'unknown')
                    gt_mapping[question] = {
                        'targets': targets,
                        'level': level
                    }

    return gt_mapping


# ============================================================================
# 评估执行模块
# ============================================================================

def evaluate_method(method_name: str, method_path: str, dataset_name: str,
                   dataset_path: str) -> Dict[str, Any]:
    """评估单个方法"""
    print(f"\n🔍 评估方法: {method_name} (数据集: {dataset_name})")

    # 加载方法结果
    results = load_method_results(method_path, method_name)
    if not results:
        print(f"  ⚠️  跳过 {method_name}: 无结果数据")
        return None

    # 加载 ground truth
    gt_mapping = load_ground_truth(dataset_path)

    # 评估每个问题
    evaluation_results = []

    for result in results:
        # 提取问题和预测答案
        question = result.get('question', '')
        predicted_answer = extract_answer_by_method(method_name, result)

        # 获取 ground truth
        if question in gt_mapping:
            gt_data = gt_mapping[question]
            gt_answers = gt_data['targets']
            level = gt_data['level']
        else:
            # HippoRAG 等方法自带 ground truth
            gt_answers = result.get('answer', [])
            level = result.get('level', 'unknown')

        # 计算指标
        metrics = get_metrics([predicted_answer], gt_answers)

        evaluation_results.append({
            'question': question,
            'predicted': predicted_answer,
            'ground_truth': gt_answers,
            'level': level,
            'em': metrics['em'],
            'f1': metrics['f1']
        })

    print(f"  ✅ 评估了 {len(evaluation_results)} 个问题")

    return {
        'method': method_name,
        'dataset': dataset_name,
        'results': evaluation_results
    }


def aggregate_results(evaluation_data: List[Dict]) -> Dict:
    """聚合评估结果，按方法、数据集、级别分组"""
    aggregated = defaultdict(lambda: {
        'em_scores': [],
        'f1_scores': [],
        'count': 0
    })

    for eval_result in evaluation_data:
        if not eval_result:
            continue

        method = eval_result['method']
        dataset = eval_result['dataset']

        for result in eval_result['results']:
            level = result['level']
            em = result['em']
            f1 = result['f1']

            # 按级别聚合
            key = (method, dataset, level)
            aggregated[key]['em_scores'].append(em)
            aggregated[key]['f1_scores'].append(f1)
            aggregated[key]['count'] += 1

            # 总体聚合
            overall_key = (method, dataset, 'overall')
            aggregated[overall_key]['em_scores'].append(em)
            aggregated[overall_key]['f1_scores'].append(f1)
            aggregated[overall_key]['count'] += 1

    # 计算平均值
    final_results = []
    for (method, dataset, level), data in aggregated.items():
        em_avg = sum(data['em_scores']) / len(data['em_scores']) * 100
        f1_avg = sum(data['f1_scores']) / len(data['f1_scores']) * 100

        final_results.append({
            'method': method,
            'dataset': dataset,
            'level': level,
            'em': em_avg,
            'f1': f1_avg,
            'count': data['count']
        })

    return final_results


# ============================================================================
# 表格生成模块
# ============================================================================

def generate_markdown_table(results: List[Dict], output_file: str):
    """生成 Markdown 格式表格"""
    lines = []
    lines.append("# QA 评估结果\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## 详细结果\n")

    # 表头
    lines.append("| Method | Dataset | Level | EM (%) | F1 (%) | Questions |")
    lines.append("|--------|---------|-------|--------|--------|-----------|")

    # 按方法和数据集分组，先显示子级别，再显示 overall
    grouped = defaultdict(list)
    for r in results:
        key = (r['method'], r['dataset'])
        grouped[key].append(r)

    for (method, dataset), items in sorted(grouped.items()):
        # 先显示 easy/hard 等子级别
        sub_levels = [item for item in items if item['level'] != 'overall']
        for item in sorted(sub_levels, key=lambda x: x['level']):
            lines.append(
                f"| {item['method']} | {item['dataset']} | {item['level']} | "
                f"{item['em']:.2f} | {item['f1']:.2f} | {item['count']} |"
            )

        # 显示 overall（加粗）
        overall_items = [item for item in items if item['level'] == 'overall']
        for item in overall_items:
            lines.append(
                f"| **{item['method']}** | **{item['dataset']}** | **{item['level']}** | "
                f"**{item['em']:.2f}** | **{item['f1']:.2f}** | **{item['count']}** |"
            )

    # 方法对比（Overall）
    lines.append("\n## 方法对比 (Overall)\n")
    lines.append("| Method | Avg EM (%) | Avg F1 (%) |")
    lines.append("|--------|------------|------------|")

    method_overall = defaultdict(lambda: {'em': [], 'f1': []})
    for r in results:
        if r['level'] == 'overall':
            method_overall[r['method']]['em'].append(r['em'])
            method_overall[r['method']]['f1'].append(r['f1'])

    method_comparison = []
    for method, data in method_overall.items():
        avg_em = sum(data['em']) / len(data['em'])
        avg_f1 = sum(data['f1']) / len(data['f1'])
        method_comparison.append((method, avg_em, avg_f1))

    # 按 F1 降序排列
    for method, avg_em, avg_f1 in sorted(method_comparison, key=lambda x: x[2], reverse=True):
        lines.append(f"| {method} | {avg_em:.2f} | {avg_f1:.2f} |")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Markdown 表格已保存到: {output_file}")


def generate_csv_table(results: List[Dict], output_file: str):
    """生成 CSV 格式表格"""
    import csv

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Dataset', 'Level', 'EM (%)', 'F1 (%)', 'Questions'])

        for r in sorted(results, key=lambda x: (x['method'], x['dataset'], x['level'])):
            writer.writerow([
                r['method'],
                r['dataset'],
                r['level'],
                f"{r['em']:.2f}",
                f"{r['f1']:.2f}",
                r['count']
            ])

    print(f"✅ CSV 表格已保存到: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='评估 QA 结果')
    parser.add_argument('--qa-result-dir', default='/workspace/ETE-Graph/QA-result',
                       help='QA-result 目录路径')
    parser.add_argument('--dataset-dir', default='/workspace/ETE-Graph/dataset',
                       help='数据集目录路径')
    parser.add_argument('--output-dir', default='/workspace/ETE-Graph/evaluation_results',
                       help='输出目录路径')
    parser.add_argument('--dataset', default=None,
                       help='仅评估指定数据集')
    parser.add_argument('--method', default=None,
                       help='仅评估指定方法')
    parser.add_argument('--output-format', default='both', choices=['markdown', 'csv', 'both'],
                       help='输出格式')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("QA 结果评估脚本")
    print("=" * 70)

    # 扫描所有数据集和方法
    qa_result_path = Path(args.qa_result_dir)
    evaluation_data = []

    for dataset_dir in qa_result_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # 过滤数据集
        if args.dataset and dataset_name != args.dataset:
            continue

        dataset_path = os.path.join(args.dataset_dir, dataset_name)

        for method_dir in dataset_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name

            # 过滤方法
            if args.method and method_name != args.method:
                continue

            # 评估
            eval_result = evaluate_method(
                method_name, str(method_dir), dataset_name, dataset_path
            )
            if eval_result:
                evaluation_data.append(eval_result)

    if not evaluation_data:
        print("\n❌ 没有找到任何评估数据")
        return

    # 聚合结果
    print("\n📊 聚合评估结果...")
    aggregated_results = aggregate_results(evaluation_data)

    # 生成表格
    if args.output_format in ['markdown', 'both']:
        md_file = os.path.join(args.output_dir, 'results_table.md')
        generate_markdown_table(aggregated_results, md_file)

    if args.output_format in ['csv', 'both']:
        csv_file = os.path.join(args.output_dir, 'results_table.csv')
        generate_csv_table(aggregated_results, csv_file)

    print("\n" + "=" * 70)
    print("✅ 评估完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
