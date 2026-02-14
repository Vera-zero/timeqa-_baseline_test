from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
import time
import json
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator



def check_dirs(opt):
    # working_dir 是中间文件目录：/workspace/ETE-Graph/workdir/{dataset_name}/{method_name}
    # result_dir 是最终结果目录：/workspace/ETE-Graph/QA-result/{dataset_name}/{method_name}
    config_dir = os.path.join(opt.working_dir, "Configs")
    metric_dir = os.path.join(opt.working_dir, "Metrics")

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # 确保 result_dir（QA-result 目录）存在，用于保存 results.json
    os.makedirs(opt.result_dir, exist_ok=True)

    # 提取配置文件名
    method_config_name = Path(args.opt).name  # 如 "HippoRAG.yaml"
    base_config_path = Path(args.opt).parent.parent / "Config2.yaml"

    # 复制配置文件到 working_dir/Configs
    copyfile(args.opt, os.path.join(config_dir, method_config_name))
    if base_config_path.exists():
        copyfile(base_config_path, os.path.join(config_dir, "Config2.yaml"))

    return metric_dir  # 返回 Metrics 目录用于保存指标


def wrapper_query(query_dataset, digimon, result_dir, opt):
    """
    基本查询函数（支持断点续传）
    """
    # Checkpoint support - 加载已有结果
    save_path = os.path.join(opt.result_dir, "results.json")
    existing_results = []
    processed_indices = set()

    if os.path.exists(save_path):
        try:
            # 读取已有的结果文件
            with open(save_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    existing_results.append(result)
                    # 使用问题索引作为唯一标识
                    question_idx = result.get('question_idx')
                    if question_idx is not None:
                        processed_indices.add(question_idx)
            print(f"\n📂 发现已有结果文件，已完成 {len(processed_indices)} 个问题")
        except Exception as e:
            print(f"\n⚠️  读取已有结果文件失败: {e}，将重新开始")
            existing_results = []
            processed_indices = set()

    all_res = existing_results.copy()

    dataset_len = len(query_dataset)
    dataset_len = 10

    save_interval = 5  # 每5个问题保存一次
    questions_since_last_save = 0
    skip_mode = False  # 标记是否进入跳过模式

    print(f"\n开始处理 {dataset_len} 个问题...")

    for _, i in enumerate(range(dataset_len)):
        # 检查是否已处理过此问题
        if i in processed_indices:
            if not skip_mode:
                print(f"\n✓ 问题 {i} 已处理，跳过...")
                skip_mode = True
            continue

        # 一旦发现未处理的问题，说明从此之后都未处理
        if skip_mode:
            print(f"\n→ 从问题 {i} 开始继续处理...")
            skip_mode = False

        query = query_dataset[i]
        start_time = time.time()
        res = asyncio.run(digimon.query(query["question"]))
        end_time = time.time()
        query_time = end_time - start_time

        # 添加问题索引用于断点续传
        query["question_idx"] = i
        query["output"] = res
        query["query_time"] = query_time
        all_res.append(query)
        processed_indices.add(i)
        questions_since_last_save += 1

        # 每5个问题保存一次
        if questions_since_last_save >= save_interval:
            all_res_df = pd.DataFrame(all_res)
            all_res_df.to_json(save_path, orient="records", lines=True)
            print(f"\n💾 已保存进度: {len(processed_indices)}/{dataset_len} 个问题")
            questions_since_last_save = 0

    # 最终保存所有结果
    all_res_df = pd.DataFrame(all_res)
    all_res_df.to_json(save_path, orient="records", lines=True)
    print(f"\n✅ 结果已保存到: {save_path}")
    print(f"   - 处理问题数: {len(processed_indices)}/{dataset_len}")
    return save_path


def wrapper_query_filtered(filtered_questions, digimon, result_dir, opt):
    """
    查询已经筛选过的问题列表（支持断点续传）

    Args:
        filtered_questions: 已筛选的问题列表
        digimon: GraphRAG实例
        result_dir: 结果保存目录（这里用于metrics）
        opt: 配置对象
    """
    # Checkpoint support - 加载已有结果
    save_path = os.path.join(opt.result_dir, "results.json")
    existing_results = []
    processed_indices = set()

    if os.path.exists(save_path):
        try:
            # 读取已有的结果文件
            with open(save_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    existing_results.append(result)
                    # 使用问题索引作为唯一标识
                    question_idx = result.get('question_idx')
                    if question_idx is not None:
                        processed_indices.add(question_idx)
            print(f"\n📂 发现已有结果文件，已完成 {len(processed_indices)} 个问题")
        except Exception as e:
            print(f"\n⚠️  读取已有结果文件失败: {e}，将重新开始")
            existing_results = []
            processed_indices = set()

    all_res = existing_results.copy()
    save_interval = 5  # 每5个问题保存一次
    questions_since_last_save = 0
    skip_mode = False  # 标记是否进入跳过模式

    print(f"\n开始处理 {len(filtered_questions)} 个问题...")

    for idx, query in enumerate(filtered_questions):
        # 检查是否已处理过此问题
        if idx in processed_indices:
            if not skip_mode:
                print(f"\n✓ 问题 {idx} 已处理，跳过...")
                skip_mode = True
            continue

        # 一旦发现未处理的问题，说明从此之后都未处理
        if skip_mode:
            print(f"\n→ 从问题 {idx} 开始继续处理...")
            skip_mode = False

        doc_id = query.get('doc_id', 'N/A')
        print(f"\n[{idx+1}/{len(filtered_questions)}] 文档{doc_id}: {query['question'][:60]}...")

        start_time = time.time()
        res = asyncio.run(digimon.query(query["question"]))
        end_time = time.time()
        query_time = end_time - start_time

        # 添加问题索引用于断点续传
        query["question_idx"] = idx
        query["output"] = res
        query["query_time"] = query_time
        all_res.append(query)
        processed_indices.add(idx)
        questions_since_last_save += 1

        print(f"  回答: {res[:100]}...")
        print(f"  查询耗时: {query_time:.2f}秒")

        # 每5个问题保存一次
        if questions_since_last_save >= save_interval:
            all_res_df = pd.DataFrame(all_res)
            all_res_df.to_json(save_path, orient="records", lines=True)
            print(f"\n💾 已保存进度: {len(processed_indices)}/{len(filtered_questions)} 个问题")
            questions_since_last_save = 0

    # 最终保存所有结果
    all_res_df = pd.DataFrame(all_res)
    all_res_df.to_json(save_path, orient="records", lines=True)
    print(f"\n✅ 结果已保存到: {save_path}")
    print(f"   - 处理问题数: {len(processed_indices)}/{len(filtered_questions)}")
    return save_path


async def wrapper_evaluation(path, opt, result_dir):
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))


if __name__ == "__main__":

    # with open("./book.txt") as f:
    #     doc = f.read()

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("-data_root", type=str, default=None,
                        help="Root directory for datasets (overrides config file).")
    parser.add_argument("-file_pattern", type=str, default=None,
                        help="Specific data file name (e.g., 'test_processed.json').")
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name, data_root=args.data_root)
    digimon = GraphRAG(config=opt)
    result_dir = check_dirs(opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name),
        file_pattern=args.file_pattern
    )

    # 只使用前2个文档
    corpus = query_dataset.get_corpus()
    corpus = corpus[:2]
    print(f"使用前 {len(corpus)} 个文档:")
    for doc in corpus:
        print(f"  - doc_id={doc['doc_id']}: {doc['title']}")

    # 筛选出属于这2个文档的问题
    doc_ids = {doc['doc_id'] for doc in corpus}
    filtered_questions = [
        query_dataset[i]
        for i in range(len(query_dataset))
        if query_dataset[i].get('doc_id') in doc_ids
    ]
    print(f"\n这些文档对应 {len(filtered_questions)} 个问题")

    # # 插入前2个文档
    # asyncio.run(digimon.insert(corpus))

    # 查询属于这2个文档的问题
    save_path = wrapper_query_filtered(filtered_questions, digimon, result_dir, opt)

    # # 评估结果
    # asyncio.run(wrapper_evaluation(save_path, opt, result_dir))

    # for train_item in dataloader:

    # a = asyncio.run(digimon.query("Who is Fred Gehrke?"))

    # asyncio.run(digimon.query("Who is Scrooge?"))
