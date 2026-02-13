from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator



def check_dirs(opt):
    # working_dir 已经是完整路径：/workspace/ETE-Graph/QA-result/{dataset_name}/{method_name}
    # 直接在其下创建子目录即可
    result_dir = os.path.join(opt.working_dir, "Results")
    config_dir = os.path.join(opt.working_dir, "Configs")
    metric_dir = os.path.join(opt.working_dir, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # 提取配置文件名
    method_config_name = Path(args.opt).name  # 如 "HippoRAG.yaml"
    base_config_path = Path(args.opt).parent.parent / "Config2.yaml"

    # 复制配置文件到配置目录
    copyfile(args.opt, os.path.join(config_dir, method_config_name))
    if base_config_path.exists():
        copyfile(base_config_path, os.path.join(config_dir, "Config2.yaml"))

    return result_dir


def wrapper_query(query_dataset, digimon, result_dir):
    all_res = []

    dataset_len = len(query_dataset)
    dataset_len = 10

    for _, i in enumerate(range(dataset_len)):
        query = query_dataset[i]
        res = asyncio.run(digimon.query(query["question"]))
        query["output"] = res
        all_res.append(query)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    return save_path


def wrapper_query_filtered(filtered_questions, digimon, result_dir):
    """
    查询已经筛选过的问题列表

    Args:
        filtered_questions: 已筛选的问题列表
        digimon: GraphRAG实例
        result_dir: 结果保存目录
    """
    all_res = []

    print(f"\n开始查询 {len(filtered_questions)} 个问题...")

    for idx, query in enumerate(filtered_questions):
        doc_id = query.get('doc_id', 'N/A')
        print(f"\n[{idx+1}/{len(filtered_questions)}] 文档{doc_id}: {query['question'][:60]}...")

        res = asyncio.run(digimon.query(query["question"]))
        query["output"] = res
        all_res.append(query)

        print(f"  回答: {res[:100]}...")

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    print(f"\n结果已保存到: {save_path}")
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
    save_path = wrapper_query_filtered(filtered_questions, digimon, result_dir)

    # # 评估结果
    # asyncio.run(wrapper_evaluation(save_path, opt, result_dir))

    # for train_item in dataloader:

    # a = asyncio.run(digimon.query("Who is Fred Gehrke?"))

    # asyncio.run(digimon.query("Who is Scrooge?"))
