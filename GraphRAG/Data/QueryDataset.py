import pandas as pd
from torch.utils.data import Dataset
import os
from pathlib import Path
from typing import Optional
from .DatasetLoaders import DatasetLoaderFactory


class RAGQueryDataset(Dataset):
    """
    统一的数据集接口，支持多种数据集格式

    支持的数据集:
    - tempreason: TEMPREASON数据集
    - timeqa: TIMEQA数据集
    - legacy: 兼容格式（Corpus.json + Question.json）

    初始化方式:
    1. RAGQueryDataset(data_dir="/path/to/data", dataset_name="tempreason")
    2. RAGQueryDataset(data_dir="/path/to/data")  # 自动检测数据集类型
    3. RAGQueryDataset(data_dir="/path/to/data", file_pattern="train_l2_processed.json")
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: Optional[str] = None,
        file_pattern: Optional[str] = None,
        auto_detect: bool = True
    ):
        """
        初始化数据集

        Args:
            data_dir: 数据目录路径
            dataset_name: 数据集名称('tempreason', 'timeqa' 或 None)
            file_pattern: 特定文件名模式(如 'train_l2_processed.json')
            auto_detect: 当dataset_name为None时是否自动检测

        Raises:
            ValueError: 如果数据集类型未知且auto_detect=False
            FileNotFoundError: 如果找不到数据集文件
        """
        super().__init__()

        self.data_dir = data_dir
        self.file_pattern = file_pattern
        self.loader = None
        self.corpus_list = None
        self.qa_pairs = None

        # 确定数据集类型
        self._dataset_name = self._determine_dataset_type(dataset_name, auto_detect)

        # 初始化加载器
        self._initialize_loader()

    def _determine_dataset_type(self, dataset_name: Optional[str], auto_detect: bool) -> str:
        """
        确定数据集类型

        优先级:
        1. 显式传入的dataset_name
        2. 自动检测（如果auto_detect=True）
        3. 默认为'legacy'（支持旧的Corpus.json + Question.json格式）
        """
        if dataset_name:
            return dataset_name.lower().strip()

        if auto_detect:
            detected = DatasetLoaderFactory.detect_dataset_type(self.data_dir)
            if detected != 'unknown':
                return detected

        # 检查是否有旧格式的文件
        corpus_path = os.path.join(self.data_dir, "Corpus.json")
        if os.path.exists(corpus_path):
            return 'legacy'

        raise ValueError(
            f"Cannot determine dataset type for {self.data_dir}. "
            f"Please provide dataset_name parameter or ensure data files exist."
        )

    def _initialize_loader(self):
        """初始化相应的数据集加载器"""
        if self._dataset_name == 'legacy':
            self._load_legacy_format()
        else:
            try:
                self.loader = DatasetLoaderFactory.create(
                    self._dataset_name,
                    self.data_dir,
                    self.file_pattern
                )
                self.loader.load()
                self.corpus_list = self.loader.get_corpus()
                self.qa_pairs = self.loader.get_qa_pairs()
            except (ValueError, FileNotFoundError) as e:
                # 尝试回退到旧格式
                if os.path.exists(os.path.join(self.data_dir, "Corpus.json")):
                    self._load_legacy_format()
                else:
                    raise

    def _load_legacy_format(self):
        """加载旧格式的数据集（Corpus.json + Question.json）"""
        corpus_path = os.path.join(self.data_dir, "Corpus.json")
        qa_path = os.path.join(self.data_dir, "Question.json")

        # 加载语料库
        corpus_df = pd.read_json(corpus_path, lines=True)
        self.corpus_list = [
            {
                "title": corpus_df.iloc[i]["title"],
                "content": corpus_df.iloc[i]["context"],
                "doc_id": i,
            }
            for i in range(len(corpus_df))
        ]

        # 加载问答对
        qa_df = pd.read_json(qa_path, lines=True, orient="records")
        self.qa_pairs = []
        for i in range(len(qa_df)):
            # 获取answer字段，确保是列表格式
            answer = qa_df.iloc[i].get("answer", "")
            # 如果answer不是列表，转换为列表
            if not isinstance(answer, list):
                answer = [answer] if answer else []

            qa_item = {
                "id": i,
                "question": qa_df.iloc[i]["question"],
                "answer": answer,
            }

            # 添加其他属性
            for col in qa_df.columns:
                if col not in ["question", "answer"]:
                    qa_item[col] = qa_df.iloc[i][col]

            self.qa_pairs.append(qa_item)

        self._dataset_name = 'legacy'

    def get_corpus(self) -> list:
        """
        获取语料库列表

        返回格式:
        [
            {
                "title": str,
                "content": str,
                "doc_id": int,
                ... (可能包含其他元数据)
            },
            ...
        ]
        """
        if self.corpus_list is None:
            self._initialize_loader()
        return self.corpus_list

    def __len__(self) -> int:
        """获取问答对数量"""
        if self.qa_pairs is None:
            self._initialize_loader()
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        获取单个问答对

        返回格式:
        {
            "id": int,
            "question": str,
            "answer": list,  # 列表形式
            ... (可能包含其他元数据如date, level等)
        }
        """
        if self.qa_pairs is None:
            self._initialize_loader()

        if idx < 0 or idx >= len(self.qa_pairs):
            raise IndexError(f"Index {idx} out of range [0, {len(self.qa_pairs)})")

        return self.qa_pairs[idx]

    @property
    def dataset_name(self) -> str:
        """获取当前数据集名称"""
        return self._dataset_name

    @property
    def dataset_info(self) -> dict:
        """获取数据集信息"""
        return {
            'dataset_name': self._dataset_name,
            'data_dir': self.data_dir,
            'num_documents': len(self.corpus_list) if self.corpus_list else 0,
            'num_qa_pairs': len(self.qa_pairs) if self.qa_pairs else 0,
        }


if __name__ == "__main__":
    # 新的用法示例
    query_dataset = RAGQueryDataset(data_dir="/path/to/dataset", dataset_name="tempreason")
    corpus = query_dataset.get_corpus()
    item = query_dataset[0]

    # 自动检测用法
    query_dataset = RAGQueryDataset(data_dir="/path/to/dataset")

    # 指定文件用法
    query_dataset = RAGQueryDataset(
        data_dir="/path/to/dataset",
        dataset_name="tempreason",
        file_pattern="val_l2_processed.json"
    )
