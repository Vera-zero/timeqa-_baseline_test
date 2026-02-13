from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path


class DatasetLoader(ABC):
    """数据集加载器的抽象基类"""

    def __init__(self, data_dir: str, file_pattern: Optional[str] = None):
        """
        初始化数据集加载器

        Args:
            data_dir: 数据目录路径
            file_pattern: 可选的特定文件名模式
        """
        self.data_dir = data_dir
        self.file_pattern = file_pattern
        self.raw_data = None
        self.documents = None
        self.qa_pairs = None

    @abstractmethod
    def _find_json_file(self) -> str:
        """找到并返回数据集JSON文件路径"""
        pass

    @abstractmethod
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        pass

    @abstractmethod
    def _extract_documents(self) -> List[Dict[str, Any]]:
        """从原始数据中提取文档列表"""
        pass

    @abstractmethod
    def _flatten_qas(self) -> List[Dict[str, Any]]:
        """将嵌套的问答对展平为单层列表"""
        pass

    def get_corpus(self) -> List[Dict[str, Any]]:
        """
        获取标准化的语料库格式
        返回: [{"title": str, "content": str, "doc_id": int, ...}, ...]
        """
        if self.documents is None:
            self.load()
        return [
            {
                'title': doc['title'],
                'content': doc['content'],
                'doc_id': doc['doc_id'],
            }
            for doc in self.documents
        ]

    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """
        获取标准化的问答对
        返回: [{"id": int, "question": str, "answer": list, "doc_id": int, ...}, ...]
        """
        if self.qa_pairs is None:
            self.load()
        return self.qa_pairs

    def load(self):
        """主加载流程"""
        file_path = self._find_json_file()
        self.raw_data = self._load_json(file_path)
        self.documents = self._extract_documents()
        self.qa_pairs = self._flatten_qas()


class TempreasionDatasetLoader(DatasetLoader):
    """TEMPREASON数据集专用加载器"""

    def __init__(self, data_dir: str, file_pattern: Optional[str] = None):
        super().__init__(data_dir, file_pattern)
        self.metadata_fields = ['date', 'id', 'none_context', 'neg_answers', 'kb_answers']

    def _find_json_file(self) -> str:
        """
        查找TEMPREASON数据集文件
        文件名格式: {train/val/test}_l{2/3}_processed.json
        优先级: train > val > test, l2 > l3
        """
        data_dir = Path(self.data_dir)

        if self.file_pattern:
            pattern_path = data_dir / self.file_pattern
            if pattern_path.exists():
                return str(pattern_path)
            else:
                raise FileNotFoundError(
                    f"Specified file pattern '{self.file_pattern}' not found in {data_dir}"
                )

        # 查找优先级: train_l2 > train_l3 > val_l2 > val_l3 > test_l2 > test_l3
        for prefix in ['train', 'val', 'test']:
            for level in ['l2', 'l3']:
                file_name = f"{prefix}_{level}_processed.json"
                file_path = data_dir / file_name
                if file_path.exists():
                    return str(file_path)

        raise FileNotFoundError(
            f"No TEMPREASON dataset files found in {data_dir}. "
            f"Expected files like train_l2_processed.json, val_l2_processed.json, etc."
        )

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_documents(self) -> List[Dict[str, Any]]:
        """提取文档，每个文档对应contents中的一个元素"""
        documents = []
        for doc_id, item in enumerate(self.raw_data.get('contents', [])):
            doc = {
                'doc_id': doc_id,
                'content': item.get('fact_context', ''),
                'title': f"TEMPREASON_DOC_{doc_id}",
                'source_item_index': doc_id,
            }
            documents.append(doc)
        return documents

    def _flatten_qas(self) -> List[Dict[str, Any]]:
        """将嵌套的问答对展平"""
        qa_pairs = []
        global_qa_id = 0

        for doc_id, item in enumerate(self.raw_data.get('contents', [])):
            for question_data in item.get('question_list', []):
                # 提取答案 - 保留完整的 text_answers['text'] 列表
                text_answers = question_data.get('text_answers', {})
                answer = text_answers.get('text', []) if isinstance(text_answers, dict) else []

                qa = {
                    'id': global_qa_id,
                    'question': question_data.get('question', ''),
                    'answer': answer,  # 列表形式
                    'doc_id': doc_id,
                    'original_id': question_data.get('id', ''),
                }

                # 保留元数据字段
                for field in self.metadata_fields:
                    if field in question_data:
                        qa[field] = question_data[field]

                qa_pairs.append(qa)
                global_qa_id += 1

        return qa_pairs


class TimeqaDatasetLoader(DatasetLoader):
    """TIMEQA数据集专用加载器"""

    def __init__(self, data_dir: str, file_pattern: Optional[str] = None):
        super().__init__(data_dir, file_pattern)
        self.metadata_fields = ['level']

    def _find_json_file(self) -> str:
        """
        查找TIMEQA数据集文件
        文件名格式: {train/dev/test}_processed.json
        优先级: train > dev > test
        """
        data_dir = Path(self.data_dir)

        if self.file_pattern:
            pattern_path = data_dir / self.file_pattern
            if pattern_path.exists():
                return str(pattern_path)
            else:
                raise FileNotFoundError(
                    f"Specified file pattern '{self.file_pattern}' not found in {data_dir}"
                )

        # 查找优先级: train > dev > test
        for prefix in ['train', 'dev', 'test']:
            file_name = f"{prefix}_processed.json"
            file_path = data_dir / file_name
            if file_path.exists():
                return str(file_path)

        raise FileNotFoundError(
            f"No TIMEQA dataset files found in {data_dir}. "
            f"Expected files like train_processed.json, dev_processed.json, etc."
        )

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_documents(self) -> List[Dict[str, Any]]:
        """提取文档，每个文档对应datas中的一个元素"""
        documents = []
        for doc_id, item in enumerate(self.raw_data.get('datas', [])):
            doc = {
                'doc_id': doc_id,
                'content': item.get('context', ''),
                'title': item.get('idx', f"TIMEQA_DOC_{doc_id}"),
                'source_item_index': doc_id,
                'original_idx': item.get('idx', ''),
            }
            documents.append(doc)
        return documents

    def _flatten_qas(self) -> List[Dict[str, Any]]:
        """将嵌套的问答对展平"""
        qa_pairs = []
        global_qa_id = 0

        for doc_id, item in enumerate(self.raw_data.get('datas', [])):
            for question_data in item.get('questions_list', []):
                # 处理targets字段 - 保留完整的列表
                targets = question_data.get('targets', [])
                # 确保answer是列表类型
                if not isinstance(targets, list):
                    answer = []
                else:
                    answer = targets

                qa = {
                    'id': global_qa_id,
                    'question': question_data.get('question', ''),
                    'answer': answer,  # 列表形式
                    'doc_id': doc_id,
                }

                # 保留元数据字段
                for field in self.metadata_fields:
                    if field in question_data:
                        qa[field] = question_data[field]

                qa_pairs.append(qa)
                global_qa_id += 1

        return qa_pairs


class DatasetLoaderFactory:
    """数据集加载器工厂类"""

    LOADERS = {
        'tempreason': TempreasionDatasetLoader,
        'timeqa': TimeqaDatasetLoader,
    }

    @staticmethod
    def create(dataset_name: str, data_dir: str, file_pattern: Optional[str] = None) -> DatasetLoader:
        """
        根据数据集名称创建对应的加载器

        Args:
            dataset_name: 数据集名称 ('tempreason' 或 'timeqa')
            data_dir: 数据目录路径
            file_pattern: 可选的特定文件名模式

        Returns:
            DatasetLoader: 对应的加载器实例

        Raises:
            ValueError: 如果dataset_name不被支持
        """
        dataset_name_lower = dataset_name.lower().strip()

        if dataset_name_lower not in DatasetLoaderFactory.LOADERS:
            supported = ', '.join(DatasetLoaderFactory.LOADERS.keys())
            raise ValueError(
                f"Unsupported dataset: '{dataset_name}'. "
                f"Supported datasets: {supported}"
            )

        loader_class = DatasetLoaderFactory.LOADERS[dataset_name_lower]
        return loader_class(data_dir, file_pattern)

    @staticmethod
    def detect_dataset_type(data_dir: str) -> str:
        """
        根据目录中的文件自动检测数据集类型

        Args:
            data_dir: 数据目录路径

        Returns:
            str: 检测到的数据集类型 ('tempreason', 'timeqa', 或 'unknown')
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            return 'unknown'

        files = list(data_path.glob('*.json'))
        file_names = [f.name for f in files]

        # 检查TEMPREASON特征
        tempreason_patterns = ['_l2_', '_l3_']
        if any(pattern in file_name for file_name in file_names for pattern in tempreason_patterns):
            return 'tempreason'

        # 检查TIMEQA特征
        timeqa_patterns = ['train_processed', 'dev_processed', 'test_processed']
        if any(pattern in file_name for file_name in file_names for pattern in timeqa_patterns):
            return 'timeqa'

        return 'unknown'
