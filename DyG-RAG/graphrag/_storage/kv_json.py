import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        raw = load_json(self._file_name)
        self._data = raw or {}

        # Immediately write back to ensure file exists and is formatted correctly
        write_json(self._data, self._file_name)
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

        # 向后兼容性检查（新增）
        self._check_timing_metadata()

    def _check_timing_metadata(self):
        """检查数据中是否包含计时元数据"""
        if not self._data:
            return

        sample_size = min(10, len(self._data))
        sample_keys = list(self._data.keys())[:sample_size]

        has_timing = sum(1 for k in sample_keys if "_timing" in self._data[k])

        if has_timing == 0:
            logger.info(
                f"Storage '{self.namespace}' contains legacy data without timing metadata. "
                f"New data will include timing information."
            )
        elif has_timing < sample_size:
            logger.info(
                f"Storage '{self.namespace}' contains mixed data "
                f"({has_timing}/{sample_size} samples have timing metadata)"
            )

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}
