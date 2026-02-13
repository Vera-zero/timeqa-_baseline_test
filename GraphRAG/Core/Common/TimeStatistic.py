import time
from datetime import datetime
from typing import Dict, List, Optional


class TimeStatistic:
    def __init__(self):
        self._start_time = {}
        self._count = {}
        self._total_time = {}
        self._stage_time = []

        # 新增属性：用于记录详细的阶段计时
        self._stage_details: List[Dict] = []
        self._current_stage: Optional[str] = None
        self._stage_start_time: Optional[float] = None
        self._phase_timings: Dict[str, float] = {}  # 子阶段开始时间
        self._phase_durations: Dict[str, float] = {}  # 子阶段持续时间

    def start_stage(self, stage_name: str = None):
        """启动一个新的阶段计时

        Args:
            stage_name: 阶段名称（可选，用于详细计时）
        """
        self._stage_start_time = time.time()
        self._stage_time.append(self._stage_start_time)
        self._current_stage = stage_name
        self._phase_timings = {}
        self._phase_durations = {}
    
    def stop_last_stage(self):
        """Stop last stage and return the time taken

        Returns:
            INT : Time taken for the last stage
        """
        self._stage_time.append(time.time())
        inc_time = self._stage_time[-1] - self._stage_time[-2]
        return inc_time
    
    def start(self, name):
        self._start_time[name] = time.time()

    def end(self, name):
        if name in self._start_time:
            inc_time = time.time() - self._start_time[name]
            self._add_time(name, inc_time)
            del self._start_time[name]
        else:
            raise RuntimeError(f"TimeStatistic: {name} not started")
        return str(inc_time)

    def _add_time(self, name, inc_time):
        if name not in self._total_time:
            self._total_time[name] = 0
            self._count[name] = 0
        self._total_time[name] += inc_time
        self._count[name] += 1

    def get_statistics(self, name):
        if name in self._total_time:
            return {
                "Total  time(s)": self._total_time[name],
                "Count": self._count[name],
                "Average_time (s)": self._total_time[name] / self._count[name]
            }
        else:
            raise RuntimeError(f"TimeStatistic: {name} has no statistics")

    def start_named_phase(self, phase_name: str):
        """在当前阶段内启动一个子阶段计时

        Args:
            phase_name: 子阶段名称
        """
        if phase_name in self._phase_timings:
            raise RuntimeError(f"Phase '{phase_name}' already started")

        self._phase_timings[phase_name] = time.time()

    def end_named_phase(self, phase_name: str) -> float:
        """结束子阶段计时并返回耗时

        Args:
            phase_name: 子阶段名称

        Returns:
            耗时（秒）
        """
        if phase_name not in self._phase_timings:
            raise RuntimeError(f"Phase '{phase_name}' not started")

        elapsed = time.time() - self._phase_timings[phase_name]
        self._phase_durations[phase_name] = elapsed
        del self._phase_timings[phase_name]
        return elapsed

    def save_stage_details(self, stage_name: str):
        """保存当前阶段的所有计时数据到详细记录中

        Args:
            stage_name: 阶段名称
        """
        total_phase_time = sum(self._phase_durations.values())
        total_stage_time = time.time() - self._stage_start_time if self._stage_start_time else 0

        stage_record = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "phase_times": {
                **self._phase_durations,
                "total_time": total_phase_time
            },
            "total_time": total_stage_time
        }

        self._stage_details.append(stage_record)
        self._current_stage = None
        self._stage_start_time = None
        self._phase_durations = {}
        self._phase_timings = {}

    def get_stage_details(self) -> List[Dict]:
        """获取所有已保存的阶段计时数据

        Returns:
            阶段计时数据列表
        """
        return self._stage_details

    def clear_stage_details(self):
        """清空已保存的阶段计时数据"""
        self._stage_details = []