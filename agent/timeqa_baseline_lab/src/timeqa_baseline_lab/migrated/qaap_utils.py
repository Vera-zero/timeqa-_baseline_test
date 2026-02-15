from __future__ import annotations

import datetime as dt
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple


def create_context_slices(context: str, max_length: int = 512, stride: int = 384) -> List[str]:
    """Migrated from qaap-main/utils.py with minor typing cleanups."""
    context_paras = context.split("\n")
    context_tokens: List[str] = []
    slices: List[str] = []
    for para in context_paras:
        context_tokens += para.split(" ")
        context_tokens.append("\n")

    while len(context_tokens) > max_length:
        slices.append(" ".join(context_tokens[:max_length]))
        context_tokens = context_tokens[stride:]
    slices.append(" ".join(context_tokens))
    return slices


def extract_code_from_string(text: str) -> str | None:
    """Extract python fenced code blocks from model output."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return None


def extract_answer(answer_key: str, information: List[Dict[str, Any]]) -> List[str]:
    if not information or information[0] == "":
        return [""]

    answer_info = information[0]
    if answer_key not in answer_info or answer_info[answer_key] is None:
        return [""]

    value = answer_info[answer_key]
    if isinstance(value, (list, tuple, frozenset)) and value:
        if isinstance(value[0], str):
            predictions = list(value)
        elif isinstance(value[0], dict):
            predictions = list(value[0].values())
        else:
            predictions = [str(v) for v in value]
    elif isinstance(value, dict):
        predictions = [str(v) for v in value.values()]
    else:
        predictions = [str(value)]

    return predictions if predictions else [""]


def calc_time_iou(code_blocks: List[str]) -> Tuple[str, List[Dict[str, Any]] | List[str]]:
    """Migrated from qaap-main/utils.py with safe fallbacks."""
    locals_: Dict[str, Any] = {"information": []}

    try:
        _safe_exec(code_blocks[0], locals_)
        query = locals_.get("query")
        answer_key = locals_.get("answer_key")
    except Exception:
        query = None
        answer_key = None

    for c in code_blocks[1:]:
        try:
            _safe_exec(c, locals_)
        except Exception:
            continue

    default_start = datetime(1, 1, 1)
    default_end = datetime(3000, 1, 1)
    information = locals_.get("information", [])

    if query is None:
        return "object", information
    if answer_key is None:
        answer_key = "object"

    if (
        "time" not in query
        or query["time"] is None
        or (
            isinstance(query["time"], dict)
            and "start" in query["time"]
            and "end" in query["time"]
            and query["time"]["start"] is None
            and query["time"]["end"] is None
        )
    ):
        query["time"] = {"start": default_start, "end": default_end}
        time_type = "overlap"
    elif isinstance(query["time"], datetime):
        query["time"] = {"start": query["time"], "end": query["time"] + dt.timedelta(365)}
        time_type = "overlap"
    elif "start" not in query["time"] or query["time"]["start"] is None:
        time_type = "before or end"
    elif "end" not in query["time"] or query["time"]["end"] is None:
        time_type = "after or start"
    else:
        time_type = "overlap"

    information = [
        x
        for x in information
        if isinstance(x, dict) and "subject" in x and "object" in x and "relation" in x and x.get(answer_key) is not None
    ]
    if len(information) == 0:
        return "object", [""]

    for ex in information:
        try:
            if (
                "time" not in ex
                or ex["time"] is None
                or (
                    isinstance(ex["time"], dict)
                    and "start" in ex["time"]
                    and "end" in ex["time"]
                    and ex["time"]["start"] is None
                    and ex["time"]["end"] is None
                )
            ):
                ex["time"] = {"start": default_start, "end": default_end}
            elif isinstance(ex["time"], datetime):
                ex["time"] = {"start": ex["time"], "end": ex["time"] + dt.timedelta(365)}
            elif len(ex["time"]) == 0:
                ex["time"] = {"start": default_start, "end": default_end}

            if "start" not in ex["time"] or ex["time"]["start"] is None:
                ex["time"].update(start=default_start)
            if "end" not in ex["time"] or ex["time"]["end"] is None:
                ex["time"].update(end=default_end)
        except Exception:
            continue

    information = [x for x in information if x.get("time") is not None]
    if time_type == "overlap":
        for ex in information:
            latest_start = max(query["time"]["start"], ex["time"]["start"])
            earliest_end = min(query["time"]["end"], ex["time"]["end"])
            delta = (earliest_end - latest_start).days + 1
            overlap = max(0, delta)
            time_union = max(
                (query["time"]["end"] - query["time"]["start"]).days
                + (ex["time"]["end"] - ex["time"]["start"]).days
                - overlap,
                1,
            )
            ex.update(overlap=overlap)
            ex.update(time_union=time_union)
            ex.update(time_iou=overlap / time_union)
        information = sorted(information, key=lambda x: (x["time_iou"], x["overlap"]), reverse=True)
    elif time_type == "after or start":
        information = sorted(information, key=lambda x: abs((x["time"]["start"] - query["time"]["start"]).days))
    elif time_type == "before or end":
        information = sorted(information, key=lambda x: abs((x["time"]["end"] - query["time"]["end"]).days))

    return answer_key, information


def _safe_exec(code: str, locals_: Dict[str, Any]) -> None:
    safe_globals = {
        "__builtins__": {},
        "datetime": datetime,
        "timedelta": dt.timedelta,
        "dt": dt,
    }
    exec(code, safe_globals, locals_)
