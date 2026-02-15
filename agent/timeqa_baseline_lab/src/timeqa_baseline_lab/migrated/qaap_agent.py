from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from ..chunking import Chunk
from ..llm import BaseGenerator
from ..retriever import ContrieverRetriever
from .qaap_utils import calc_time_iou, create_context_slices, extract_answer, extract_code_from_string


SYSTEM_PROMPT = "You are a precise QA assistant. Always follow the required output format."


def _asset_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / name


def _load_init_prompt() -> str:
    path = _asset_path("qaap_timeqa_prompt.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "prompt_text" in data:
            return data["prompt_text"]
    return (
        "Solve the question by parsing it into a Python query dict, then extract evidence as "
        "information.append({...}) entries and derive the final answer from temporal overlap."
    )


def _dedup_chunks(chunks: List[Chunk]) -> List[Chunk]:
    seen = set()
    out = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


def _format_context(chunks: List[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i}] {c.title}\n{c.text}")
    return "\n\n".join(parts)


def _parse_entities(search_code: str) -> List[str]:
    locals_ = {}
    safe_globals = {"__builtins__": {}}
    try:
        exec(search_code, safe_globals, locals_)
    except Exception:
        return []
    entities = locals_.get("entities_to_search", [])
    if isinstance(entities, str):
        return [entities]
    if isinstance(entities, list):
        return [str(x) for x in entities if str(x).strip()]
    return []


def run_qaap(
    llm: BaseGenerator,
    retriever: ContrieverRetriever,
    question: str,
    top_k: int,
    max_slice_length: int = 512,
    slice_stride: int = 384,
    max_extract_rounds: int = 6,
    max_entities: int = 5,
    max_slices_per_chunk: int = 2,
) -> Tuple[str, List[Chunk], List[str]]:
    """QAaP-style pipeline migrated from qaap-main (parse -> search -> extract -> execute)."""
    trace: List[str] = []
    init_prompt = _load_init_prompt()

    base_prompt = init_prompt + "\n\nQuestion:" + question + "\nQuestion parsing:\n"

    parsed_query_raw = llm.generate(
        base_prompt
        + "Output only one ```python``` block that defines `query` and `answer_key`.",
        system_prompt=SYSTEM_PROMPT,
    )
    trace.append("Question parsing:\n" + parsed_query_raw)

    parsed_query_code = extract_code_from_string(parsed_query_raw)
    if not parsed_query_code:
        parsed_query_code = (
            "query = {\"subject\": None, \"relation\": \"\", \"object\": None, "
            "\"time\": {\"start\": None, \"end\": None}}\n"
            "answer_key = \"object\""
        )

    search_prompt = (
        base_prompt
        + "```python\n"
        + parsed_query_code
        + "\n```\n"
        + "Search:\nOutput only one ```python``` block defining `entities_to_search` (a list of strings)."
    )
    parsed_search_raw = llm.generate(search_prompt, system_prompt=SYSTEM_PROMPT)
    trace.append("Search:\n" + parsed_search_raw)

    parsed_search_code = extract_code_from_string(parsed_search_raw) or "entities_to_search = []"
    entities = _parse_entities(parsed_search_code)

    if not entities:
        entities = [question]

    collected: List[Chunk] = []
    for entity in entities[: max(1, max_entities)]:
        collected.extend(retriever.search(entity, top_k=max(2, top_k // 2)))
    if not collected:
        collected = retriever.search(question, top_k=top_k)

    collected = _dedup_chunks(collected)[: max(top_k * 3, top_k)]

    extracted_code_blocks = [parsed_query_code]
    num_extract = 0

    for chunk in collected:
        slices = create_context_slices(chunk.text, max_length=max_slice_length, stride=slice_stride)
        for context_slice in slices[: max(1, max_slices_per_chunk)]:
            extract_prompt = (
                base_prompt
                + "```python\n"
                + parsed_query_code
                + "\n```\n"
                + "Context: "
                + context_slice
                + "\nExtract information relevant to the query:\n"
                + "Output either: 'There is nothing relevant to the query.' or one ```python``` block "
                + "with only `information.append({...})` statements."
            )
            res = llm.generate(extract_prompt, system_prompt=SYSTEM_PROMPT)
            trace.append("Extract:\n" + res)
            code = extract_code_from_string(res)
            if code:
                extracted_code_blocks.append(code)
            num_extract += 1
            if num_extract >= max(1, max_extract_rounds):
                break
        if num_extract >= max(1, max_extract_rounds):
            break

    answer = ""
    if len(extracted_code_blocks) > 1:
        try:
            answer_key, information = calc_time_iou(extracted_code_blocks)
            predictions = extract_answer(answer_key, information)  # type: ignore[arg-type]
            answer = str(predictions[0]).strip() if predictions else ""
        except Exception:
            answer = ""

    if not answer:
        fallback_prompt = (
            "Answer the question based on retrieved context.\n"
            f"Question: {question}\n\n"
            f"Context:\n{_format_context(collected[:top_k])}\n\n"
            "Output only: Final Answer: <answer>"
        )
        answer = llm.generate(fallback_prompt, system_prompt=SYSTEM_PROMPT)

    return answer, collected[:top_k], trace
