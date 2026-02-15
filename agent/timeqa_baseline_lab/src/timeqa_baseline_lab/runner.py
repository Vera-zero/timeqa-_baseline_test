from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from inspect import Parameter, signature
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm

from .chunking import TokenChunker
from .config import ExperimentConfig
from .data import append_jsonl, iter_jsonl, load_corpus, load_questions_from_arrow
from .evaluation import em_f1, mean, substring_recall
from .llm import build_generator
from .retriever import ContrieverRetriever
from .strategies import (
    SYSTEM_PROMPT,
    build_rag_cot_prompt,
    build_zero_shot_cot_prompt,
    build_zero_shot_prompt,

    qaap,
    rag_cot,
    react,
    zero_shot,
    zero_shot_cot,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _answer_text(raw: str) -> str:
    for key in ["Final Answer:", "Answer:"]:
        if key in raw:
            return raw.split(key, 1)[1].strip()
    return raw.strip()


def _targets_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def _dispatch(strategy: str):
    table = {
        "zero_shot": zero_shot,
        "zero_shot_cot": zero_shot_cot,
        "rag_cot": rag_cot,
        "react": react,
        "qaap": qaap,

    }
    if strategy not in table:
        raise ValueError(f"Unknown strategy: {strategy}")
    return table[strategy]


def _is_interactive_strategy(strategy: str) -> bool:
    return strategy in {"react", "qaap"}


def _is_batchable_strategy(strategy: str) -> bool:
    return strategy in {"zero_shot", "zero_shot_cot", "rag_cot"}


def _batch_prompt(strategy: str, question: str, retrieved) -> str:
    if strategy == "zero_shot":
        return build_zero_shot_prompt(question)
    if strategy == "zero_shot_cot":
        return build_zero_shot_cot_prompt(question)
    if strategy == "rag_cot":
        return build_rag_cot_prompt(question, retrieved)
    raise ValueError(f"Strategy is not batch-promptable: {strategy}")


def _filter_strategy_kwargs(fn, params: Dict[str, object]) -> Dict[str, object]:
    if not params:
        return {}
    sig = signature(fn)
    allowed = {}
    for name, val in params.items():
        p = sig.parameters.get(name)
        if p is None:
            continue
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            allowed[name] = val
    return allowed


def _load_done_ids(result_path: Path) -> Set[str]:
    done = set()
    for row in iter_jsonl(result_path):
        done.add(row["idx"])
    return done


def run_experiment(cfg: ExperimentConfig) -> Dict[str, object]:
    _seed_everything(cfg.run.seed)

    strategy_name = cfg.run.strategy
    run_tag = f"{cfg.io.run_name}_{strategy_name}"
    result_path = Path(cfg.io.output_dir) / f"{run_tag}.jsonl"
    metric_path = Path(cfg.io.output_dir) / f"{run_tag}_metrics.json"
    config_path = Path(cfg.io.output_dir) / f"{run_tag}_config_used.json"

    config_snapshot = {
        "data": asdict(cfg.data),
        "model": asdict(cfg.model),
        "retriever": asdict(cfg.retriever),
        "chunk": asdict(cfg.chunk),
        "run": asdict(cfg.run),
        "io": asdict(cfg.io),
    }
    config_path.write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    questions = load_questions_from_arrow(cfg.data.question_arrow_path, limit=cfg.run.max_questions)

    llm = build_generator(cfg.model)

    need_retrieval = strategy_name in {"rag_cot", "react", "qaap"}
    retriever = None
    if need_retrieval:
        tokenizer_name = cfg.model.model_name or "Qwen/Qwen3-4B-Instruct-2507"
        docs = load_corpus(cfg.data.corpus_path)
        chunker = TokenChunker(
            tokenizer_name=tokenizer_name,
            chunk_size=cfg.chunk.chunk_size,
            chunk_overlap=cfg.chunk.chunk_overlap,
            min_chunk_size=cfg.chunk.min_chunk_size,
        )
        chunks = chunker.chunk_corpus(docs)
        retriever = ContrieverRetriever(
            model_name=cfg.retriever.model_name,
            device=cfg.model.device,
        )
        cache_dir = Path(cfg.io.cache_dir) / "retriever"
        retriever.build_or_load_index(chunks, str(cache_dir))

    done_ids = _load_done_ids(result_path) if (cfg.run.resume and result_path.exists()) else set()

    fn = _dispatch(strategy_name)
    strategy_kwargs = _filter_strategy_kwargs(fn, cfg.run.strategy_params)
    started = time.time()
    pending_questions = [q for q in questions if q.idx not in done_ids]

    configured_batch_size = max(1, int(cfg.model.batch_size))
    effective_batch_size = configured_batch_size
    if _is_interactive_strategy(strategy_name):
        effective_batch_size = 1
    if not _is_batchable_strategy(strategy_name):
        effective_batch_size = 1
    if strategy_kwargs:
        # keep deterministic behavior for custom strategy kwargs
        effective_batch_size = 1

    if effective_batch_size > 1:
        for i in tqdm(range(0, len(pending_questions), effective_batch_size), desc=f"Running {strategy_name} (batch)"):
            batch_questions = pending_questions[i : i + effective_batch_size]

            retrieved_batches = []
            for q in batch_questions:
                if need_retrieval:
                    retrieved = retriever.search(q.question, top_k=cfg.retriever.top_k)
                else:
                    retrieved = []
                retrieved_batches.append(retrieved)

            prompts = [
                _batch_prompt(strategy_name, q.question, retrieved_batches[idx])
                for idx, q in enumerate(batch_questions)
            ]
            raw_answers = llm.batch_generate(prompts, system_prompt=SYSTEM_PROMPT, batch_size=effective_batch_size)

            for q, retrieved, raw_answer in zip(batch_questions, retrieved_batches, raw_answers):
                pred = _answer_text(raw_answer)
                targets = _targets_list(q.targets)
                qa = em_f1(pred, targets)
                em = float(qa["em"])
                f1 = float(qa["f1"])
                sub_recall = substring_recall(pred, targets)

                row = {
                    "idx": q.idx,
                    "question": q.question,
                    "targets": targets,
                    "prediction": pred,
                    "raw_output": raw_answer,
                    # Keep `score` for backward compatibility with old recall-style runs.
                    "score": sub_recall,
                    "em": em,
                    "f1": f1,
                    "substring_recall": sub_recall,
                    "strategy": strategy_name,
                    "retrieved": [
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "title": c.title,
                            "source_idx": c.source_idx,
                            "text": c.text,
                        }
                        for c in retrieved
                    ],
                    "trace": [],
                }
                append_jsonl(result_path, row)
    else:
        for q in tqdm(pending_questions, desc=f"Running {strategy_name}"):
            if need_retrieval:
                out = fn(llm, retriever, q.question, cfg.retriever.top_k, **strategy_kwargs)
            else:
                out = fn(llm, q.question, **strategy_kwargs)

            pred = _answer_text(out.answer)
            targets = _targets_list(q.targets)
            qa = em_f1(pred, targets)
            em = float(qa["em"])
            f1 = float(qa["f1"])
            sub_recall = substring_recall(pred, targets)

            row = {
                "idx": q.idx,
                "question": q.question,
                "targets": targets,
                "prediction": pred,
                "raw_output": out.answer,
                # Keep `score` for backward compatibility with old recall-style runs.
                "score": sub_recall,
                "em": em,
                "f1": f1,
                "substring_recall": sub_recall,
                "strategy": strategy_name,
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "source_idx": c.source_idx,
                        "text": c.text,
                    }
                    for c in out.retrieved
                ],
                "trace": out.trace,
            }
            append_jsonl(result_path, row)

    em_values: List[float] = []
    f1_values: List[float] = []
    recall_values: List[float] = []
    if result_path.exists():
        for row in iter_jsonl(result_path):
            if "em" in row and "f1" in row:
                em_val = float(row["em"])
                f1_val = float(row["f1"])
            else:
                pred = str(row.get("prediction", ""))
                targets = _targets_list(row.get("targets", []))
                qa = em_f1(pred, targets)
                em_val = float(qa["em"])
                f1_val = float(qa["f1"])

            if "substring_recall" in row:
                recall_val = float(row["substring_recall"])
            else:
                pred = str(row.get("prediction", ""))
                targets = _targets_list(row.get("targets", []))
                recall_val = substring_recall(pred, targets)

            em_values.append(em_val)
            f1_values.append(f1_val)
            recall_values.append(recall_val)

    metrics = {
        "strategy": strategy_name,
        "count": len(em_values),
        "avg_em": mean(em_values),
        "avg_f1": mean(f1_values),
        "avg_substring_recall": mean(recall_values),
        "avg_recall": mean(recall_values),
        "metric_source": "Aligned with qaap-main/utils.py get_metrics + ReAct token-F1 style normalization",
        "elapsed_sec": round(time.time() - started, 3),
        "result_path": str(result_path),
        "config_used_path": str(config_path),
        "strategy_params": cfg.run.strategy_params,
        "model_provider": cfg.model.provider,
        "batch_size_configured": configured_batch_size,
        "batch_size_effective": effective_batch_size,
    }
    metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics