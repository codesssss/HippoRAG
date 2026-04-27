#!/usr/bin/env python3
"""Run a minimal FiD-lite reader-aware Stage 2 pilot for D-PathRAG.

This is a go/no-go diagnostic, not a final training recipe.  The reader is
frozen.  Gradients flow from answer NLL through soft selector weights that gate
per-document T5 encoder states.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from src.dpathrag.io import write_json
from src.dpathrag.reader import gold_answers
from src.dpathrag.selector import AutoregressivePathSelector
from src.dpathrag.selector_data import (
    FEATURE_NAMES,
    SelectorExample,
    featurize_selector_record,
    summarize_selector_metrics,
    support_metrics_for_indices,
)
from dpathrag_train_selector_warmstart import load_embedding_features, load_jsonl, split_rows


def candidate_body(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("text") or "")
    return text.split("\n", 1)[1] if "\n" in text else text


def load_selector(
    checkpoint_path: str | Path,
    *,
    feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    device: Any,
) -> AutoregressivePathSelector:
    import torch

    model = AutoregressivePathSelector(
        candidate_feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        dropout=float(dropout),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def build_doc_prompt(question: str, candidate: dict[str, Any]) -> str:
    title = str(candidate.get("title") or "")
    body = candidate_body(candidate)
    return "\n".join(
        [
            "Answer the question using only the provided document.",
            "",
            f"Question: {question}",
            "",
            "Document:",
            f"[1] {title}",
            body,
            "",
            "Answer:",
        ]
    )


class Stage2Dataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        max_candidates: int,
        path_len: int,
        extra_features_by_qid: dict[str, tuple[Any, Any]] | None = None,
    ) -> None:
        self.records = records
        self.examples: list[SelectorExample] = []
        extra_features_by_qid = extra_features_by_qid or {}
        for record in records:
            qid = str(record.get("qid") or record.get("query_idx") or "")
            candidate_extra, query_extra = extra_features_by_qid.get(qid, (None, None))
            self.examples.append(
                featurize_selector_record(
                    record,
                    max_candidates=max_candidates,
                    path_len=path_len,
                    candidate_extra_features=candidate_extra,
                    query_extra_features=query_extra,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"record": self.records[idx], "example": self.examples[idx]}


def collate_stage2(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    examples = [item["example"] for item in batch]
    return {
        "records": [item["record"] for item in batch],
        "examples": examples,
        "features": torch.tensor([example.candidate_features for example in examples], dtype=torch.float32),
        "query_features": torch.tensor([example.query_features for example in examples], dtype=torch.float32),
        "candidate_mask": torch.tensor([example.candidate_mask for example in examples], dtype=torch.bool),
    }


def encode_docs(
    reader: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_candidates: int,
    max_doc_tokens: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    prompts: list[str] = []
    for record in records:
        candidates = list(record.get("candidates") or [])[: int(max_candidates)]
        prompts.extend(build_doc_prompt(str(record.get("question") or ""), candidate) for candidate in candidates)
        while len(candidates) < int(max_candidates):
            prompts.append(build_doc_prompt(str(record.get("question") or ""), {"title": "", "text": ""}))
            candidates.append({})
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(max_doc_tokens),
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    batch_size = len(records)
    with torch.no_grad():
        hidden = reader.get_encoder()(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        ).last_hidden_state
    seq_len = int(hidden.shape[1])
    hidden = hidden.view(batch_size, int(max_candidates), seq_len, int(hidden.shape[-1]))
    attention = encoded["attention_mask"].view(batch_size, int(max_candidates), seq_len)
    return hidden, attention


def encode_labels(tokenizer: Any, records: list[dict[str, Any]], *, max_target_tokens: int, device: Any) -> Any:
    labels = tokenizer(
        text_target=[gold_answers(record)[0] for record in records],
        padding=True,
        truncation=True,
        max_length=int(max_target_tokens),
        return_tensors="pt",
    )["input_ids"].to(device)
    labels = labels.masked_fill(labels.eq(tokenizer.pad_token_id), -100)
    return labels


def selector_doc_weights(output: Any, *, eps: float = 1e-8) -> Any:
    weights = output.path_mask.clamp_min(0.0)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return weights / denom


def answer_nll(
    reader: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    selector_output: Any,
    *,
    max_candidates: int,
    max_doc_tokens: int,
    max_target_tokens: int,
    device: Any,
) -> Any:
    from transformers.modeling_outputs import BaseModelOutput

    doc_hidden, doc_attention = encode_docs(
        reader,
        tokenizer,
        records,
        max_candidates=int(max_candidates),
        max_doc_tokens=int(max_doc_tokens),
        device=device,
    )
    weights = selector_doc_weights(selector_output)
    fused_hidden = (doc_hidden * weights[:, :, None, None]).reshape(doc_hidden.shape[0], -1, doc_hidden.shape[-1])
    fused_attention = doc_attention.reshape(doc_attention.shape[0], -1)
    labels = encode_labels(tokenizer, records, max_target_tokens=int(max_target_tokens), device=device)
    return reader(
        encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
        attention_mask=fused_attention,
        labels=labels,
        return_dict=True,
    ).loss


def categorical_kl(current: Any, reference: Any, mask: Any, *, eps: float = 1e-8) -> float:
    import torch

    current = current.clamp_min(float(eps))
    reference = reference.clamp_min(float(eps))
    step_mask = mask.unsqueeze(1).expand_as(current)
    kl = current * (current.log() - reference.log())
    denom = step_mask.float().sum().clamp_min(1.0)
    return float((kl.masked_fill(~step_mask, 0.0).sum() / denom).detach().cpu())


def selected_change_stats(examples: list[SelectorExample], current_indices: list[list[int]], reference_indices: list[list[int]]) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for example, current, reference in zip(examples, current_indices, reference_indices):
        current_set = {int(index) for index in current}
        reference_set = {int(index) for index in reference}
        added = current_set - reference_set
        removed = reference_set - current_set
        gold = {idx for idx, label in enumerate(example.candidate_gold_support) if int(label) == 1}
        rows.append(
            {
                "changed_docs": float(len(added | removed)),
                "added_gold": float(len(added & gold)),
                "removed_gold": float(len(removed & gold)),
                "added_non_gold": float(len(added - gold)),
                "removed_non_gold": float(len(removed - gold)),
            }
        )
    denom = float(max(1, len(rows)))
    return {key: round(sum(row[key] for row in rows) / denom, 4) for key in rows[0]} if rows else {}


def summarize_hard_selection(examples: list[SelectorExample], selected_indices: list[list[int]]) -> dict[str, Any]:
    metrics = [support_metrics_for_indices(example, indices) for example, indices in zip(examples, selected_indices)]
    return summarize_selector_metrics(metrics)


def grad_norm(parameters: Any) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        value = float(param.grad.detach().data.norm(2).cpu())
        total += value * value
    return math.sqrt(total)


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# D-PathRAG FiD-Lite Stage 2 Pilot",
        "",
        f"- Steps requested: `{report['config']['steps']}`",
        f"- Steps run: `{report['summary']['steps_run']}`",
        f"- Final loss: `{report['summary']['final_loss']}`",
        f"- Mean gradient norm: `{report['summary']['mean_grad_norm']}`",
        f"- Final KL drift: `{report['summary']['final_kl_drift']}`",
        "",
        "## Final Support Metrics",
        "",
        "| Metric | Warm-start | Stage 2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    warm = report["summary"]["warm_support"]
    stage = report["summary"]["stage2_support"]
    for key in ("support_recall", "support_complete", "selected_gold_count", "bridge_entity_recall"):
        lines.append(f"| {key} | {warm.get(key, 0.0):.4f} | {stage.get(key, 0.0):.4f} | {stage.get(key, 0.0) - warm.get(key, 0.0):+.4f} |")
    lines.extend(["", "## Recent Logs", "", "| Step | Loss | KL | Grad Norm | Support Complete | Changed Docs | Removed Gold |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["logs"][-10:]:
        change = row.get("change", {})
        support = row.get("stage2_support", {})
        lines.append(
            f"| {row['step']} | {row['loss']:.4f} | {row['kl_drift']:.4f} | {row['grad_norm']:.4f} | "
            f"{support.get('support_complete', 0.0):.4f} | {change.get('changed_docs', 0.0):.4f} | {change.get('removed_gold', 0.0):.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--embedding_npz", default="data/dpathrag/cache/selector_embedding_proprag_local1000_rp64.npz")
    parser.add_argument("--selector_checkpoint", default="data/dpathrag/models/kfold_proprag/fold_0/selector.pt")
    parser.add_argument("--reader_model", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--report_json", default="reports/dpathrag/stage2_fid_lite_pilot100.json")
    parser.add_argument("--report_md", default="reports/dpathrag/stage2_fid_lite_pilot100.md")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--train_from_complement", action="store_true")
    parser.add_argument("--pilot_train_limit", type=int, default=200)
    parser.add_argument("--pilot_eval_limit", type=int, default=64)
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--path_len", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max_doc_tokens", type=int, default=128)
    parser.add_argument("--max_target_tokens", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = load_jsonl(args.cache_jsonl, limit=int(args.limit))
    embedding_features, embedding_feature_names = load_embedding_features(str(args.embedding_npz), limit=int(args.limit))
    train_rows, eval_rows, split_metadata = split_rows(
        rows,
        train_size=int(args.train_size),
        eval_size=int(args.eval_size),
        eval_start=int(args.eval_start),
        train_from_complement=bool(args.train_from_complement),
    )
    train_rows = train_rows[: int(args.pilot_train_limit)]
    eval_rows = eval_rows[: int(args.pilot_eval_limit)]
    train_dataset = Stage2Dataset(
        train_rows,
        max_candidates=int(args.max_candidates),
        path_len=int(args.path_len),
        extra_features_by_qid=embedding_features,
    )
    eval_dataset = Stage2Dataset(
        eval_rows,
        max_candidates=int(args.max_candidates),
        path_len=int(args.path_len),
        extra_features_by_qid=embedding_features,
    )
    if not train_dataset.examples or not eval_dataset.examples:
        raise ValueError("Stage 2 pilot needs non-empty train and eval rows")
    feature_dim = len(train_dataset.examples[0].candidate_features[0])
    selector = load_selector(
        args.selector_checkpoint,
        feature_dim=feature_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        device=device,
    )
    warm_selector = copy.deepcopy(selector).to(device)
    warm_selector.eval()
    for param in warm_selector.parameters():
        param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.reader_model)
    reader = AutoModelForSeq2SeqLM.from_pretrained(args.reader_model).to(device)
    reader.eval()
    for param in reader.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(args.learning_rate), weight_decay=0.01)
    loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_stage2)
    logs: list[dict[str, Any]] = []
    steps_run = 0
    last_loss = 0.0
    grad_norms: list[float] = []

    while steps_run < int(args.steps):
        for batch in loader:
            selector.train()
            features = batch["features"].to(device)
            query_features = batch["query_features"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            output = selector(
                features,
                query_features=query_features,
                candidate_mask=candidate_mask,
                path_len=int(args.path_len),
                tau=float(args.tau),
                hard=False,
                add_gumbel_noise=True,
            )
            loss = answer_nll(
                reader,
                tokenizer,
                batch["records"],
                output,
                max_candidates=int(args.max_candidates),
                max_doc_tokens=int(args.max_doc_tokens),
                max_target_tokens=int(args.max_target_tokens),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            current_grad_norm = grad_norm(selector.parameters())
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            steps_run += 1
            last_loss = float(loss.detach().cpu())
            grad_norms.append(float(current_grad_norm))

            if steps_run == 1 or steps_run % int(args.log_every) == 0 or steps_run >= int(args.steps):
                selector.eval()
                with torch.no_grad():
                    current_hard = selector(
                        features,
                        query_features=query_features,
                        candidate_mask=candidate_mask,
                        path_len=int(args.path_len),
                        hard=True,
                        add_gumbel_noise=False,
                    )
                    warm_hard = warm_selector(
                        features,
                        query_features=query_features,
                        candidate_mask=candidate_mask,
                        path_len=int(args.path_len),
                        hard=True,
                        add_gumbel_noise=False,
                    )
                    current_soft = selector(
                        features,
                        query_features=query_features,
                        candidate_mask=candidate_mask,
                        path_len=int(args.path_len),
                        hard=False,
                        add_gumbel_noise=False,
                    )
                    warm_soft = warm_selector(
                        features,
                        query_features=query_features,
                        candidate_mask=candidate_mask,
                        path_len=int(args.path_len),
                        hard=False,
                        add_gumbel_noise=False,
                    )
                current_indices = current_hard.selected_indices.detach().cpu().tolist()
                warm_indices = warm_hard.selected_indices.detach().cpu().tolist()
                logs.append(
                    {
                        "step": int(steps_run),
                        "loss": round(last_loss, 6),
                        "grad_norm": round(float(current_grad_norm), 6),
                        "kl_drift": round(categorical_kl(current_soft.step_probs, warm_soft.step_probs, candidate_mask), 6),
                        "stage2_support": summarize_hard_selection(batch["examples"], current_indices),
                        "warm_support": summarize_hard_selection(batch["examples"], warm_indices),
                        "change": selected_change_stats(batch["examples"], current_indices, warm_indices),
                    }
                )
            if steps_run >= int(args.steps):
                break

    def eval_selector(model: AutoregressivePathSelector) -> tuple[dict[str, Any], list[list[int]]]:
        eval_loader = DataLoader(eval_dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_stage2)
        all_metrics: list[dict[str, float]] = []
        all_indices: list[list[int]] = []
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                output = model(
                    batch["features"].to(device),
                    query_features=batch["query_features"].to(device),
                    candidate_mask=batch["candidate_mask"].to(device),
                    path_len=int(args.path_len),
                    hard=True,
                    add_gumbel_noise=False,
                )
                selected = output.selected_indices.detach().cpu().tolist()
                all_indices.extend(selected)
                all_metrics.extend(
                    support_metrics_for_indices(example, indices)
                    for example, indices in zip(batch["examples"], selected)
                )
        return summarize_selector_metrics(all_metrics), all_indices

    stage2_support, stage2_indices = eval_selector(selector)
    warm_support, warm_indices = eval_selector(warm_selector)
    summary = {
        "steps_run": int(steps_run),
        "final_loss": round(float(last_loss), 6),
        "mean_grad_norm": round(sum(grad_norms) / max(1, len(grad_norms)), 6),
        "final_kl_drift": logs[-1]["kl_drift"] if logs else 0.0,
        "warm_support": warm_support,
        "stage2_support": stage2_support,
        "eval_change": selected_change_stats(eval_dataset.examples, stage2_indices, warm_indices),
    }
    report = {
        "cache_jsonl": str(args.cache_jsonl),
        "embedding_npz": str(args.embedding_npz),
        "selector_checkpoint": str(args.selector_checkpoint),
        "reader_model": str(args.reader_model),
        "split": split_metadata,
        "feature_names": FEATURE_NAMES + embedding_feature_names,
        "config": vars(args),
        "summary": summary,
        "logs": logs,
    }
    write_json(report, args.report_json)
    write_markdown(report, args.report_md)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
