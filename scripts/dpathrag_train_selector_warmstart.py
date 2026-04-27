#!/usr/bin/env python3
"""Train a D-PathRAG autoregressive selector with gold-support warm-start labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.io import write_json, write_jsonl
from src.dpathrag.selector_data import (
    FEATURE_NAMES,
    IGNORE_TARGET,
    SelectorExample,
    featurize_selector_record,
    selection_overlap,
    summarize_selector_metrics,
    support_metrics_for_indices,
)
from src.dpathrag.selector import AutoregressivePathSelector, teacher_forced_path_nll


def load_jsonl(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= int(limit):
                    break
    return rows


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("query_idx") or "")


def split_rows(
    rows: list[dict[str, Any]],
    *,
    train_size: int,
    eval_size: int,
    eval_start: int,
    train_from_complement: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split rows for selector warm-start training.

    The default path preserves the historical contiguous ``train/eval`` split.
    Cross-fit runs set ``eval_start`` and ``train_from_complement`` so the held
    out fold is never used for warm-start training.
    """

    train_size = int(train_size)
    eval_size = int(eval_size)
    eval_start = int(eval_start)
    if eval_start < 0:
        train_rows = list(rows[:train_size])
        eval_rows = list(rows[train_size : train_size + eval_size])
    else:
        if eval_start >= len(rows):
            raise ValueError(f"eval_start={eval_start} is outside rows length {len(rows)}")
        eval_rows = list(rows[eval_start : eval_start + eval_size])
        if len(eval_rows) < eval_size:
            raise ValueError(f"Requested eval_size={eval_size} from eval_start={eval_start}, got {len(eval_rows)}")
        if train_from_complement:
            eval_qids = {row_qid(row) for row in eval_rows}
            train_rows = [row for row in rows if row_qid(row) not in eval_qids]
            if train_size > 0:
                train_rows = train_rows[:train_size]
        else:
            train_rows = list(rows[:train_size])

    train_qids = {row_qid(row) for row in train_rows}
    eval_qids = {row_qid(row) for row in eval_rows}
    metadata = {
        "eval_start": eval_start,
        "train_from_complement": bool(train_from_complement),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_unique_qids": len(train_qids),
        "eval_unique_qids": len(eval_qids),
        "train_eval_qid_overlap": len(train_qids & eval_qids),
    }
    return train_rows, eval_rows, metadata


class SelectorDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        max_candidates: int,
        path_len: int,
        shuffle_targets: bool,
        seed: int,
        extra_features_by_qid: dict[str, tuple[Any, Any]] | None = None,
    ) -> None:
        extra_features_by_qid = extra_features_by_qid or {}
        self.examples = [
            featurize_selector_record(
                record,
                max_candidates=max_candidates,
                path_len=path_len,
                candidate_extra_features=extra_features_by_qid.get(str(record.get("qid") or record.get("query_idx") or ""), (None, None))[0],
                query_extra_features=extra_features_by_qid.get(str(record.get("qid") or record.get("query_idx") or ""), (None, None))[1],
            )
            for record in records
        ]
        self.path_len = int(path_len)
        self.shuffle_targets = bool(shuffle_targets)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        example = self.examples[idx]
        targets = list(example.target_indices)
        valid_targets = [target for target in targets if target != IGNORE_TARGET]
        if self.shuffle_targets and len(valid_targets) > 1:
            self.rng.shuffle(valid_targets)
            targets = valid_targets[: self.path_len]
            while len(targets) < self.path_len:
                targets.append(IGNORE_TARGET)
        return {
            "features": torch.tensor(example.candidate_features, dtype=torch.float32),
            "query_features": torch.tensor(example.query_features, dtype=torch.float32),
            "candidate_mask": torch.tensor(example.candidate_mask, dtype=torch.bool),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def evaluate_selector(
    model: AutoregressivePathSelector,
    examples: list[SelectorExample],
    *,
    batch_size: int,
    path_len: int,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    model_rows: list[dict[str, float]] = []
    rank_rows: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start : start + int(batch_size)]
        features = torch.tensor([example.candidate_features for example in batch], dtype=torch.float32, device=device)
        query_features = torch.tensor([example.query_features for example in batch], dtype=torch.float32, device=device)
        candidate_mask = torch.tensor([example.candidate_mask for example in batch], dtype=torch.bool, device=device)
        with torch.no_grad():
            output = model(
                features,
                query_features=query_features,
                candidate_mask=candidate_mask,
                path_len=int(path_len),
                hard=True,
                add_gumbel_noise=False,
            )
        selected = output.selected_indices.detach().cpu().tolist()
        for example, selected_indices in zip(batch, selected):
            model_metric = support_metrics_for_indices(example, selected_indices)
            rank_indices = list(range(min(int(path_len), sum(1 for mask in example.candidate_mask if mask))))
            rank_metric = support_metrics_for_indices(example, rank_indices)
            model_metric["selection_overlap"] = selection_overlap(selected_indices, rank_indices)
            rank_metric["selection_overlap"] = 1.0
            model_rows.append(model_metric)
            rank_rows.append(rank_metric)
            prediction_rows.append(
                {
                    "qid": example.qid,
                    "gold_titles": example.gold_titles,
                    "selected_indices": selected_indices,
                    "selected_titles": [example.candidate_titles[index] for index in selected_indices],
                    "rank_topk_titles": [example.candidate_titles[index] for index in rank_indices],
                    "model": model_metric,
                    "rank_topk": rank_metric,
                }
            )
    model_summary = summarize_selector_metrics(model_rows)
    rank_summary = summarize_selector_metrics(rank_rows)
    return {"model": model_summary, "rank_topk": rank_summary}, prediction_rows


def load_embedding_features(path: str, *, limit: int = 0) -> tuple[dict[str, tuple[Any, Any]], list[str]]:
    if not path:
        return {}, []
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    qids = [str(qid) for qid in payload["qids"].tolist()]
    candidate_features = payload["candidate_features"]
    query_features = payload["query_features"]
    if limit > 0:
        qids = qids[: int(limit)]
        candidate_features = candidate_features[: int(limit)]
        query_features = query_features[: int(limit)]
    if len(qids) != candidate_features.shape[0] or len(qids) != query_features.shape[0]:
        raise ValueError("Embedding feature rows must align with qids")
    feature_names = [str(item) for item in payload["feature_names"].tolist()] if "feature_names" in payload.files else []
    return {
        qid: (candidate_features[idx].astype("float32"), query_features[idx].astype("float32"))
        for idx, qid in enumerate(qids)
    }, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_jsonl", default="data/dpathrag/cache/2wiki_proprag_pool100_smoke.jsonl")
    parser.add_argument("--output_dir", default="data/dpathrag/models/selector_warmstart_proprag_local1000")
    parser.add_argument("--report_json", default="reports/dpathrag/selector_warmstart_proprag_local1000.json")
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--eval_start", type=int, default=-1)
    parser.add_argument("--train_from_complement", action="store_true")
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--path_len", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding_npz", default="")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--shuffle_targets", action="store_true")
    parser.add_argument("--eval_each_epoch", action="store_true")
    parser.add_argument("--best_metric", choices=["support_complete", "support_recall", "bridge_entity_recall"], default="support_complete")
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--load_best_at_end", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

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
    if not train_rows or not eval_rows:
        raise ValueError("Both train and eval splits must be non-empty")
    if int(split_metadata["train_eval_qid_overlap"]) != 0:
        raise ValueError(f"Train/eval qid overlap is not allowed: {split_metadata['train_eval_qid_overlap']}")
    train_dataset = SelectorDataset(
        train_rows,
        max_candidates=int(args.max_candidates),
        path_len=int(args.path_len),
        shuffle_targets=bool(args.shuffle_targets),
        seed=int(args.seed),
        extra_features_by_qid=embedding_features,
    )
    eval_examples = [
        featurize_selector_record(
            record,
            max_candidates=int(args.max_candidates),
            path_len=int(args.path_len),
            candidate_extra_features=embedding_features.get(str(record.get("qid") or record.get("query_idx") or ""), (None, None))[0],
            query_extra_features=embedding_features.get(str(record.get("qid") or record.get("query_idx") or ""), (None, None))[1],
        )
        for record in eval_rows
    ]
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
    model = AutoregressivePathSelector(
        candidate_feature_dim=len(train_dataset.examples[0].candidate_features[0]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.01)
    report: dict[str, Any] = {
        "cache_jsonl": str(args.cache_jsonl),
        "embedding_npz": str(args.embedding_npz),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "split": split_metadata,
        "feature_names": FEATURE_NAMES + embedding_feature_names,
        "config": {
            "max_candidates": int(args.max_candidates),
            "path_len": int(args.path_len),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "shuffle_targets": bool(args.shuffle_targets),
            "eval_each_epoch": bool(args.eval_each_epoch),
            "best_metric": str(args.best_metric),
            "early_stop_patience": int(args.early_stop_patience),
            "load_best_at_end": bool(args.load_best_at_end),
        },
    }
    before_summary, _ = evaluate_selector(
        model,
        eval_examples,
        batch_size=int(args.eval_batch_size),
        path_len=int(args.path_len),
        device=device,
    )
    report["eval_before_train"] = before_summary

    epoch_losses: list[float] = []
    epoch_evals: list[dict[str, Any]] = []
    best_metric_value = float("-inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    epochs_run = 0
    for epoch in tqdm(range(int(args.epochs)), desc="Warm-start selector"):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            query_features = batch["query_features"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            targets = batch["targets"].to(device)
            output = model(
                features,
                query_features=query_features,
                candidate_mask=candidate_mask,
                path_len=int(args.path_len),
                hard=True,
                add_gumbel_noise=False,
            )
            loss = teacher_forced_path_nll(output.step_logits, targets, candidate_mask=candidate_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        epoch_losses.append(round(total_loss / max(1, total_batches), 6))
        epochs_run += 1
        if bool(args.eval_each_epoch):
            epoch_summary, _ = evaluate_selector(
                model,
                eval_examples,
                batch_size=int(args.eval_batch_size),
                path_len=int(args.path_len),
                device=device,
            )
            metric_value = float(epoch_summary["model"].get(str(args.best_metric), 0.0))
            epoch_evals.append({"epoch": int(epoch + 1), "loss": epoch_losses[-1], "eval": epoch_summary})
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_epoch = int(epoch + 1)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if int(args.early_stop_patience) > 0 and stale_epochs >= int(args.early_stop_patience):
                break

    if bool(args.load_best_at_end) and best_state is not None:
        model.load_state_dict(best_state)

    after_summary, prediction_rows = evaluate_selector(
        model,
        eval_examples,
        batch_size=int(args.eval_batch_size),
        path_len=int(args.path_len),
        device=device,
    )
    report["training"] = {
        "epoch_losses": epoch_losses,
        "final_epoch_loss": epoch_losses[-1] if epoch_losses else None,
        "epochs_run": int(epochs_run),
        "epoch_evals": epoch_evals,
        "best_epoch": int(best_epoch),
        "best_metric": str(args.best_metric),
        "best_metric_value": round(best_metric_value, 6) if best_metric_value != float("-inf") else None,
    }
    report["eval_after_train"] = after_summary
    model_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "config": report["config"],
        },
        model_dir / "selector.pt",
    )
    write_json(report, args.report_json)
    predictions_path = Path(args.predictions_jsonl) if args.predictions_jsonl else Path(args.report_json).with_suffix(".predictions.jsonl")
    write_jsonl(prediction_rows, predictions_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
