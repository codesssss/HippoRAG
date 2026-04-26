#!/usr/bin/env python3
"""Fine-tune a HuggingFace seq2seq reader on D-PathRAG reader JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dpathrag.reader import format_reader_input, gold_answers, load_reader_jsonl, score_prediction, summarize_scores, write_jsonl
from src.dpathrag.io import write_json


class ReaderDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_input_tokens: int,
        max_target_tokens: int,
        max_docs: int,
        max_doc_chars: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_tokens = int(max_input_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.max_docs = int(max_docs)
        self.max_doc_chars = int(max_doc_chars)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        source = format_reader_input(record, max_docs=self.max_docs, max_doc_chars=self.max_doc_chars)
        target = gold_answers(record)[0]
        model_inputs = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        labels = self.tokenizer(
            text_target=target,
            truncation=True,
            max_length=self.max_target_tokens,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def evaluate_model(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    max_docs: int,
    max_doc_chars: int,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    scored: list[dict[str, Any]] = []
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        inputs = [format_reader_input(row, max_docs=max_docs, max_doc_chars=max_doc_chars) for row in batch]
        encoded = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=int(max_input_tokens),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(**encoded, max_new_tokens=int(max_new_tokens))
        predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        scored.extend(score_prediction(row, pred) for row, pred in zip(batch, predictions))
    return summarize_scores(scored), scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", default="data/dpathrag/reader_baselines/2wiki_train_gold_k5_5k.jsonl")
    parser.add_argument("--eval_jsonl", default="data/dpathrag/reader_baselines/2wiki_validation_gold_k5_1k.jsonl")
    parser.add_argument("--model_name_or_path", default="google/flan-t5-base")
    parser.add_argument("--output_dir", default="data/dpathrag/models/flan_t5_base_gold_k5_5k")
    parser.add_argument("--report_json", default="reports/dpathrag/reader_flan_t5_base_finetune_gold_k5.json")
    parser.add_argument("--train_limit", type=int, default=5000)
    parser.add_argument("--eval_limit", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_target_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_before_train", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

    torch.manual_seed(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name_or_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_name_or_path))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    train_records = load_reader_jsonl(args.train_jsonl, limit=int(args.train_limit))
    eval_records = load_reader_jsonl(args.eval_jsonl, limit=int(args.eval_limit))
    train_dataset = ReaderDataset(
        train_records,
        tokenizer,
        max_input_tokens=int(args.max_input_tokens),
        max_target_tokens=int(args.max_target_tokens),
        max_docs=int(args.max_docs),
        max_doc_chars=int(args.max_doc_chars),
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    total_update_steps = max(1, math.ceil(len(train_loader) * float(args.epochs) / int(args.gradient_accumulation_steps)))
    warmup_steps = max(1, int(0.03 * total_update_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    report: dict[str, Any] = {
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "model_name_or_path": str(args.model_name_or_path),
        "output_dir": str(args.output_dir),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "config": {
            "epochs": float(args.epochs),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "max_input_tokens": int(args.max_input_tokens),
            "max_target_tokens": int(args.max_target_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "total_update_steps": int(total_update_steps),
        },
    }
    if bool(args.eval_before_train):
        before_summary, before_rows = evaluate_model(
            model,
            tokenizer,
            eval_records,
            batch_size=int(args.eval_batch_size),
            max_input_tokens=int(args.max_input_tokens),
            max_new_tokens=int(args.max_new_tokens),
            max_docs=int(args.max_docs),
            max_doc_chars=int(args.max_doc_chars),
            device=device,
        )
        report["eval_before_train"] = before_summary
        write_jsonl(before_rows, Path(args.report_json).with_suffix(".before.predictions.jsonl"))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    loss_sum = 0.0
    micro_step = 0
    target_micro_steps = int(math.ceil(len(train_loader) * float(args.epochs)))
    progress = tqdm(total=target_micro_steps, desc="Fine-tuning reader")
    while micro_step < target_micro_steps:
        for batch in train_loader:
            if micro_step >= target_micro_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / int(args.gradient_accumulation_steps)
            loss.backward()
            loss_sum += float(outputs.loss.detach().cpu())
            micro_step += 1
            progress.update(1)
            if micro_step % int(args.gradient_accumulation_steps) == 0 or micro_step >= target_micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
    progress.close()

    eval_summary, eval_rows = evaluate_model(
        model,
        tokenizer,
        eval_records,
        batch_size=int(args.eval_batch_size),
        max_input_tokens=int(args.max_input_tokens),
        max_new_tokens=int(args.max_new_tokens),
        max_docs=int(args.max_docs),
        max_doc_chars=int(args.max_doc_chars),
        device=device,
    )
    report["training"] = {
        "micro_steps": int(micro_step),
        "update_steps": int(update_step),
        "avg_loss_per_micro_step": round(loss_sum / max(1, micro_step), 6),
    }
    report["eval_after_train"] = eval_summary
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_json(report, args.report_json)
    write_jsonl(eval_rows, Path(args.report_json).with_suffix(".predictions.jsonl"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

