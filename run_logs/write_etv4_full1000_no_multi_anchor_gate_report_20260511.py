#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/mnt/nvme/code/HippoRAG")
CLEAN_ROOT = ROOT / "run_logs/etv4_clean_mainline_full1000_reuse_gpt4omini_20260511"
ABL_ROOT = ROOT / "run_logs/etv4_no_multi_anchor_precision_full1000_reuse_gpt4omini_20260511"
OUT_ROOT = ROOT / "run_logs/etv4_clean_vs_no_multi_anchor_full1000_20260511/reports"
DATASETS = ["2wikimultihopqa", "musique", "hotpotqa"]
METHOD = "evidence_transition_graphragv4_fact_witnessed_sto"


def load_report(base: Path, dataset: str) -> dict:
    path = base / dataset / "reports" / f"{dataset}_{METHOD}_retrieval.json"
    return json.loads(path.read_text(encoding="utf-8"))


def top5(row: dict) -> tuple[int, ...]:
    return tuple(int(x) for x in (row.get("retrieved_doc_indices_top5") or [])[:5])


def recall(row: dict) -> float:
    return float(row.get("query_grounded_sto_recall_at5") or 0.0)


def all_gold(row: dict) -> bool:
    return bool(row.get("query_grounded_sto_all_gold_at5"))


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "clean_root": str(CLEAN_ROOT),
        "ablation_root": str(ABL_ROOT),
        "ablation": "no_multi_anchor_precision",
        "datasets": [],
    }
    md = [
        "# ETv4 No Multi-Anchor Precision Full1000 Retrieval Gate",
        "",
        "| Dataset | Rows | Clean R@5 | No-rule R@5 | Delta R@5 | Clean all-gold@5 | No-rule all-gold@5 | Delta all-gold@5 | Top5 changed | Gold gains | Gold losses | Net gold losses | Decision signal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset in DATASETS:
        clean = load_report(CLEAN_ROOT, dataset)
        abl = load_report(ABL_ROOT, dataset)
        clean_rows = clean.get("rows", []) or []
        abl_rows = abl.get("rows", []) or []
        if len(clean_rows) != len(abl_rows):
            raise ValueError(f"row count mismatch for {dataset}: {len(clean_rows)} vs {len(abl_rows)}")

        changed = gains = losses = 0
        changed_examples = []
        for idx, (c, a) in enumerate(zip(clean_rows, abl_rows)):
            c_top = top5(c)
            a_top = top5(a)
            if c_top == a_top:
                continue
            changed += 1
            c_rec = recall(c)
            a_rec = recall(a)
            delta = a_rec - c_rec
            if delta > 1e-12:
                gains += 1
            elif delta < -1e-12:
                losses += 1
            if len(changed_examples) < 20:
                changed_examples.append(
                    {
                        "query_index": c.get("query_index", idx),
                        "question": c.get("question", ""),
                        "clean_top5": list(c_top),
                        "no_rule_top5": list(a_top),
                        "clean_recall_at5": c_rec,
                        "no_rule_recall_at5": a_rec,
                        "delta_recall_at5": delta,
                        "clean_all_gold_at5": all_gold(c),
                        "no_rule_all_gold_at5": all_gold(a),
                    }
                )

        clean_metrics = clean.get("metrics", {}) or {}
        abl_metrics = abl.get("metrics", {}) or {}
        clean_r5 = float(clean_metrics.get("r5") or 0.0)
        abl_r5 = float(abl_metrics.get("r5") or 0.0)
        clean_all = float(clean_metrics.get("all_gold_at5") or 0.0)
        abl_all = float(abl_metrics.get("all_gold_at5") or 0.0)
        delta_r5 = abl_r5 - clean_r5
        delta_all = abl_all - clean_all
        net_losses = losses - gains
        if abs(delta_r5) <= 0.002 and abs(delta_all) <= 0.003 and net_losses <= 2:
            decision = "retrieval_tie_delete_candidate"
        elif losses > 2 or delta_r5 < -0.002 or delta_all < -0.003:
            decision = "retrieval_loss_keep_candidate"
        else:
            decision = "hold"

        dataset_summary = {
            "dataset": dataset,
            "rows": len(clean_rows),
            "clean_r5": clean_r5,
            "ablation_r5": abl_r5,
            "delta_r5": delta_r5,
            "clean_all_gold_at5": clean_all,
            "ablation_all_gold_at5": abl_all,
            "delta_all_gold_at5": delta_all,
            "top5_changed": changed,
            "gold_gains": gains,
            "gold_losses": losses,
            "net_gold_losses": net_losses,
            "decision_signal": decision,
            "changed_examples": changed_examples,
        }
        summary["datasets"].append(dataset_summary)
        md.append(
            f"| {dataset} | {len(clean_rows)} | {clean_r5:.4f} | {abl_r5:.4f} | {delta_r5:+.4f} | "
            f"{clean_all:.4f} | {abl_all:.4f} | {delta_all:+.4f} | {changed} | {gains} | {losses} | {net_losses} | {decision} |"
        )

    (OUT_ROOT / "no_multi_anchor_precision_full1000_gate.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "no_multi_anchor_precision_full1000_gate.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
