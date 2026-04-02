import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT_PATH = SCRIPT_DIR / "eval_causal_qwen3.py"
DEFAULT_CACHE_TEMPLATE = (
    "research_memory/emnlp_expand_then_compose/models/"
    "{dataset}_requirement_cache_pool100_ann50_limit{limit}.json"
)


@dataclass(frozen=True)
class RequirementReserveAblationJob:
    dataset: str
    label: str
    anchor_count: int
    reserve_top_m: int
    reserve_policy: str
    cache_path: str
    output_json: str


def parse_csv_ints(value: str) -> List[int]:
    parsed_values: List[int] = []
    for raw_part in str(value or "").split(","):
        stripped = raw_part.strip()
        if not stripped:
            continue
        parsed_values.append(int(stripped))
    return parsed_values


def resolve_dataset_save_dir(save_dir: str, dataset: str) -> Path:
    if str(save_dir) == "outputs":
        return Path(save_dir) / dataset
    return Path(f"{save_dir}_{dataset}")


def build_requirement_reserve_ablation_jobs(datasets: Sequence[str],
                                            reserve_values: Sequence[int],
                                            limit: int,
                                            save_dir: str,
                                            cache_template: str = DEFAULT_CACHE_TEMPLATE,
                                            default_anchor_count: int = 2,
                                            include_adaptive: bool = True,
                                            adaptive_policy: str = "adaptive_requirement_count",
                                            adaptive_base_reserve_top_m: int = 3,
                                            output_prefix: str = "requirement_beam_oracle") -> List[RequirementReserveAblationJob]:
    jobs: List[RequirementReserveAblationJob] = []
    normalized_reserves = [max(int(value), 0) for value in reserve_values]
    for dataset in datasets:
        dataset_save_dir = resolve_dataset_save_dir(save_dir=save_dir, dataset=dataset)
        report_dir = dataset_save_dir / "eval_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_template.format(dataset=dataset, limit=int(limit))

        for effective_reserved_count in normalized_reserves:
            anchor_count = min(max(int(default_anchor_count), 0), effective_reserved_count)
            reserve_top_m = int(effective_reserved_count)
            label = f"reserve{effective_reserved_count}"
            output_json = report_dir / (
                f"{output_prefix}_smoke{int(limit)}_"
                f"v2_legacyfact_pool100_ann50_{label}.json"
            )
            jobs.append(RequirementReserveAblationJob(
                dataset=str(dataset),
                label=label,
                anchor_count=int(anchor_count),
                reserve_top_m=int(reserve_top_m),
                reserve_policy="fixed",
                cache_path=str(cache_path),
                output_json=str(output_json),
            ))

        if include_adaptive:
            output_json = report_dir / (
                f"{output_prefix}_smoke{int(limit)}_"
                f"v2_legacyfact_pool100_ann50_adaptive_reqcount.json"
            )
            jobs.append(RequirementReserveAblationJob(
                dataset=str(dataset),
                label="adaptive_reqcount",
                anchor_count=max(int(default_anchor_count), 0),
                reserve_top_m=max(int(adaptive_base_reserve_top_m), 0),
                reserve_policy=str(adaptive_policy),
                cache_path=str(cache_path),
                output_json=str(output_json),
            ))
    return jobs


def build_eval_command(job: RequirementReserveAblationJob, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT_PATH),
        "--dataset", job.dataset,
        "--limit", str(int(args.limit)),
        "--save_dir", str(args.save_dir),
        "--llm_base_url", str(args.llm_base_url),
        "--llm_name", str(args.llm_name),
        "--embedding_name", str(args.embedding_name),
        "--embedding_base_url", str(args.embedding_base_url),
        "--max_retry_attempts", str(int(args.max_retry_attempts)),
        "--causal_enabled", str(args.causal_enabled),
        "--causal_engine_version", str(args.causal_engine_version),
        "--causal_v2_base_retrieval_mode", str(args.causal_v2_base_retrieval_mode),
        "--setwise_selector", "requirement_beam",
        "--setwise_requirement_cache_path", str(job.cache_path),
        "--setwise_requirement_mode", str(args.setwise_requirement_mode),
        "--setwise_requirement_reserve_policy", str(job.reserve_policy),
        "--setwise_pool_k", str(int(args.setwise_pool_k)),
        "--setwise_anchor_count", str(int(job.anchor_count)),
        "--setwise_reserve_top_m", str(int(job.reserve_top_m)),
        "--setwise_non_anchor_title_dedup", str(args.setwise_non_anchor_title_dedup),
        "--setwise_beam_width", str(int(args.setwise_beam_width)),
        "--setwise_beam_expand_per_state", str(int(args.setwise_beam_expand_per_state)),
        "--setwise_beam_projected_shortlist_factor", str(int(args.setwise_beam_projected_shortlist_factor)),
        "--output_json", str(job.output_json),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run requirement_beam reserve ablations over one or more datasets.")
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated dataset names, e.g. musique,2wikimultihopqa")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--save_dir", type=str, default="outputs_step0_general")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="VLLM//mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_base_url", type=str, default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--max_retry_attempts", type=int, default=12)
    parser.add_argument("--causal_enabled", type=str, default="false")
    parser.add_argument("--causal_engine_version", choices=["legacy", "v2"], default="v2")
    parser.add_argument("--causal_v2_base_retrieval_mode", choices=["dense", "legacy_fact_graph", "general_relation_graph"], default="legacy_fact_graph")
    parser.add_argument("--setwise_requirement_mode", choices=["oracle", "learned"], default="oracle")
    parser.add_argument("--setwise_pool_k", type=int, default=100)
    parser.add_argument("--setwise_non_anchor_title_dedup", type=str, default="true")
    parser.add_argument("--setwise_beam_width", type=int, default=4)
    parser.add_argument("--setwise_beam_expand_per_state", type=int, default=4)
    parser.add_argument("--setwise_beam_projected_shortlist_factor", type=int, default=3)
    parser.add_argument("--reserve_values", type=str, default="3,1,0",
                        help="Comma-separated effective reserved-prefix sizes. Each value n maps to anchor_count=min(default_anchor_count,n), reserve_top_m=n.")
    parser.add_argument("--default_anchor_count", type=int, default=2)
    parser.add_argument("--include_adaptive", type=str, default="true")
    parser.add_argument("--adaptive_policy", choices=["adaptive_requirement_count"], default="adaptive_requirement_count")
    parser.add_argument("--adaptive_base_reserve_top_m", type=int, default=3)
    parser.add_argument("--cache_template", type=str, default=DEFAULT_CACHE_TEMPLATE)
    parser.add_argument("--output_prefix", type=str, default="requirement_beam_oracle")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    datasets = [part.strip() for part in str(args.datasets).split(",") if part.strip()]
    reserve_values = parse_csv_ints(args.reserve_values)
    include_adaptive = str(args.include_adaptive).strip().lower() in {"1", "true", "yes", "y"}

    jobs = build_requirement_reserve_ablation_jobs(
        datasets=datasets,
        reserve_values=reserve_values,
        limit=int(args.limit),
        save_dir=str(args.save_dir),
        cache_template=str(args.cache_template),
        default_anchor_count=int(args.default_anchor_count),
        include_adaptive=include_adaptive,
        adaptive_policy=str(args.adaptive_policy),
        adaptive_base_reserve_top_m=int(args.adaptive_base_reserve_top_m),
        output_prefix=str(args.output_prefix),
    )

    print(json.dumps({
        "job_count": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    for job in jobs:
        command = build_eval_command(job=job, args=args)
        print(f"\n[reserve-ablation] running {job.dataset}:{job.label}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
