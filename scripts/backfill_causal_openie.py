import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.hipporag.information_extraction import OpenIE
from src.hipporag.llm import _get_llm_class
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import CausalRawOutput, string_to_bool


logger = logging.getLogger(__name__)


def build_config(args) -> BaseConfig:
    dataset_name = args.dataset
    save_dir = args.save_dir
    if save_dir == "outputs":
        save_dir = os.path.join(save_dir, dataset_name)
    else:
        save_dir = f"{save_dir}_{dataset_name}"

    return BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_base_url=args.embedding_base_url,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        max_retry_attempts=args.max_retry_attempts,
        openie_mode="online",
        causal_enabled=True,
        causal_query_only=True,
        causal_confidence_threshold=args.causal_confidence_threshold,
    )


def resolve_openie_results_path(config: BaseConfig) -> Path:
    return Path(config.save_dir) / f"openie_results_ner_{config.llm_name.replace('/', '_')}.json"


def load_openie_results(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def save_openie_results(path: Path, payload: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f)
    tmp_path.replace(path)


def should_process_doc(doc: Dict, overwrite_existing: bool) -> bool:
    metadata = doc.get("causal_extraction_metadata", {})
    if overwrite_existing:
        return True
    if metadata.get("backfilled") is True:
        return False
    return True


def prepare_docs_for_backfill(docs: List[Dict], overwrite_existing: bool, limit: int) -> List[int]:
    indices = [idx for idx, doc in enumerate(docs) if should_process_doc(doc, overwrite_existing=overwrite_existing)]
    if limit and limit > 0:
        return indices[:limit]
    return indices


def run_single_backfill(openie: OpenIE, doc: Dict) -> Tuple[str, CausalRawOutput]:
    chunk_id = str(doc["idx"])
    passage = str(doc["passage"])
    triples = doc.get("extracted_triples", [])
    causal_output = openie.causal_relation_extraction(
        chunk_key=chunk_id,
        passage=passage,
        triples=triples,
    )
    return chunk_id, causal_output


def apply_causal_output(doc: Dict, causal_output: CausalRawOutput, model_name: str) -> None:
    doc["extracted_causal_relations"] = [
        relation.to_dict() for relation in causal_output.causal_relations
    ]
    metadata = deepcopy(causal_output.metadata) if causal_output.metadata else {}
    metadata["backfilled"] = True
    metadata["llm_name"] = model_name
    metadata["num_relations"] = len(causal_output.causal_relations)
    doc["causal_extraction_metadata"] = metadata


def maybe_backup_file(path: Path, skip_backup: bool) -> None:
    if skip_backup:
        return
    backup_path = path.with_suffix(path.suffix + ".bak")
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)


def main():
    parser = argparse.ArgumentParser(description="Backfill causal relations into an existing OpenIE results file.")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8039/v1")
    parser.add_argument("--llm_name", type=str, default="qwen3-8b")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding_base_url", type=str, default="")
    parser.add_argument("--max_retry_attempts", type=int, default=5)
    parser.add_argument("--causal_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--overwrite_existing", type=str, default="false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--skip_backup", type=str, default="false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    overwrite_existing = string_to_bool(args.overwrite_existing)
    skip_backup = string_to_bool(args.skip_backup)

    config = build_config(args)
    openie_results_path = resolve_openie_results_path(config)
    if not openie_results_path.exists():
        raise FileNotFoundError(f"OpenIE results file not found: {openie_results_path}")

    maybe_backup_file(openie_results_path, skip_backup=skip_backup)
    openie_payload = load_openie_results(openie_results_path)
    docs = openie_payload.get("docs", [])

    target_indices = prepare_docs_for_backfill(docs, overwrite_existing=overwrite_existing, limit=args.limit)
    logger.info("Loaded %s docs from %s", len(docs), openie_results_path)
    logger.info("Processing %s docs for causal backfill", len(target_indices))

    llm_model = _get_llm_class(config)
    openie = OpenIE(llm_model=llm_model)

    processed = 0
    skipped_no_triples = 0
    failed = 0
    total_relations = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_doc_index = {}
        for doc_index in target_indices:
            doc = docs[doc_index]
            triples = doc.get("extracted_triples", [])
            if not triples:
                doc["extracted_causal_relations"] = []
                doc["causal_extraction_metadata"] = {
                    "backfilled": True,
                    "llm_name": args.llm_name,
                    "num_relations": 0,
                    "skipped_reason": "no_triples",
                }
                skipped_no_triples += 1
                processed += 1
                continue
            future = executor.submit(run_single_backfill, openie, doc)
            future_to_doc_index[future] = doc_index

        pbar = tqdm(as_completed(future_to_doc_index), total=len(future_to_doc_index), desc="Backfilling causal OpenIE")
        for future in pbar:
            doc_index = future_to_doc_index[future]
            doc = docs[doc_index]
            try:
                _, causal_output = future.result()
                apply_causal_output(doc, causal_output, model_name=args.llm_name)
                total_relations += len(causal_output.causal_relations)
                if causal_output.metadata.get("error"):
                    failed += 1
            except Exception as exc:
                failed += 1
                doc["extracted_causal_relations"] = []
                doc["causal_extraction_metadata"] = {
                    "backfilled": True,
                    "llm_name": args.llm_name,
                    "num_relations": 0,
                    "error": str(exc),
                }

            processed += 1
            if processed % args.checkpoint_every == 0:
                save_openie_results(openie_results_path, openie_payload)
                logger.info(
                    "Checkpoint saved after %s docs. total_relations=%s failed=%s skipped_no_triples=%s",
                    processed,
                    total_relations,
                    failed,
                    skipped_no_triples,
                )

            pbar.set_postfix({
                "processed": processed,
                "relations": total_relations,
                "failed": failed,
                "skip_no_triples": skipped_no_triples,
            })

    save_openie_results(openie_results_path, openie_payload)
    logger.info(
        "Causal backfill completed. processed=%s total_relations=%s failed=%s skipped_no_triples=%s output=%s",
        processed,
        total_relations,
        failed,
        skipped_no_triples,
        openie_results_path,
    )
    print(json.dumps({
        "output_path": str(openie_results_path),
        "processed_docs": processed,
        "total_relations": total_relations,
        "failed_docs": failed,
        "skipped_no_triples": skipped_no_triples,
    }, indent=2))


if __name__ == "__main__":
    main()
