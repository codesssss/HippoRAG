from __future__ import annotations

from pathlib import Path

import pytest

from evidenceflow.contract import (
    MAIN_UPSTREAM_RETRIEVER_NAME,
    METHOD_CONTRACT,
    is_main_evidenceflow_pool,
    pool_upstream_retriever,
    residual_budget,
    validate_budgets,
)
from evidenceflow.readout import (
    enrich_payload,
    pcec_readout_trace,
    summarize_pcec_traces,
)


def make_trace(*, swapped: bool = True) -> dict:
    selector_trace = {
        "selector": "daec_noisyor_safe",
        "baseline_objective": 0.4,
        "selected_binding_id": "b0",
        "safe_projection_trace": {
            "baseline_objective": 0.4,
            "safe_decision": "minimal_edit_applied" if swapped else "fallback_no_eligible_swap",
            "safe_swap_steps": [
                {
                    "step": 1,
                    "mode": "daec_noisyor_safe_swap",
                    "out_position": 4,
                    "out_title": "Noise",
                    "in_position": 12,
                    "in_title": "Residual Evidence",
                    "objective_gain": 0.15,
                    "objective": 0.55,
                }
            ]
            if swapped
            else [],
        },
    }
    return {
        "question": "Who connected the two facts?",
        "gold_titles": ["A", "Residual Evidence"],
        "baseline_top_titles": ["A", "B", "C", "D", "Noise"],
        "selector_top_titles": ["A", "B", "C", "D", "Residual Evidence"]
        if swapped
        else ["A", "B", "C", "D", "Noise"],
        "selector_trace": selector_trace,
    }


def test_budget_validation_and_residual_budget() -> None:
    validate_budgets(reader_budget_k=5, prefix_budget_m=4)
    assert residual_budget(5, 4) == 1
    assert residual_budget(5, 5) == 0
    with pytest.raises(ValueError):
        validate_budgets(reader_budget_k=0, prefix_budget_m=0)
    with pytest.raises(ValueError):
        validate_budgets(reader_budget_k=5, prefix_budget_m=6)
    with pytest.raises(ValueError):
        validate_budgets(reader_budget_k=5, prefix_budget_m=-1)


def test_contract_exposes_pcec_method_boundary() -> None:
    assert METHOD_CONTRACT["paper_facing_method_name"] == "EvidenceFlow"
    assert METHOD_CONTRACT["main_protocol"] == "etv4_fact_witnessed_sto_pool_to_pcec_native_readout"
    assert METHOD_CONTRACT["main_entrypoint"] == "evidenceflow/run_native_pool.py"
    assert METHOD_CONTRACT["main_upstream_retriever"] == MAIN_UPSTREAM_RETRIEVER_NAME
    assert METHOD_CONTRACT["main_pool_provenance_key"] == "retrieval.input_method"
    assert METHOD_CONTRACT["uses_etv4_fact_witnessed_sto_pool_for_main_table"] is True
    assert METHOD_CONTRACT["uses_frozen_etv3_for_main_table"] is False
    assert METHOD_CONTRACT["legacy_fresh_expander_is_main_table_retriever"] is False
    assert (
        METHOD_CONTRACT["legacy_fresh_expander_package"]
        == "evidenceflow.frozen_etv3_variable_flow"
    )
    assert METHOD_CONTRACT["readout_component"] == "Preservation-Constrained Evidence Composition"
    assert METHOD_CONTRACT["readout_component_abbreviation"] == "PCEC"
    assert METHOD_CONTRACT["default_reader_budget_k"] == 5
    assert METHOD_CONTRACT["default_prefix_budget_m"] == 4
    assert METHOD_CONTRACT["default_residual_budget"] == 1
    assert METHOD_CONTRACT["uses_hard_preservation_constraint"] is True
    assert METHOD_CONTRACT["uses_lambda_regularization"] is False
    assert METHOD_CONTRACT["uses_weighted_score_fusion"] is False
    assert METHOD_CONTRACT["uses_dataset_routing"] is False
    assert METHOD_CONTRACT["uses_cross_signal_agreement_retention"] is False


def test_pool_protocol_helpers_identify_main_etv4_pool() -> None:
    main_pool = {"retrieval": {"input_method": MAIN_UPSTREAM_RETRIEVER_NAME}}
    legacy_pool = {"retrieval": {"input_method": "evidence_transition_graphragv3_variable_flow"}}
    missing_pool = {"rows": []}

    assert pool_upstream_retriever(main_pool) == MAIN_UPSTREAM_RETRIEVER_NAME
    assert is_main_evidenceflow_pool(main_pool) is True
    assert is_main_evidenceflow_pool(legacy_pool) is False
    assert is_main_evidenceflow_pool(missing_pool) is False


def test_legacy_package_imports_delegate_to_evidenceflow() -> None:
    from evidence_transition_graphragv4_composition.contract import (
        METHOD_CONTRACT as legacy_contract,
    )
    from evidence_transition_graphragv4_composition.native_readout import (
        compose_pcec_readout as legacy_compose_pcec_readout,
    )
    from evidence_transition_graphragv4_composition.readout import (
        pcec_readout_trace as legacy_pcec_readout_trace,
    )
    from evidenceflow.native_readout import compose_pcec_readout

    assert legacy_contract is METHOD_CONTRACT
    assert legacy_compose_pcec_readout is compose_pcec_readout
    assert legacy_pcec_readout_trace is pcec_readout_trace


def test_legacy_runner_paths_remain_available() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "evidence_transition_graphragv4_composition" / "run_fresh_e2e.py").exists()
    assert (
        root
        / "evidence_transition_graphragv4_composition"
        / "frozen_etv3_variable_flow"
        / "run_reader_qa.py"
    ).exists()


def test_frozen_etv3_snapshot_does_not_import_live_etv3_package() -> None:
    root = Path(__file__).resolve().parents[1] / "evidenceflow" / "frozen_etv3_variable_flow"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if (
            "from evidence_transition_graphragv3_variable_flow" in text
            or "import evidence_transition_graphragv3_variable_flow" in text
        ):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_pcec_readout_trace_records_prefix_residual_admission() -> None:
    trace = pcec_readout_trace(
        make_trace(swapped=True),
        reader_budget_k=5,
        prefix_budget_m=4,
        pool_k=100,
    )

    assert trace["reader_budget_k"] == 5
    assert trace["prefix_budget_m"] == 4
    assert trace["residual_budget"] == 1
    assert trace["retained_prefix_titles"] == ["A", "B", "C", "D"]
    assert trace["residual_incumbent_title"] == "Noise"
    assert trace["best_candidate_title"] == "Residual Evidence"
    assert trace["best_candidate_pool_rank"] == 13
    assert trace["best_candidate_gain"] == 0.15
    assert trace["decision"] == "admit"
    assert trace["final_top5_titles"] == ["A", "B", "C", "D", "Residual Evidence"]
    assert trace["ordering_policy"] == "preserve_et_prefix_order_and_fill_residual_slot"


def test_pcec_readout_trace_records_noop_decision() -> None:
    trace = pcec_readout_trace(
        make_trace(swapped=False),
        reader_budget_k=5,
        prefix_budget_m=4,
        pool_k=100,
    )

    assert trace["decision"] == "keep_baseline"
    assert trace["best_candidate_title"] is None
    assert trace["final_top5_titles"] == ["A", "B", "C", "D", "Noise"]


def test_enrich_payload_adds_contract_summary_and_per_query_trace() -> None:
    payload = {
        "dataset": "toy",
        "setwise_selector_query_traces": [
            make_trace(swapped=True),
            make_trace(swapped=False),
        ],
    }

    enriched = enrich_payload(payload, reader_budget_k=5, prefix_budget_m=4, pool_k=100)

    assert enriched["method"] == "evidenceflow"
    assert enriched["pcec_config"]["residual_budget"] == 1
    assert enriched["pcec_summary"]["count"] == 2
    assert enriched["pcec_summary"]["changed_count"] == 1
    assert enriched["pcec_summary"]["total_swaps"] == 1
    assert enriched["pcec_summary"]["pcec_title_all_gold_top5"] == 0.5
    assert "pcec_readout" in enriched["setwise_selector_query_traces"][0]


def test_summarize_pcec_traces_counts_gold_out() -> None:
    trace = make_trace(swapped=True)
    trace["gold_titles"] = ["Noise"]

    summary = summarize_pcec_traces([trace], reader_budget_k=5)

    assert summary["queries_swapped_out_gold"] == 1
    assert summary["queries_swapped_in_gold"] == 0
