import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_alias_closure_cases import (
    cluster_descriptor_aliases,
    diagnose_case,
    extract_bare_title_alias_candidates,
)


def test_extract_bare_title_alias_candidates_detects_comma_suffix_alias():
    candidates = extract_bare_title_alias_candidates(
        "Seria, Belait",
        {"seria", "brunei"},
    )

    assert candidates == [{
        "full_form": "seria belait",
        "alias_form": "seria",
        "source": "comma_suffix_strip",
    }]


def test_cluster_descriptor_aliases_groups_president_variants():
    clusters = cluster_descriptor_aliases({
        "us president woodrow wilson",
        "president woodrow wilson",
        "jessie woodrow wilson sayre",
    })

    assert clusters == [{
        "core_form": "woodrow wilson",
        "forms": ["president woodrow wilson", "us president woodrow wilson"],
    }]


def test_diagnose_case_distinguishes_missing_alias_from_descriptor_alias():
    q65_case = {
        "query_id": 65,
        "question": "Q65",
        "best_offrank_title": "Seria, Belait",
        "gold_titles": ["Seria, Belait", "Adult contemporary music"],
    }
    q85_case = {
        "query_id": 85,
        "question": "Q85",
        "best_offrank_title": "Jessie Woodrow Wilson Sayre",
        "gold_titles": ["Jessie Woodrow Wilson Sayre", "Prohibition in the United States"],
    }
    title_to_docs = {
        "Seria, Belait": [{
            "passage": "Seria, Belait\nText",
            "extracted_triples": [["Seria", "is located in", "Brunei"]],
        }],
        "Adult contemporary music": [{
            "passage": "Adult contemporary music\nText",
            "extracted_triples": [["Walter Sabo", "worked at", "NBC"]],
        }],
        "Jessie Woodrow Wilson Sayre": [{
            "passage": "Jessie Woodrow Wilson Sayre\nText",
            "extracted_triples": [["Jessie Woodrow Wilson Sayre", "is daughter of", "US President Woodrow Wilson"]],
        }],
        "Prohibition in the United States": [{
            "passage": "Prohibition in the United States\nText",
            "extracted_triples": [["Volstead Act", "passed over", "President Woodrow Wilson"]],
        }],
    }

    q65 = diagnose_case(q65_case, title_to_docs)
    q85 = diagnose_case(q85_case, title_to_docs)

    assert q65["primary_diagnosis"] == "title_alias_missing"
    assert q65["alias_only_plausible"] is True
    assert q85["primary_diagnosis"] == "alias_plus_relation_family"
    assert q85["alias_only_plausible"] is False


def test_diagnose_case_marks_best_offrank_alias_as_already_available():
    q69_case = {
        "query_id": 69,
        "question": "Q69",
        "best_offrank_title": "Tucson, Arizona",
        "gold_titles": ["Tucson, Arizona", "Drexel Heights, Arizona"],
    }
    title_to_docs = {
        "Tucson, Arizona": [{
            "passage": "Tucson, Arizona\nText",
            "extracted_triples": [
                ["Tucson", "is located in", "Pima County"],
                ["Tucson, Arizona", "is located in", "Pima County"],
            ],
        }],
        "Drexel Heights, Arizona": [{
            "passage": "Drexel Heights, Arizona\nText",
            "extracted_triples": [["Drexel Heights", "is located in", "Pima County"]],
        }],
    }

    q69 = diagnose_case(q69_case, title_to_docs)

    assert q69["best_offrank_alias_already_available"] is True
    assert q69["primary_diagnosis"] == "alias_present_but_additional_reachability_needed"
    assert q69["alias_only_plausible"] is False
