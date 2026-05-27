from importlib import util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_hipporag_pool.py"
_SPEC = util.spec_from_file_location("export_hipporag_pool", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_doc_id_index = _MODULE.build_doc_id_index
normalize_doc_text = _MODULE.normalize_doc_text
pool_doc_ids_for_docs = _MODULE.pool_doc_ids_for_docs


def test_pool_doc_ids_are_corpus_indices_not_pool_positions():
    docs = [
        "Doc A\nalpha",
        "Doc B\nbeta",
        "Doc C\ngamma",
    ]
    index = build_doc_id_index(docs)

    assert pool_doc_ids_for_docs(["Doc C\ngamma", "Doc A\nalpha"], index) == [2, 0]


def test_pool_doc_id_resolution_normalizes_whitespace():
    docs = ["Doc A\nalpha beta"]
    index = build_doc_id_index(docs)

    assert normalize_doc_text("Doc A\nalpha   beta") == "Doc A alpha beta"
    assert pool_doc_ids_for_docs(["Doc A\nalpha   beta"], index) == [0]


def test_pool_doc_id_resolution_fails_loudly_for_unmapped_docs():
    index = build_doc_id_index(["Doc A\nalpha"])

    try:
        pool_doc_ids_for_docs(["Doc B\nbeta"], index)
    except ValueError as exc:
        assert "Could not resolve" in str(exc)
        assert "Doc B" in str(exc)
    else:
        raise AssertionError("expected unresolved document to fail")
