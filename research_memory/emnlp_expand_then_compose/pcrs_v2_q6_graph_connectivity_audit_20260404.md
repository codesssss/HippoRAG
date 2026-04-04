# PCRS-RAG V2 Q6 Graph Connectivity Audit

Date: 2026-04-04

## Scope

This note follows the q6 source-admission probes.

Question under audit:

`Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`

Goal:

> determine whether q6 fails because the graph already contains a usable path that source scoring fails to exploit, or because the current structure graph never encodes the needed bridge chain at all.

## Inputs

Artifacts used:

- corpus:
  - `reproduce/dataset/musique_corpus.json`
- dataset sample:
  - `reproduce/dataset/musique.json`
- current workspace:
  - `outputs_step0_general_musique/qwen3-8b-train_VLLM__mnt_nvme_Qwen3-Embedding-8B/graph.pickle`
  - `outputs_step0_general_musique/openie_results_ner_qwen3-8b-train.json`
- prior selector notes:
  - `research_memory/emnlp_expand_then_compose/pcrs_v2_pool_coverage_census_q6_20260403.md`
  - `research_memory/emnlp_expand_then_compose/pcrs_v2_q6_source_admission_probes_20260404.md`

All checks below use the same V2 substrate:

- `openie_mode=online`
- `causal_engine_version=v2`
- `causal_v2_base_retrieval_mode=legacy_fact_graph`
- `causal_enabled=false`

## Exact Gold Docs for Q6

The q6 sample in `musique.json` uses these supporting titles:

- `Riverside Plaza`
- `Mississippi River`
- `Minneapolis`
- `Southeast Library`

These exact supporting paragraphs match the first title instances in the current corpus workspace.

## Gold Doc Mapping in Current Workspace

After loading the current workspace and calling `prepare_retrieval_objects()`, the exact q6 docs map to:

- `Southeast Library`
  - `doc_idx=133`
- `Colorado River (Texas)`
  - `doc_idx=128`
- `Gulf of Mexico`
  - `doc_idx=122`
- `Riverside Plaza`
  - `doc_idx=119`
- `Minneapolis`
  - `doc_idx=124`
- `Mississippi River`
  - `doc_idx=120`

Note:

- there is no standalone `Ralph Rapson` title in this corpus slice
- only `Southeast Library` and `Riverside Plaza` explicitly mention `Ralph Rapson`

## Document-Level Structure Objects

### Extracted entities exist

The current `doc_idx_to_structure_entities` does include the expected q6 bridge entities:

- `Southeast Library`
  - includes `southeast library`, `ralph rapson`
- `Riverside Plaza`
  - includes `riverside plaza`, `ralph rapson`, `minneapolis  minnesota`
- `Minneapolis`
  - includes `minneapolis`, `mississippi river`
- `Mississippi River`
  - includes `mississippi river`, `gulf of mexico`, `minnesota`

So the problem is not missing NER mentions.

### But `doc_idx_to_structure_edges` is empty

For the exact q6 gold docs inspected above:

- `Southeast Library`: `edge_count=0`
- `Riverside Plaza`: `edge_count=0`
- `Minneapolis`: `edge_count=0`
- `Mississippi River`: `edge_count=0`
- `Colorado River (Texas)`: `edge_count=0`
- `Gulf of Mexico`: `edge_count=0`

This is the first critical result:

> the current structure retrieval layer sees the entities, but it creates no directed structure edges for the q6 chain documents.

## Reachability Audit

Using the q6 pre-bridge covered set implied by the selected early docs:

- `Southeast Library`
- `Colorado River (Texas)`
- `Gulf of Mexico`

the union of covered entities includes:

- `southeast library`
- `ralph rapson`
- `gulf of mexico`
- `colorado river texas`
- and other local entities from those three docs

From that covered entity set, `expand_directed_entities(..., max_hops=h)` was tested for `h=1..6`.

Result:

- `riverside plaza`: unreachable for all `h<=6`
- `minneapolis`: unreachable for all `h<=6`
- `mississippi river`: unreachable for all `h<=6`
- `ralph rapson`: unreachable for all `h<=6`

This is the second critical result:

> q6 is not failing because the graph path is longer than `structure_max_hops=2`. In the current structure graph, these bridge entities are not reachable even at much larger hop limits.

## Adjacency Audit

The current `structure_graph_out` neighbor counts for the relevant q6 entities are:

- `ralph rapson`: `0`
- `southeast library`: `0`
- `riverside plaza`: `0`
- `minneapolis`: `0`
- `mississippi river`: `0`
- `gulf of mexico`: `0`

So the audit confirms:

> the needed q6 bridge entities are present as normalized mentions, but they are isolated in the current structure graph.

## Raw OpenIE Audit

The failure is not due to empty extraction. Raw OpenIE triples for the exact q6 docs already contain the right semantic relations:

### `Southeast Library`

- `Southeast Library` `designed by` `Ralph Rapson`

### `Riverside Plaza`

- `Riverside Plaza` `designed by` `Ralph Rapson`
- `Riverside Plaza` `opened in` `Minneapolis, Minnesota`

### `Minneapolis`

- `Minneapolis` `lies on` `Mississippi River`

### `Mississippi River`

- `Mississippi River` `drains into` `Gulf of Mexico`
- `Mississippi River` `rises in` `Minnesota`

So the missing path is not an upstream OpenIE recall failure.

## Predicate-Level Root Cause

The current directed structure edge builder only promotes a narrow predicate family.

A direct predicate check shows:

- `designed by` -> `None`
- `opened in` -> `None`
- `lies on` -> `None`
- `drains into` -> `None`
- `rises in` -> `None`
- `flows into` -> `None`
- `empties into` -> `None`
- `died in` -> `('state_transition', False, 0.8)`
- `born in` -> `('state_transition', False, 0.8)`

This explains the whole audit:

> the raw q6 bridge triples exist, but the current `classify_directed_predicate()` does not recognize the predicates that actually carry the q6 chain. Therefore they never become `doc_idx_to_structure_edges`, never enter `structure_graph_out`, and never become reachable bridge evidence.

## Diagnosis

This audit resolves the earlier ambiguity between:

1. graph has a path, but scorer cannot see it
2. graph never encodes the path

The evidence strongly supports case 2.

More precisely:

> q6's upstream bridge failure is primarily a graph / adjacency coverage problem, not a pure atomic-scorer or source-ranking problem.

The current selector cannot score a path that the current structure graph does not contain.

## Practical Reading

This means the next step should **not** be another source-ranking weight tweak.

The next useful probe should instead target graph coverage, for example:

- eval-only expansion of the directed predicate vocabulary to include relation families such as:
  - `designed by`
  - `opened in`
  - `lies on`
  - `drains into`
  - `flows into`
  - `empties into`
- then rerun q6 to see whether:
  - `doc_idx_to_structure_edges` becomes non-empty
  - `structure_graph_out` gains neighbors for `ralph rapson / riverside plaza / minneapolis / mississippi river`
  - the q6 bridge docs become reachable under the existing source scorer

## Bottom Line

The q6 graph connectivity audit supports the following conclusion:

> q6 is not currently blocked because the selector overlooks an existing graph path. It is blocked earlier: the current structure graph never encodes the key bridge chain, because the directed predicate normalizer ignores the exact relation phrases that appear in the q6 gold evidence.
