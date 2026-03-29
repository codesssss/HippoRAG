# Non-Oracle Bridge-Greedy Pilot

Last updated: 2026-03-29

## Implementation

Code files:
- `scripts/eval_causal_qwen3.py`
- `src/hipporag/HippoRAG.py`
- `tests/test_setwise_selector.py`

What landed:
- Added a minimal non-oracle setwise selector: `--setwise_selector bridge_greedy`
- Selector widens the pool to `pool_k`, keeps top anchors, then greedily fills the reader top-`k` using:
  - normalized base retrieval score
  - structure bridge score from entity graph reachability
  - entity novelty
- Added lexical seed fallback so the selector still works when V2 fact rerank seeds are empty
- Added per-bucket QA reporting for selector vs baseline

Critical runtime fix:
- In `HippoRAG._prepare_retrieval_objects_v2()`, structure retrieval objects were not being prepared under the current `v2 + general_relation_graph` path.
- Fix: load existing OpenIE and call `_prepare_structure_retrieval_objects(...)`
- Without this, the selector had no usable structure signal.

Validation:
- `py_compile` passed for the edited files
- direct test-function invocation passed for `tests/test_setwise_selector.py`

## 2Wiki Pilot

Pilot setup:
- dataset: `2wikimultihopqa`
- limit: `20`
- selector: `bridge_greedy`
- pool: `100`
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_v2.json`

Results:
- baseline `EM = 0.5000`, `F1 = 0.5000`
- selector `EM = 0.6000`, `F1 = 0.6375`
- delta `EM = +0.1000`, `F1 = +0.1375`
- baseline `Recall@5 = 0.7125`, selector `Recall@5 = 0.8125`
- baseline `Recall@20 = 0.8125`, selector `Recall@20 = 0.9500`

Per-bucket:
- `2-doc`: baseline `EM = 0.4667` -> selector `EM = 0.5333` (`+0.0667`)
- `4-doc`: baseline `EM = 0.6000` -> selector `EM = 0.8000` (`+0.2000`)

Interpretation:
- This is the first non-oracle positive signal that the diagnosis can translate into a practical method.
- Gains are larger on `4-doc` queries, which matches the paper story.
- The method is still simple and heuristic, but it already improves both QA and support recall.

## Running Status

Current run:
- dataset: `MuSiQue`
- setup: `limit=20`, `pool_k=100`, selector `bridge_greedy`
- output target: `/tmp/setwise_selector_pilot_musique_20.json`

Important note:
- The first `MuSiQue` non-oracle run is building the `causal_v2` cache from scratch because `outputs_step0_general_musique/.../causal_v2/manifest.json` was not present.
- This is a runtime cost issue, not a logic failure.
- Current observed speed is hours-scale, so treat `MuSiQue` pilot as an ongoing background job rather than a quick smoke test.
