# Non-Oracle Bridge-Greedy Pilot

Last updated: 2026-03-30

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

## 2Wiki Pilot on `v2 + general_relation_graph`

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

## Canonical-Backbone Pilots

These are the more important pilots for the paper, because the canonical retrieval backbone is:
- `causal_engine_version = v2`
- `causal_v2_base_retrieval_mode = legacy_fact_graph`
- `causal_enabled = false`

### 2Wiki pilot on canonical backbone

Report path:
- `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_legacy.json`

Results:
- baseline `EM/F1 = 0.5000 / 0.5523`
- selector `EM/F1 = 0.5500 / 0.5774`
- delta `EM/F1 = +0.0500 / +0.0251`
- baseline `Recall@5 = 0.8375`, selector `Recall@5 = 0.8250`
- baseline `Recall@20 = 0.8875`, selector `Recall@20 = 0.8875`

Per-bucket:
- `2-doc`: `EM +0.1333`, `F1 +0.1000`
- `4-doc`: `EM -0.2000`, `F1 -0.1995`

Interpretation:
- The selector still helps some easy `2-doc` cases.
- It does not yet solve the hard-case story on the canonical backbone.
- In its current form, it behaves more like a diversity selector than a true bridge completer.

### MuSiQue pilot on canonical backbone

Report path:
- `outputs_step0_general_musique/eval_reports/setwise_bridge_greedy_pilot_20_legacy.json`

Results:
- baseline `EM/F1 = 0.3000 / 0.3250`
- selector `EM/F1 = 0.1500 / 0.1559`
- delta `EM/F1 = -0.1500 / -0.1691`
- baseline `Recall@5 = 0.5417`, selector `Recall@5 = 0.4333`
- baseline `Recall@20 = 0.6917`, selector `Recall@20 = 0.6917`

Per-bucket:
- `2-doc`: `EM -0.2727`, `F1 -0.3182`
- `3-doc`: flat
- `4-doc`: `EM` flat at `0.0`, `F1 +0.0294`

Interpretation:
- The current heuristic does not generalize to `MuSiQue`.
- It can waste top-5 budget on semantically nearby but non-completing evidence.
- This is not yet strong enough for the method section of an EMNLP main paper.

### 2Wiki search follow-up on canonical backbone

This follow-up isolates search from local scoring by keeping the bridge-aware local score fixed and comparing:
- `bridge_greedy`
- `bridge_beam`

Report paths:
- greedy: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_100_legacy.json`
- beam: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_beam_pilot_100_legacy.json`

Matched setup:
- dataset: `2wikimultihopqa`
- limit: `100`
- pool: `100`
- anchor count: `2`
- beam params: `beam_width=4`, `beam_expand_per_state=4`

Results:
- baseline `EM/F1 = 0.3800 / 0.4332`
- greedy `EM/F1 = 0.4300 / 0.4726`
- beam `EM/F1 = 0.4600 / 0.4924`
- greedy delta `EM/F1 = +0.0500 / +0.0394`
- beam delta `EM/F1 = +0.0800 / +0.0592`
- greedy `Recall@5 = 0.8150`, beam `Recall@5 = 0.8075`
- greedy `Recall@20 = 0.8825`, beam `Recall@20 = 0.8825`

Per-bucket:
- `2-doc`: greedy `EM +0.0909`, beam `EM +0.0909`
- `4-doc`: greedy `EM -0.0870`, beam `EM +0.0435`

Interpretation:
- Search is now part of the bottleneck, not just local scoring.
- `bridge_beam` beats `bridge_greedy` on QA despite slightly lower `Recall@5`, which argues for better evidence composition rather than only shallower support coverage.
- The strongest gain is on `4-doc` queries: beam flips the hard bucket from negative to positive.
- On `2Wiki`, this is strong enough to use `bridge_beam` as the current practical non-oracle selector.

## Failed Micro-Tweak

Attempt:
- union `fact seeds` with `lexical seeds`
- add light title-level dedup during selection

Result:
- report path: `outputs_step0_general_2wikimultihopqa/eval_reports/setwise_bridge_greedy_pilot_20_legacy_seedunion_dedup.json`
- `2Wiki` canonical pilot moved from `EM +0.0500` to `EM -0.0500`
- `4-doc` bucket became worse: `EM -0.4000`

Decision:
- This tweak was reverted from code.
- Keep the stable checkpoint implementation only; treat the tweak as a failed branch.

## Current Status

What is true now:
- The oracle diagnosis story is strong.
- A practical non-oracle baseline exists and runs end-to-end.
- `bridge_greedy` alone is not yet paper-ready on the canonical backbone.
- `bridge_beam` is the more promising line because it improves the canonical `2Wiki` setting and rescues the hard bucket.

Implication:
- The paper can already support a strong diagnosis-first story.
- For EMNLP main, the practical method should now be centered on `bridge_beam`, with `bridge_greedy` retained as the search-control ablation.
- The next decision depends on whether the same trend appears on `MuSiQue`.
