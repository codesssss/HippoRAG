# Minimal Demand Repair Aligned Limit100 Summary

Date: 2026-05-01

## Prototype

This is the one-day minimal prototype requested before writing a new formulation.

Algorithm boundary:
- Demand decomposition: reuse existing LLM DtC decomposition.
- `phi(r, d)`: only cosine between embedded demand subquery and embedded candidate document.
- `tau`: median of `phi` values over the query's candidate pool.
- Repair order: LLM dependency topological order, with original order as fallback.
- Eviction loss: integer count of demands for which the evicted baseline document is the unique top supporter.
- Edit budget: fixed protocol constant `2`.
- No anchor features, no binding, no relation scoring, no sentence-level scoring, no retriever prior.

Launcher:
`run_logs/launch_minimal_demand_repair_limit100_aligned_20260501.sh`

## Results

Aligned protocol:
- Prop pool100
- `qa_top_k=5`
- `qa_doc_max_chars=2048`
- Qwen3-8B no-think reader endpoints

| Dataset | Method | R@5 | R@20 | EM | F1 | KEEP | edits/q | gains/reg/same |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | Prop | 0.9350 | 0.9625 | 0.5800 | 0.6318 | - | - | - |
| 2wikimultihopqa | Prop+DAEC | 0.9525 | 0.9675 | 0.5800 | 0.6435 | - | - | - |
| 2wikimultihopqa | Prop+DAEC-L1 | 0.9375 | 0.9650 | 0.5700 | 0.6164 | - | - | - |
| 2wikimultihopqa | MinimalRepair | 0.9300 | 0.9625 | 0.5800 | 0.6300 | 0.91 | 0.09 | 0/0/100 |
| hotpotqa | Prop | 0.9300 | 0.9950 | 0.5700 | 0.6912 | - | - | - |
| hotpotqa | Prop+DAEC | 0.9500 | 0.9950 | 0.5700 | 0.6912 | - | - | - |
| hotpotqa | Prop+DAEC-L1 | 0.9500 | 0.9950 | 0.6000 | 0.7079 | - | - | - |
| hotpotqa | MinimalRepair | 0.9300 | 0.9950 | 0.5700 | 0.6912 | 0.99 | 0.01 | 0/0/100 |
| musique | Prop | 0.6775 | 0.8900 | 0.3800 | 0.4374 | - | - | - |
| musique | Prop+DAEC | 0.7175 | 0.8983 | 0.3900 | 0.4660 | - | - | - |
| musique | Prop+DAEC-L1 | 0.6600 | 0.9058 | 0.3700 | 0.4202 | - | - | - |
| musique | MinimalRepair | 0.6775 | 0.8900 | 0.3700 | 0.4340 | 0.97 | 0.03 | 0/1/99 |

## Judgment

Minimal Demand Repair with naive cosine `phi` does not establish the method.

The prototype is overwhelmingly conservative:
- 2Wiki KEEP rate: 91%, edits/query: 0.09
- HotpotQA KEEP rate: 99%, edits/query: 0.01
- MuSiQue KEEP rate: 97%, edits/query: 0.03

It essentially preserves Prop and does not approach DAEC:
- 2Wiki: MinimalRepair matches Prop EM but loses F1 versus DAEC.
- HotpotQA: MinimalRepair exactly matches Prop and misses DAEC-L1's gain.
- MuSiQue: MinimalRepair is below Prop and below DAEC.

This falls into the pre-defined outcome:

`Minimal Repair < DAEC and KEEP rate > 80%`

Interpretation: the minimal repair operator is too conservative under naive doc-level cosine `phi`. The framing is not disproven, but the one-page prototype shows that cosine-only demand support is not enough to identify repair opportunities.

## Next Boundary

Do not add anchor/binding/relation/sentence features into this prototype immediately. That would restart the feature-stack loop.

The clean conclusion from this run is:
- Minimal repair is a useful formulation candidate.
- Its viability is bottlenecked by support estimation `phi`.
- Existing DAEC's stronger results likely come from nontrivial protective/support signals, not merely from the minimal repair policy.

