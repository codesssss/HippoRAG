# Aligned Full1000 Master Comparison

Scope:

- Full1000 rows only.
- Reader top-k is 5.
- Prop-side methods consume the same exported PropRAG pool100 where noted.
- NEOCORRAG and HGRAG are native retrievers under the aligned local Qwen/NV-Embed protocol.
- `Prop+V13B*` is a full1000 run, but only the first 100 queries have cached query obligations; the remaining 900 queries fall back to the prior/top5 behavior. Treat it as `V13B-100oblg+fallback`, not a true full1000-obligation V13B.

| Dataset | Method | Regime | R@5 | R@20 | EM | F1 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| 2Wiki | PropRAG pool100 top5 | Prop-pool selector | 0.9028 | 0.9607 | 0.5790 | 0.6503 | baseline top5 |
| 2Wiki | ETv3 variable-flow | Native ET retriever/readout | 0.9015 | - | 0.5860 | 0.6601 | method-owned candidate universe; no Prop pool |
| 2Wiki | Prop+DAEC | Prop-pool selector | 0.9355 | 0.9675 | 0.6070 | 0.6810 | same Prop pool |
| 2Wiki | Prop+DAEC-L1 | Prop-pool selector | 0.9215 | 0.9692 | 0.6130 | 0.6828 | same Prop pool |
| 2Wiki | Prop+V13B* | Prop-pool selector | 0.9040 | - | 0.5750 | 0.6464 | top5 only; active 42/1000; obligation cache 100 |
| 2Wiki | NEOCORRAG aligned K1 | Native graph retriever | 0.7558 | 0.8858 | 0.4020 | 0.4553 | native; ret100 greedy k1 |
| 2Wiki | HGRAG aligned | Native graph retriever | 0.7692 | 0.8403 | 0.4700 | 0.5174 | native aligned |
| HotpotQA | PropRAG pool100 top5 | Prop-pool selector | 0.9500 | 0.9925 | 0.5940 | 0.7212 | baseline top5 |
| HotpotQA | ETv3 variable-flow | Native ET retriever/readout | 0.9505 | - | 0.6170 | 0.7330 | method-owned candidate universe; no Prop pool |
| HotpotQA | Prop+DAEC | Prop-pool selector | 0.9625 | 0.9935 | 0.6060 | 0.7348 | same Prop pool |
| HotpotQA | Prop+DAEC-L1 | Prop-pool selector | 0.9460 | 0.9935 | 0.6180 | 0.7370 | same Prop pool |
| HotpotQA | Prop+V13B* | Prop-pool selector | 0.9505 | - | 0.5960 | 0.7224 | top5 only; active 20/1000; obligation cache 100 |
| HotpotQA | NEOCORRAG aligned K1 | Native graph retriever | 0.9200 | 0.9885 | 0.5470 | 0.6686 | native; ret100 greedy k1 |
| HotpotQA | HGRAG aligned | Native graph retriever | 0.9450 | 0.9810 | 0.5910 | 0.7267 | native aligned |
| MuSiQue | PropRAG pool100 top5 | Prop-pool selector | 0.7131 | 0.8942 | 0.3300 | 0.4266 | baseline top5 |
| MuSiQue | ETv3 variable-flow | Native ET retriever/readout | 0.7184 | - | 0.3320 | 0.4319 | method-owned candidate universe; no Prop pool |
| MuSiQue | Prop+DAEC | Prop-pool selector | 0.7433 | 0.9000 | 0.3390 | 0.4404 | same Prop pool |
| MuSiQue | Prop+DAEC-L1 | Prop-pool selector | 0.6892 | 0.9004 | 0.3130 | 0.4070 | same Prop pool |
| MuSiQue | Prop+V13B* | Prop-pool selector | 0.7131 | - | 0.3280 | 0.4269 | top5 only; active 9/1000; obligation cache 100 |
| MuSiQue | NEOCORRAG aligned K1 | Native graph retriever | 0.6647 | 0.8528 | 0.2750 | 0.3650 | native; ret100 greedy k1 |
| MuSiQue | HGRAG aligned | Native graph retriever | 0.6952 | 0.8140 | 0.2970 | 0.3985 | native aligned |

## Current NEO K3 Limit100 Pilot

This is not mixed into the full1000 table.

| Dataset | Status | R@5 | R@20 | R@100 | EM | F1 |
|---|---|---:|---:|---:|---:|---:|
| 2Wiki | done | 0.8250 | 0.9075 | 0.9425 | 0.4800 | 0.5342 |
| HotpotQA | running | - | - | - | - | - |
| MuSiQue | pending | - | - | - | - | - |

## Winners Within This Table

| Dataset | Best EM | Best F1 |
|---|---|---|
| 2Wiki | Prop+DAEC-L1, 0.6130 | Prop+DAEC-L1, 0.6828 |
| HotpotQA | Prop+DAEC-L1, 0.6180 | Prop+DAEC-L1, 0.7370 |
| MuSiQue | Prop+DAEC, 0.3390 | Prop+DAEC, 0.4404 |

## ETv3 Variable-Flow Full1000 Deposition

Result note:

- `docs/etv3_variable_flow_full1000_qwen8b_nv2_20260510.md`

Interpretation:

- ETv3 is a native method-owned retrieval/readout line, not a PropRAG-pool
  composer.  It should be compared directly to bare retrieval/readout baselines
  such as PropRAG top5, HGRAG, and NEOCORRAG.
- Prop+DAEC, Prop+DAEC-L1, and SetR-style results are stronger two-stage
  references because they use an exported PropRAG pool plus a composition
  selector.  They are useful system references, but they should not be phrased
  as direct evidence that bare ETv3 is weaker as an original retrieval method.
- The fair next experiment is ETv3 pool plus baseline-stable DBEC/DAEC versus
  PropRAG pool plus DBEC/DAEC / SetR-style references.
