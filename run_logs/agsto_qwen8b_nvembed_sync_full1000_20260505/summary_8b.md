# AG-STO Qwen3-8B + NV-Embed Sync Full1000

Date: 2026-05-05

This run is the 8B-synchronized AG-STO + DAEC checkpoint. Keep it labeled as
`qwen3-8b` / `qwen3-8b-train` when comparing with later 32B reruns.

## Configuration

- Pool exporter: `scripts/export_agsto_pool.py`
- Eval harness: `scripts/eval_causal_qwen3.py`
- OpenIE graph source: `outputs_step0_general_nvembed_{dataset}/openie_results_ner_qwen3-8b.json`
- QA / decomposition / LLM binding model: `qwen3-8b-train`
- QA ports: `musique=8041`, `hotpotqa=8042`, `2wikimultihopqa=8043`
- Embedding model: `NV-Embed-v2`
- Embedding API model: `nvidia/NV-Embed-v2`
- Eval embedding name: `VLLM/nvidia/NV-Embed-v2`
- Native dense anchor: enabled, `top_k=20`
- Semantic residual: enabled, weight `12.0`
- Dense query instruction mode: `raw`
- DAEC selector: `daec_noisyor_llm`
- Binding title match mode: `wiki_title`
- Qwen thinking: disabled with `/no_think` and `enable_thinking=false`

## Main Comparison

Metric format: `EM / F1 / R@5 / R@100`.

```text
+----------+-----------------------------+----------------------------+----------------------------+
| Dataset  | Bare HippoRAG EM/F1/R5/R100 | Bare ProPRAG EM/F1/R5/R100 | AG-STO 8B EM/F1/R5/R100    |
+----------+-----------------------------+----------------------------+----------------------------+
| 2Wiki    | 0.524/0.5850/0.8313/0.9545  | 0.575/0.6457/0.9028/0.9872 | 0.605/0.6796/0.9420/0.9692 |
| HotpotQA | 0.577/0.7010/0.9230/0.9965  | 0.595/0.7227/0.9500/0.9990 | 0.611/0.7293/0.9345/0.9785 |
| MuSiQue  | 0.311/0.3947/0.7011/0.9382  | 0.330/0.4266/0.7372/0.9689 | 0.308/0.4011/0.6819/0.8406 |
| Avg      | 0.471/0.5602/0.8185/0.9631  | 0.500/0.5983/0.8633/0.9850 | 0.508/0.6033/0.8528/0.9294 |
+----------+-----------------------------+----------------------------+----------------------------+
```

## Delta

Metric format: `EM / F1 / R@5 / R@100`.

```text
+----------+--------------------------------+--------------------------------+
| Dataset  | AG-STO 8B - HippoRAG           | AG-STO 8B - ProPRAG            |
+----------+--------------------------------+--------------------------------+
| 2Wiki    | +0.081/+0.0946/+0.1107/+0.0147 | +0.030/+0.0339/+0.0392/-0.0180 |
| HotpotQA | +0.034/+0.0283/+0.0115/-0.0180 | +0.016/+0.0066/-0.0155/-0.0205 |
| MuSiQue  | -0.003/+0.0064/-0.0192/-0.0976 | -0.022/-0.0255/-0.0553/-0.1283 |
| Avg      | +0.037/+0.0431/+0.0343/-0.0336 | +0.008/+0.0050/-0.0105/-0.0556 |
+----------+--------------------------------+--------------------------------+
```

## Notes

- AG-STO 8B + DAEC beats bare HippoRAG on average QA metrics.
- It is only slightly above bare ProPRAG on average QA metrics.
- Pool recall is weaker than ProPRAG and weaker than HippoRAG on average R@100, so the gain is from DAEC selection rather than a stronger pool.
- MuSiQue remains the main regression case.
