# HippoRAG Pool Bridge Append Parity Limit100

Config: Qwen3-8B, HippoRAG legacy_fact_graph pool100, legacy causal path, structure_rerank_enabled=true.

| Dataset | Variant | Base EM | EM | dEM | Base F1 | F1 | dF1 | Base R@5 | R@5 | dR@5 | SeedEnt | StructDoc | AppendN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | bridge_daec | 0.5100 | 0.5500 | 0.0400 | 0.5470 | 0.6124 | 0.0654 | 0.8150 | 0.8675 | 0.0525 | 4.8700 | 4.8400 | 0.4000 |
| 2wikimultihopqa | bridge_ce | 0.5100 | 0.4500 | -0.0600 | 0.5470 | 0.5112 | -0.0358 | 0.8150 | 0.8225 | 0.0075 | 4.8700 | 4.9400 | 0.4000 |
| hotpotqa | bridge_daec | 0.5700 | 0.6300 | 0.0600 | 0.6872 | 0.7322 | 0.0450 | 0.9250 | 0.9550 | 0.0300 | 4.6100 | 5.0000 | 0.6200 |
| hotpotqa | bridge_ce | 0.5700 | 0.6000 | 0.0300 | 0.6872 | 0.7296 | 0.0424 | 0.9250 | 0.9500 | 0.0250 | 4.6100 | 4.9900 | 0.6200 |
| musique | bridge_daec | 0.3000 | 0.3100 | 0.0100 | 0.3673 | 0.4086 | 0.0413 | 0.6417 | 0.6925 | 0.0508 | 5.4700 | 5.0000 | 1.7000 |
| musique | bridge_ce | 0.3000 | 0.3500 | 0.0500 | 0.3673 | 0.4134 | 0.0461 | 0.6417 | 0.6783 | 0.0366 | 5.4700 | 4.9900 | 1.7000 |
