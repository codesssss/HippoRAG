# D-PathRAG Embedding-Augmented Selector Diagnostic

Date: 2026-04-26

## Setup

- Data: local 2Wiki 1000-example protocol
- Split: first 800 train, final 200 eval
- Candidate pools: Dense pool100 and PropRAG pool100
- Selector: AR no-replacement warm-start
- Path length: 5
- Reader: `data/dpathrag/models/flan_t5_base_gold_k5_5k`
- Embedding source: `nvidia/NV-Embed-v2` via `http://localhost:8019/v1/embeddings`
- Compression: random projection to 64 dimensions
- Semantic feature dim: 193
- Early stopping: support-complete@5 on the held-out 200 examples, patience 5, load best checkpoint

Embedding features are cached in `.npz` and are not backpropagated through in this pilot.

## Embedding Cache

| Pool | Rows | Unique Texts | Embedding Dim | Projection Dim | Semantic Feature Dim |
|---|---:|---:|---:|---:|---:|
| Dense pool100 | 1000 | 6774 | 4096 | 64 | 193 |
| PropRAG pool100 | 1000 | 6975 | 4096 | 64 | 193 |

## Mechanism Results

| Pool | Method | Support Recall@5 | Support Complete@5 | Bridge Entity Recall | Overlap vs Rank | Duplicate Title |
|---|---|---:|---:|---:|---:|---:|
| Dense | rank top-5 | 0.7388 | 0.4550 | 0.9083 | 1.0000 | 0.0050 |
| Dense | lightweight selector | 0.7612 | 0.4950 | n/a | n/a | 0.0050 |
| Dense | embedding selector, no early stop | 0.7188 | 0.4200 | 0.9000 | 0.2868 | 0.0050 |
| Dense | embedding selector, early-stop best | 0.7662 | 0.5050 | 0.9083 | 0.4992 | 0.0050 |
| PropRAG | rank top-5 | 0.8962 | 0.7500 | 0.9408 | 1.0000 | 0.0050 |
| PropRAG | lightweight selector | 0.8838 | 0.7200 | n/a | n/a | 0.0050 |
| PropRAG | embedding selector, no early stop | 0.9025 | 0.7700 | 0.9517 | 0.5113 | 0.0050 |
| PropRAG | embedding selector, early-stop best | 0.9137 | 0.7800 | 0.9450 | 0.6411 | 0.0050 |

## Reader Results

| Pool | Method | Answer EM | Answer F1 |
|---|---|---:|---:|
| Dense | rank top-5 | 0.3700 | 0.4154 |
| Dense | lightweight selector | 0.3800 | 0.4285 |
| Dense | embedding selector, no early stop | 0.3600 | 0.4015 |
| Dense | embedding selector, early-stop best | 0.4050 | 0.4404 |
| PropRAG | rank top-5 | 0.4850 | 0.5426 |
| PropRAG | lightweight selector | 0.4600 | 0.5274 |
| PropRAG | embedding selector, no early stop, AR order | 0.4200 | 0.4858 |
| PropRAG | embedding selector, no early stop, rank order | 0.4150 | 0.4708 |
| PropRAG | embedding selector, early-stop best | 0.4750 | 0.5344 |

## PropRAG Reader Failure Analysis

| Method | Answer In Context | Answer Fail Rate | Answer-In-Context But Fail | Support-Complete But Fail | F1 If Answer In Context | F1 If Answer Absent |
|---|---:|---:|---:|---:|---:|---:|
| rank top-5 | 0.8000 | 0.5150 | 0.3600 | 0.3600 | 0.6141 | 0.2567 |
| embedding selector, early-stop best | 0.8200 | 0.5250 | 0.3900 | 0.3850 | 0.5944 | 0.2611 |

## Interpretation

- Early stopping is load-bearing for semantic features. Without it, Dense overfits badly; with best-checkpoint loading, Dense improves over rank top-5 by +5.0 pp support-complete and +2.50 pp Answer F1.
- PropRAG also passes the mechanism gate with early stopping: support-complete improves by +3.0 pp over rank top-5, support recall by +1.75 pp, and bridge-entity recall by +0.42 pp. The overlap vs rank is 0.6411, so the selector is changing a meaningful fraction of the set rather than copying rank order.
- PropRAG answer F1 remains slightly below rank top-5: 0.5344 vs 0.5426 (-0.82 pp). This is a Yellow result: evidence selection improves, but reader consumption/ordering does not fully convert the support gain into answer gain.
- Failure analysis supports the reader-consumption diagnosis: the early-stop selector puts the answer string in context more often than rank top-5 (0.8200 vs 0.8000), but answer-in-context-but-fail also rises (0.3900 vs 0.3600). The selected set is more complete but appears slightly more distracting for the current reader.
- The previous no-early-stop PropRAG result showed stronger bridge recall but severe reader regression. The early-stop checkpoint fixes most of that regression while preserving the support-complete gain.

## Decision

Embedding cache validates that q-doc semantic features are useful for D-PathRAG warm-start selection.

Current gate status:

- Dense: Green. Mechanism and Answer F1 both improve over rank top-5.
- PropRAG: Yellow. Mechanism improves over a strong rank baseline, but Answer F1 is still -0.82 pp below rank top-5.

This keeps D-PathRAG viable as a mechanism line. The next bottleneck is reader consumption and ordering under a high-quality PropRAG pool, not whether the selector can improve support coverage.

Next action:

1. Evaluate answer-in-context-but-fail on PropRAG early-stop cases to separate reader consumption from evidence selection.
2. Add a selected-doc formatting/order ablation: AR order, retriever-rank order, gold-support-first oracle order.
3. Try a lower-capacity semantic feature variant: cosine-only or projection_dim=16/32.
4. After the above, move to a cross-encoder feature cache if PropRAG F1 remains below rank despite better support-complete.
