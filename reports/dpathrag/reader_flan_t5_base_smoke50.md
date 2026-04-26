# D-PathRAG Reader Baseline Smoke: flan-t5-base

Date: 2026-04-26

## Config

```text
model: google/flan-t5-base
HF_ENDPOINT: https://hf-mirror.com
device: CUDA_VISIBLE_DEVICES=3
batch_size: 8
max_input_tokens: 1024
max_new_tokens: 32
limit: 50
```

## Results

| Input | EM | F1 | Support Recall | Support Complete |
|---|---:|---:|---:|---:|
| validation gold@5 | 0.3400 | 0.4560 | 1.0000 | 1.0000 |
| local1000 dense@5 | 0.3400 | 0.4080 | 0.7150 | 0.4000 |
| local1000 PropRAG@5 | 0.3600 | 0.4213 | 0.9300 | 0.8400 |

## Read

The real HuggingFace seq2seq reader path works, but zero-shot `flan-t5-base` is
too weak for the D-PathRAG gate.  Even with gold support, answer F1 is only
0.456 on the 50-example smoke.  The support exposure gap between Dense/PropRAG
and gold does not cleanly transfer into answer F1 under this reader.

This does not block D-PathRAG implementation, but it means the next step should
not be selector warm-start yet.  First make the same-reader baseline reliable:

```text
Option A: small fine-tune flan-t5-base on train gold-support records.
Option B: evaluate a stronger reader, e.g. flan-t5-large/XL, if GPU memory allows.
```

Observed failure mode: accent/normalization-sensitive entity generation can
hurt exact match. Example: `Małgorzata Braunek` predicted as `Magorzata Braunek`.
