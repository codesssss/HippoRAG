# DAEC Saturation-Aware Binding Verifier Probe

This is a post-hoc audit only. It does not rerun the reader and does not change selector outputs.

- Objective epsilon: `0.01`
- Support mode: `selected_companion`
- Policy: `keep base unless objective-tied, base unsupported, and a tied alternative has structural support`

```text
+----------+------+-------+---------+---------+----------+----------+----------+
| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |
+----------+------+-------+---------+---------+----------+----------+----------+
| 2Wiki    |  100 |    22 |      82 |       2 |        0 |        0 |  +0.0000 |
| HotpotQA |  100 |    22 |      70 |       1 |        0 |        0 |  +0.0000 |
| MuSiQue  |  100 |    29 |      57 |       8 |        1 |        0 |  -0.0050 |
+----------+------+-------+---------+---------+----------+----------+----------+
```

Decision counts:

```text
+----------+----------------------+-------+
| Dataset  | Decision             | Count |
+----------+----------------------+-------+
| 2Wiki    | flip_alt_supported   |     2 |
| 2Wiki    | keep_base_supported  |    16 |
| 2Wiki    | keep_no_supported_alt |     4 |
| 2Wiki    | keep_not_tied        |    78 |
| HotpotQA | flip_alt_supported   |     1 |
| HotpotQA | keep_base_supported  |    11 |
| HotpotQA | keep_no_supported_alt |    10 |
| HotpotQA | keep_not_tied        |    78 |
| MuSiQue  | flip_alt_supported   |     8 |
| MuSiQue  | keep_base_supported  |     6 |
| MuSiQue  | keep_no_supported_alt |    15 |
| MuSiQue  | keep_not_tied        |    71 |
+----------+----------------------+-------+
```

Flip examples:

```text
+----------+-------+----------------------+----------------------+----------+
| Dataset  | Query | Base                 | Verifier             | dRecall  |
+----------+-------+----------------------+----------------------+----------+
| 2wikimultihopqa |    44 | b0                   | b1                   |  +0.0000 |
| 2wikimultihopqa |    65 | b1                   | b0                   |  +0.0000 |
| hotpotqa |     7 | b4                   | b0                   |  +0.0000 |
| musique  |     7 | b0                   | b5                   |  +0.2500 |
| musique  |    17 | b1                   | b0                   |  -0.5000 |
| musique  |    39 | b0                   | b1                   |  +0.0000 |
| musique  |    44 | b3                   | b0                   |  +0.0000 |
| musique  |    49 | b0                   | b3                   |  +0.0000 |
| musique  |    68 | b0                   | b2                   |  +0.0000 |
| musique  |    94 | b2                   | b1                   |  -0.2500 |
| musique  |    99 | b0                   | b1                   |  +0.0000 |
+----------+-------+----------------------+----------------------+----------+
```
