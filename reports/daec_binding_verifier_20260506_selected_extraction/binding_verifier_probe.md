# DAEC Saturation-Aware Binding Verifier Probe

This is a post-hoc audit only. It does not rerun the reader and does not change selector outputs.

- Objective epsilon: `1e-09`
- Support mode: `selected_extraction_or_companion`
- Policy: `keep base unless objective-tied, base unsupported, and a tied alternative has structural support`

```text
+----------+------+-------+---------+---------+----------+----------+----------+
| Dataset  | Rows | Tied  | BaseSup | Flips   | SC 1->0  | SC 0->1  | dRecall  |
+----------+------+-------+---------+---------+----------+----------+----------+
| 2Wiki    |  100 |    21 |      82 |       1 |        0 |        0 |  +0.0000 |
| HotpotQA |  100 |    21 |      70 |       0 |        0 |        0 |  +0.0000 |
| MuSiQue  |  100 |    24 |      57 |       3 |        0 |        0 |  +0.0025 |
+----------+------+-------+---------+---------+----------+----------+----------+
```

Decision counts:

```text
+----------+----------------------+-------+
| Dataset  | Decision             | Count |
+----------+----------------------+-------+
| 2Wiki    | flip_alt_supported   |     1 |
| 2Wiki    | keep_base_supported  |    16 |
| 2Wiki    | keep_no_supported_alt |     4 |
| 2Wiki    | keep_not_tied        |    79 |
| HotpotQA | keep_base_supported  |    11 |
| HotpotQA | keep_no_supported_alt |    10 |
| HotpotQA | keep_not_tied        |    79 |
| MuSiQue  | flip_alt_supported   |     3 |
| MuSiQue  | keep_base_supported  |     4 |
| MuSiQue  | keep_no_supported_alt |    17 |
| MuSiQue  | keep_not_tied        |    76 |
+----------+----------------------+-------+
```

Flip examples:

```text
+----------+-------+----------------------+----------------------+----------+
| Dataset  | Query | Base                 | Verifier             | dRecall  |
+----------+-------+----------------------+----------------------+----------+
| 2wikimultihopqa |    44 | b0                   | b1                   |  +0.0000 |
| musique  |     7 | b0                   | b5                   |  +0.2500 |
| musique  |    68 | b0                   | b2                   |  +0.0000 |
| musique  |    99 | b0                   | b1                   |  +0.0000 |
+----------+-------+----------------------+----------------------+----------+
```
