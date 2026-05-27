# ETv4 MuSiQue Chain-Consistency Diagnostic

Retrieval report: `run_logs/etv4_full1000_optimized_equivalence_20260511/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json`
OpenIE: `/mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv4_clean_mainline_multi_anchor_strict_musique100_20260511/musique/index/openie_results_ner_gpt-4o-mini.json`

This is diagnostic-only. Gold is used only to label gain/loss buckets.

## Summary

```text
Hop  Bucket  Rows  Inserted  Consumed%  Evi-consumed%  Evi-supported%  Weak-graph%  Non-consumed%  Weak/isolated%  Tail gold drop%  Mean drop rank
---  ------  ----  --------  ---------  -------------  --------------  -----------  -------------  --------------  ---------------  --------------
2    gain    3     3         0.0        0.0            33.33           66.67        100.0          0.0             0.0              0.0           
3    gain    1     2         50.0       0.0            0.0             100.0        50.0           0.0             0.0              0.0           
4    gain    2     3         33.33      0.0            0.0             100.0        66.67          33.33           0.0              0.0           
all  gain    6     8         25.0       0.0            12.5            87.5         75.0           12.5            0.0              0.0           
```

## Key Buckets

### MuSiQue 3-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
1     2         1               1               0             0         2           0                  0         0.0            
```

### MuSiQue 4-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
2     3         1               1               0             0         3           1                  0         0.0            
```

### MuSiQue 2-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
3     3         0               3               0             1         2           0                  0         0.0            
```

## Example Losses

```text
QID  Hits  Dropped gold  Inserted  Inserted classes  Evidence classes  Dropped dense rank
---  ----  ------------  --------  ----------------  ----------------  ------------------
```
