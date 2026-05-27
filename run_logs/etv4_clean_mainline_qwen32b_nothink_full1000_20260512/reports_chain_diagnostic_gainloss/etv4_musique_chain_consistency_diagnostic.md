# ETv4 MuSiQue Chain-Consistency Diagnostic

Retrieval report: `/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json`
OpenIE: `/mnt/nvme/code/HippoRAG/run_logs/etv4_clean_mainline_qwen32b_nothink_full1000_20260512/musique/index/openie_results_ner_qwen3-32b-judge.json`

This is diagnostic-only. Gold is used only to label gain/loss buckets.

## Summary

```text
Hop  Bucket  Rows  Inserted  Consumed%  Evi-consumed%  Evi-supported%  Weak-graph%  Non-consumed%  Weak/isolated%  Tail gold drop%  Mean drop rank
---  ------  ----  --------  ---------  -------------  --------------  -----------  -------------  --------------  ---------------  --------------
2    gain    69    74        24.32      16.22          56.76           43.24        75.68          1.35            0.0              0.0           
2    loss    15    22        72.73      27.27          36.36           63.64        27.27          4.55            100.0            4.6           
3    gain    35    49        48.98      22.45          57.14           38.78        51.02          8.16            0.0              0.0           
3    loss    30    43        39.53      20.93          39.53           60.47        60.47          0.0             93.55            4.5484        
4    gain    32    49        38.78      18.37          57.14           42.86        61.22          6.12            0.0              0.0           
4    loss    8     10        20.0       0.0            0.0             90.0         80.0           20.0            100.0            4.625         
all  gain    136   172       35.47      18.6           56.98           41.86        64.53          4.65            0.0              0.0           
all  loss    53    75        46.67      20.0           33.33           65.33        53.33          4.0             96.3             4.5741        
```

## Key Buckets

### MuSiQue 3-hop losses

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
30    43        17              26              9             8         26          0                  0         93.55          
```

### MuSiQue 3-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
35    49        24              21              11            17        19          2                  2         0.0            
```

### MuSiQue 4-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
32    49        19              27              9             19        21          3                  0         0.0            
```

### MuSiQue 2-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
69    74        18              55              12            30        32          1                  0         0.0            
```

## Example Losses

```text
QID  Hits  Dropped gold  Inserted        Inserted classes                   Evidence classes                            Dropped dense rank
---  ----  ------------  --------------  ---------------------------------  ------------------------------------------  ------------------
48   2->0  680,913       2435,1473,1859  chain_consumed:1,supported_leaf:2  evidence_consumed:1,evidence_leaf:2         3,4               
79   3->2  1402          1145,5122       supported_leaf:2                   weak_graph_connected:2                      5                 
159  3->2  537           4533            supported_leaf:1                   weak_graph_connected:1                      5                 
166  2->1  2803          9782            supported_leaf:1                   evidence_leaf:1                             5                 
188  3->2  876           1469            supported_leaf:1                   weak_graph_connected:1                      5                 
222  2->1  3542          3555            supported_leaf:1                   weak_graph_connected:1                      5                 
223  3->2  1304          1297,1299       chain_consumed:2                   evidence_consumed:1,weak_graph_connected:1  4                 
235  3->2  3748          5532            chain_consumed:1                   weak_graph_connected:1                      5                 
313  3->2  399           6645            supported_leaf:1                   weak_graph_connected:1                      5                 
323  2->1  1627          4155,17         chain_consumed:1,supported_leaf:1  weak_graph_connected:2                      4                 
```
