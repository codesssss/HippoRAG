# ETv4 MuSiQue Chain-Consistency Diagnostic

Retrieval report: `run_logs/etv4_full1000_optimized_equivalence_20260511/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json`
OpenIE: `/mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv4_clean_mainline_multi_anchor_strict_musique100_20260511/musique/index/openie_results_ner_gpt-4o-mini.json`

This is diagnostic-only. Gold is used only to label gain/loss buckets.

## Summary

```text
Hop  Bucket  Rows  Inserted  Consumed%  Evi-consumed%  Evi-supported%  Weak-graph%  Non-consumed%  Weak/isolated%  Tail gold drop%  Mean drop rank
---  ------  ----  --------  ---------  -------------  --------------  -----------  -------------  --------------  ---------------  --------------
2    gain    70    74        28.38      16.22          59.46           40.54        71.62          0.0             0.0              0.0           
2    loss    11    15        80.0       33.33          33.33           66.67        20.0           0.0             100.0            4.7273        
3    gain    33    42        57.14      23.81          64.29           30.95        42.86          4.76            0.0              0.0           
3    loss    29    39        43.59      23.08          38.46           61.54        56.41          0.0             96.55            4.5517        
4    gain    46    68        41.18      14.71          38.24           60.29        58.82          7.35            0.0              0.0           
4    loss    9     13        30.77      15.38          23.08           76.92        69.23          7.69            88.89            4.4444        
all  gain    149   184       39.67      17.39          52.72           45.65        60.33          3.8             0.0              0.0           
all  loss    49    67        49.25      23.88          34.33           65.67        50.75          1.49            95.92            4.5714        
```

## Key Buckets

### MuSiQue 3-hop losses

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
29    39        17              22              9             6         24          0                  0         96.55          
```

### MuSiQue 3-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
33    42        24              16              10            17        13          0                  2         0.0            
```

### MuSiQue 4-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
46    68        28              35              10            16        41          4                  1         0.0            
```

### MuSiQue 2-hop gains

```text
Rows  Inserted  Chain-consumed  Supported leaf  Evi-consumed  Evi-leaf  Weak-graph  Weak/topical leaf  Isolated  Tail gold drop%
----  --------  --------------  --------------  ------------  --------  ----------  -----------------  --------  ---------------
70    74        21              53              12            32        30          0                  0         0.0            
```

## Example Losses

```text
QID  Hits  Dropped gold  Inserted   Inserted classes                   Evidence classes                            Dropped dense rank
---  ----  ------------  ---------  ---------------------------------  ------------------------------------------  ------------------
20   3->2  387           6515       supported_leaf:1                   weak_graph_connected:1                      5                 
79   3->2  1402          6606,5122  supported_leaf:2                   weak_graph_connected:2                      5                 
166  2->1  2803          9782       supported_leaf:1                   evidence_leaf:1                             5                 
188  3->2  876           1469       supported_leaf:1                   weak_graph_connected:1                      5                 
222  2->1  3542          3555       supported_leaf:1                   weak_graph_connected:1                      5                 
223  3->2  1304          1297,1299  chain_consumed:2                   evidence_consumed:1,weak_graph_connected:1  4                 
235  3->2  3748          2664       chain_consumed:1                   weak_graph_connected:1                      5                 
323  2->1  1627          4155,252   chain_consumed:1,supported_leaf:1  weak_graph_connected:2                      4                 
338  3->2  5120          1297       chain_consumed:1                   weak_graph_connected:1                      5                 
346  3->2  911           679,808    chain_consumed:1,supported_leaf:1  weak_graph_connected:2                      5                 
```
