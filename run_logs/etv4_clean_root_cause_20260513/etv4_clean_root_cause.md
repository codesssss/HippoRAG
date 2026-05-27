# ETv4 Clean Root-Cause Diagnostic

Diagnostic-only. Gold labels are used only for offline error classification.

## 2wikimultihopqa vs PropRAG

```text
Metric                  ETv4      Baseline
----------------------  --------  --------
Title R@5               0.9350    0.9090
Title all-gold@5        0.8260    0.7800
Title pool all-gold@200 0.9840    0.9890
EM                      0.6470    0.6110
F1                      0.7329    0.6909
```

```text
PropRAG EM wins: 43
ETv4 EM wins: 79
EM ties: 878
PropRAG win buckets: {'ETV4_pool200_missing_title_gold': 1, 'ETV4_reader_wrong_with_title_all_gold': 24, 'ETV4_readout_misses_title_gold': 18}
PropRAG title-all@5 only: 85
ETv4 title-all@5 only: 131
PropRAG title-pool-all@200 only: 15
ETv4 title-pool-all@200 only: 10
```

## hotpotqa vs PropRAG

```text
Metric                  ETv4      Baseline
----------------------  --------  --------
Title R@5               0.9505    0.9510
Title all-gold@5        0.9050    0.9070
Title pool all-gold@200 0.9970    0.9990
EM                      0.6100    0.6180
F1                      0.7413    0.7510
```

```text
PropRAG EM wins: 39
ETv4 EM wins: 31
EM ties: 930
PropRAG win buckets: {'ETV4_reader_wrong_with_title_all_gold': 20, 'ETV4_readout_misses_title_gold': 19}
PropRAG title-all@5 only: 50
ETv4 title-all@5 only: 48
PropRAG title-pool-all@200 only: 2
ETv4 title-pool-all@200 only: 0
```

## musique vs PropRAG

```text
Metric                  ETv4      Baseline
----------------------  --------  --------
Title R@5               0.7442    0.7424
Title all-gold@5        0.4720    0.4760
Title pool all-gold@200 0.8950    0.9530
EM                      0.3670    0.3650
F1                      0.4801    0.4760
```

```text
PropRAG EM wins: 77
ETv4 EM wins: 79
EM ties: 844
PropRAG win buckets: {'ETV4_pool200_missing_title_gold': 9, 'ETV4_reader_wrong_with_title_all_gold': 22, 'ETV4_readout_misses_title_gold': 46}
PropRAG title-all@5 only: 99
ETv4 title-all@5 only: 95
PropRAG title-pool-all@200 only: 67
ETv4 title-pool-all@200 only: 9
```
