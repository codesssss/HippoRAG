# 2Wiki Oracle Ceiling Analysis

- Report: `outputs_step0_general_2wikimultihopqa/eval_reports/oracle_select_sweep_1000_reorder20.json`
- Queries: `1000`

## Metrics

### baseline_recall

- `Recall@1`: 0.3990
- `Recall@2`: 0.6475
- `Recall@5`: 0.7865
- `Recall@10`: 0.8273
- `Recall@20`: 0.8530

### baseline_full_support

- `FullSupport@1`: 0.0000
- `FullSupport@2`: 0.3390
- `FullSupport@5`: 0.5390
- `FullSupport@10`: 0.6140
- `FullSupport@20`: 0.6590

### oracle_all_missing_recall

- `Recall@1`: 0.4412
- `Recall@2`: 0.8825
- `Recall@5`: 1.0000
- `Recall@10`: 1.0000
- `Recall@20`: 1.0000

### oracle_all_missing_full_support

- `FullSupport@1`: 0.0000
- `FullSupport@2`: 0.7650
- `FullSupport@5`: 1.0000
- `FullSupport@10`: 1.0000
- `FullSupport@20`: 1.0000

### oracle_inject_one_recall

- `Recall@1`: 0.4412
- `Recall@2`: 0.8738
- `Recall@5`: 0.9607
- `Recall@10`: 0.9655
- `Recall@20`: 0.9702

### oracle_inject_one_full_support

- `FullSupport@1`: 0.0000
- `FullSupport@2`: 0.7480
- `FullSupport@5`: 0.8440
- `FullSupport@10`: 0.8620
- `FullSupport@20`: 0.8810

### oracle_bridgeable_recall

- `Recall@1`: 0.4118
- `Recall@2`: 0.7173
- `Recall@5`: 0.8972
- `Recall@10`: 0.9193
- `Recall@20`: 0.9335

### oracle_bridgeable_full_support

- `FullSupport@1`: 0.0000
- `FullSupport@2`: 0.4490
- `FullSupport@5`: 0.7160
- `FullSupport@10`: 0.7680
- `FullSupport@20`: 0.8050

## Anchor vs Bridge Recall

- `anchor_query_count`: 1000
- `bridge_query_count`: 732

### anchor_recall

- `Recall@1`: 0.6430
- `Recall@2`: 0.9090
- `Recall@5`: 0.9970
- `Recall@10`: 0.9990
- `Recall@20`: 0.9995

### anchor_full_support

- `FullSupport@1`: 0.4070
- `FullSupport@2`: 0.8480
- `FullSupport@5`: 0.9940
- `FullSupport@10`: 0.9980
- `FullSupport@20`: 0.9990

### bridge_recall

- `Recall@1`: 0.0396
- `Recall@2`: 0.2036
- `Recall@5`: 0.4228
- `Recall@10`: 0.5301
- `Recall@20`: 0.5997

### bridge_full_support

- `FullSupport@1`: 0.0396
- `FullSupport@2`: 0.1981
- `FullSupport@5`: 0.3743
- `FullSupport@10`: 0.4740
- `FullSupport@20`: 0.5355

## Anchor vs Bridge Depth

- `anchor_depth`: {"count": 1503, "found": 1503, "missing": 0, "median_depth": 1.0, "mean_depth": 1.6, "p90_depth": 3.0, "max_depth": 33}
- `bridge_depth`: {"count": 967, "found": 725, "missing": 242, "median_depth": 6.0, "mean_depth": 29.3, "p90_depth": 100.6, "max_depth": 200}

## Breakdown by Gold Doc Count

### 2-doc queries (n=765)

**baseline_recall**:
- `Recall@1`: 0.4497
- `Recall@2`: 0.7105
- `Recall@5`: 0.8458
- `Recall@10`: 0.8908
- `Recall@20`: 0.9163

**baseline_full_support**:
- `FullSupport@1`: 0.0000
- `FullSupport@2`: 0.4431
- `FullSupport@5`: 0.6915
- `FullSupport@10`: 0.7817
- `FullSupport@20`: 0.8327

### 4-doc queries (n=235)

**baseline_recall**:
- `Recall@1`: 0.2340
- `Recall@2`: 0.4426
- `Recall@5`: 0.5936
- `Recall@10`: 0.6202
- `Recall@20`: 0.6468

**baseline_full_support**:
- `FullSupport@1`: 0.0000
- `FullSupport@2`: 0.0000
- `FullSupport@5`: 0.0426
- `FullSupport@10`: 0.0681
- `FullSupport@20`: 0.0936

## Feasibility

- `queries_with_partial_support_top5`: 461
- `queries_with_bridgeable_missing_support_top5`: 328
- `missing_support_docs_top5`: 618
- `bridgeable_missing_support_docs_top5`: 449
- `bridgeable_missing_doc_rate_top5`: 72.65%
- `partial_query_bridgeable_rate_top5`: 71.15%
- `direct_bridge_count_top5`: 446
- `two_hop_bridge_count_top5`: 3

## Top Relation Counts

- `director`: 611
- `date of birth`: 454
- `father`: 243
- `date of death`: 194
- `country of citizenship`: 192
- `publication date`: 156
- `place of birth`: 128
- `spouse`: 106
- `place of death`: 79
- `mother`: 78
- `country of origin`: 56
- `country`: 53
- `composer`: 35
- `performer`: 32
- `educated at`: 16
- `place of burial`: 14
- `inception`: 11
- `employer`: 10
- `award received`: 8
- `child`: 7

## Top Bridge Relation Paths

- `director`: 377
- `father`: 25
- `spouse`: 21
- `mother`: 10
- `performer`: 5
- `composer`: 4
- `country of citizenship -> country of citizenship`: 3
- `producer`: 3
- `publisher`: 1