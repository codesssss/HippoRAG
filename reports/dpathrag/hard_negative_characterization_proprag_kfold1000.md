# D-PathRAG Hard-Negative Characterization

- Rows: `1000`
- Top-k: `5`
- Max candidates: `100`

## Category Means

| Category | Docs | Queries | rank | retriever_score | title_question_jaccard | body_question_jaccard | question_token_coverage | q_doc_cosine | answer_in_doc | bridge_entity_in_doc | log_doc_chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selector_added_non_gold | 1205 | 807 | 18.0083 | 0.0025 | 0.0373 | 0.0572 | 0.3190 | 0.1895 | 0.0407 | 0.0481 | 6.1481 |
| rank_retained_non_gold | 1529 | 827 | 3.3479 | 0.0057 | 0.0685 | 0.0723 | 0.3628 | 0.2629 | 0.0837 | 0.1262 | 6.0471 |
| rank_removed_non_gold | 1244 | 826 | 4.3344 | 0.0037 | 0.0683 | 0.0802 | 0.2586 | 0.2128 | 0.0362 | 0.0908 | 5.2046 |
| selector_added_gold | 76 | 76 | 9.6842 | 0.0032 | 0.0908 | 0.0662 | 0.3540 | 0.2635 | 0.5263 | 0.7632 | 6.2374 |

## Mean Differences

### selector_added_non_gold_minus_rank_retained_non_gold

| Feature | Mean Delta |
|---|---:|
| rank | +14.6604 |
| retriever_score | -0.0032 |
| title_question_jaccard | -0.0312 |
| body_question_jaccard | -0.0151 |
| question_token_coverage | -0.0437 |
| q_doc_cosine | -0.0735 |
| answer_in_doc | -0.0431 |
| bridge_entity_in_doc | -0.0781 |
| log_doc_chars | +0.1010 |

### selector_added_non_gold_minus_rank_removed_non_gold

| Feature | Mean Delta |
|---|---:|
| rank | +13.6739 |
| retriever_score | -0.0012 |
| title_question_jaccard | -0.0310 |
| body_question_jaccard | -0.0230 |
| question_token_coverage | +0.0604 |
| q_doc_cosine | -0.0233 |
| answer_in_doc | +0.0045 |
| bridge_entity_in_doc | -0.0427 |
| log_doc_chars | +0.9434 |

### selector_added_gold_minus_selector_added_non_gold

| Feature | Mean Delta |
|---|---:|
| rank | -8.3241 |
| retriever_score | +0.0007 |
| title_question_jaccard | +0.0536 |
| body_question_jaccard | +0.0090 |
| question_token_coverage | +0.0350 |
| q_doc_cosine | +0.0741 |
| answer_in_doc | +0.4857 |
| bridge_entity_in_doc | +0.7150 |
| log_doc_chars | +0.0893 |
