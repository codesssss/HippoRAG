# AREC Fixed-Pool Projection Control

- Dataset: `musique`
- Limit: `100`
- Projection: closure greedy top-5 inside fixed top-N pool, using cached NLI support matrices.

| Source | Pool | Mode | Rows | Recall@5 | SC@5 | Initial Recall@5 | Initial SC@5 | Avg Selected |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| generated | 5 | greedy_fill_rank | 100 | 0.6458 | 0.2900 | 0.6458 | 0.2900 | 5.0000 |
| generated | 5 | greedy_only | 100 | 0.6458 | 0.2900 | 0.6458 | 0.2900 | 5.0000 |
| generated | 20 | greedy_fill_rank | 100 | 0.4933 | 0.2300 | 0.6458 | 0.2900 | 5.0000 |
| generated | 20 | greedy_only | 100 | 0.4933 | 0.2300 | 0.6458 | 0.2900 | 5.0000 |
| generated | 100 | greedy_fill_rank | 100 | 0.3042 | 0.1300 | 0.6458 | 0.2900 | 5.0000 |
| generated | 100 | greedy_only | 100 | 0.3042 | 0.1300 | 0.6458 | 0.2900 | 5.0000 |
| silver_oracle | 5 | greedy_fill_rank | 100 | 0.6458 | 0.2900 | 0.6458 | 0.2900 | 5.0000 |
| silver_oracle | 5 | greedy_only | 100 | 0.6458 | 0.2900 | 0.6458 | 0.2900 | 5.0000 |
| silver_oracle | 20 | greedy_fill_rank | 100 | 0.7417 | 0.4500 | 0.6458 | 0.2900 | 5.0000 |
| silver_oracle | 20 | greedy_only | 100 | 0.7417 | 0.4500 | 0.6458 | 0.2900 | 5.0000 |
| silver_oracle | 100 | greedy_fill_rank | 100 | 0.6767 | 0.3700 | 0.6458 | 0.2900 | 5.0000 |
| silver_oracle | 100 | greedy_only | 100 | 0.6767 | 0.3700 | 0.6458 | 0.2900 | 5.0000 |
