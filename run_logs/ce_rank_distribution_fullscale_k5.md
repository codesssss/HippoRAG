# CE-Rank Distribution of Appended Docs (Full-Scale K=5)

| Dataset | Append Policy | Queries w/ Append | Total Appended | Mean CE Rank | In Top-5 (%) | Rank 6-10 (%) | Rank 11+ (%) | Final Penetration (%) | CE Score Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| musique | bridge_append_plus_ce | 701 | 1741 | 9.3670 | 12.6364 | 42.5617 | 44.8018 | 12.6400 | 2.5182 |
| musique | random3_deep_plus_ce | 962 | 2724 | 9.3073 | 14.7944 | 40.0514 | 45.1542 | 14.7900 | 2.8559 |
| hotpotqa | bridge_append_plus_ce | 371 | 582 | 8.6924 | 15.9794 | 52.5773 | 31.4433 | 15.9800 | 3.3377 |
| hotpotqa | random3_deep_plus_ce | 1000 | 3000 | 9.9193 | 9.5000 | 37.9333 | 52.5667 | 9.5000 | 4.6402 |
| 2wikimultihopqa | bridge_append_plus_ce | 379 | 545 | 7.0954 | 35.4128 | 48.0734 | 16.5138 | 35.4100 | 2.0302 |
| 2wikimultihopqa | random3_deep_plus_ce | 1000 | 3000 | 8.6373 | 20.8000 | 43.5667 | 35.6333 | 20.8000 | 3.2178 |

## musique / bridge_append_plus_ce

- report: `outputs_step0_general_musique/eval_reports/bridge_append_plus_ce_qatopk5_20260409fullfix.json`
- appended docs analyzed: `1741`
- mean / median CE rank: `9.3670 / 10.0000`
- final penetration: `12.6400%`

| Rank | Count |
|---:|---:|
| 1 | 5 |
| 2 | 36 |
| 3 | 48 |
| 4 | 69 |
| 5 | 62 |
| 6 | 116 |
| 7 | 131 |
| 8 | 147 |
| 9 | 153 |
| 10 | 194 |
| 11 | 241 |
| 12 | 263 |
| 13 | 276 |

## musique / random3_deep_plus_ce

- report: `outputs_step0_general_musique/eval_reports/random3_deep_plus_ce_qatopk5_20260409fullfix.json`
- appended docs analyzed: `2724`
- mean / median CE rank: `9.3073 / 10.0000`
- final penetration: `14.7900%`

| Rank | Count |
|---:|---:|
| 1 | 8 |
| 2 | 56 |
| 3 | 84 |
| 4 | 128 |
| 5 | 127 |
| 6 | 151 |
| 7 | 187 |
| 8 | 233 |
| 9 | 239 |
| 10 | 281 |
| 11 | 400 |
| 12 | 396 |
| 13 | 434 |

## hotpotqa / bridge_append_plus_ce

- report: `outputs_step0_general_hotpotqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409fullfix.json`
- appended docs analyzed: `582`
- mean / median CE rank: `8.6924 / 9.0000`
- final penetration: `15.9800%`

| Rank | Count |
|---:|---:|
| 1 | 0 |
| 2 | 9 |
| 3 | 15 |
| 4 | 22 |
| 5 | 47 |
| 6 | 42 |
| 7 | 53 |
| 8 | 64 |
| 9 | 60 |
| 10 | 87 |
| 11 | 97 |
| 12 | 57 |
| 13 | 29 |

## hotpotqa / random3_deep_plus_ce

- report: `outputs_step0_general_hotpotqa/eval_reports/width_match_random3_deep_plus_ce_qatopk5_20260409fullfix.json`
- appended docs analyzed: `3000`
- mean / median CE rank: `9.9193 / 11.0000`
- final penetration: `9.5000%`

| Rank | Count |
|---:|---:|
| 1 | 3 |
| 2 | 16 |
| 3 | 47 |
| 4 | 90 |
| 5 | 129 |
| 6 | 154 |
| 7 | 183 |
| 8 | 214 |
| 9 | 247 |
| 10 | 340 |
| 11 | 479 |
| 12 | 506 |
| 13 | 592 |

## 2wikimultihopqa / bridge_append_plus_ce

- report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409fullfix.json`
- appended docs analyzed: `545`
- mean / median CE rank: `7.0954 / 7.0000`
- final penetration: `35.4100%`

| Rank | Count |
|---:|---:|
| 1 | 0 |
| 2 | 17 |
| 3 | 45 |
| 4 | 58 |
| 5 | 73 |
| 6 | 62 |
| 7 | 50 |
| 8 | 54 |
| 9 | 52 |
| 10 | 44 |
| 11 | 52 |
| 12 | 27 |
| 13 | 11 |

## 2wikimultihopqa / random3_deep_plus_ce

- report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_random3_deep_plus_ce_qatopk5_20260409fullfix.json`
- appended docs analyzed: `3000`
- mean / median CE rank: `8.6373 / 9.0000`
- final penetration: `20.8000%`

| Rank | Count |
|---:|---:|
| 1 | 0 |
| 2 | 57 |
| 3 | 147 |
| 4 | 193 |
| 5 | 227 |
| 6 | 252 |
| 7 | 233 |
| 8 | 257 |
| 9 | 283 |
| 10 | 282 |
| 11 | 328 |
| 12 | 357 |
| 13 | 384 |
