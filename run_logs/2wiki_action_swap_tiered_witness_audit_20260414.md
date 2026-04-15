# Action-Level Audit: 2wikimultihopqa

- selected actions: `107`
- oracle-positive selected swaps: `55`
- oracle-negative sampled swaps: `52`
- dryrun executed swaps in audit: `21`
- judge executed swaps in audit: `3`
- heuristic tiers queries: `14`
- flat fallback queries: `24`

## Oracle-Positive Gain Audit

| Group | Count | Median gain | Nonpositive |
|---|---:|---:|---:|
| oracle_positive | 55 | 0.0 | 55 (100.0) |
| oracle_negative | 52 | 0.0 | 52 (100.0) |

## Ranking Recall

- top-1 oracle-positive hit rate: `55.26` over `38` queries
- top-3 oracle-positive hit rate: `55.26` over `38` queries

## Sample Failure Cases

- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Har Dil Jo Pyar Karega` for `Phalitamsha`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Har Dil Jo Pyar Karega` for `Chaowa Pawa (2009 film)`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Rani Mukerji` for `Phalitamsha`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Rani Mukerji` for `Chaowa Pawa (2009 film)`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Goopy Gyne Bagha Byne` for `Phalitamsha`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Goopy Gyne Bagha Byne` for `Chaowa Pawa (2009 film)`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `...Baby One More Time (album)` for `2 Chainz`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `...Baby One More Time (album)` for `Alex da Kid`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `Alicia Keys` for `2 Chainz`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `Alicia Keys` for `Alex da Kid`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
