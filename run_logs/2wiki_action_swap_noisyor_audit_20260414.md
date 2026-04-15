# Action-Level Audit: 2wikimultihopqa

- action margin: `0.05`
- selected actions: `107`
- oracle-positive selected swaps: `55`
- oracle-negative sampled swaps: `52`
- dryrun executed swaps in audit: `21`
- judge executed swaps in audit: `3`

## Oracle-Positive Delta Audit

| Variant | Count | Median Δ | Nonpositive | <= margin |
|---|---:|---:|---:|---:|
| flat_top2 | 55 | 0.0 | 55 (100.0) | 55 (100.0) |
| dep_top2 | 55 | 0.0 | 55 (100.0) | 55 (100.0) |
| flat_max | 55 | 0.0 | 55 (100.0) | 55 (100.0) |
| flat_full | 55 | 0.0 | 52 (94.55) | 55 (100.0) |

## Ranking Recall

| Ranking | Top-1 positive hit rate | Top-3 positive hit rate | Queries |
|---|---:|---:|---:|
| ranking_flat_max | 44.74 | 55.26 | 38 |
| ranking_flat_full | 52.63 | 55.26 | 38 |
| ranking_flat_top2 | 44.74 | 55.26 | 38 |
| ranking_dep_top2 | 44.74 | 55.26 | 38 |

## Dep vs Flat

- dep-triggered queries: `21`
- flat/dep ranking changed: `1` (`4.76`)
- flat/dep top-1 action changed: `0` (`0.0`)

## Sample Failure Cases

- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Har Dil Jo Pyar Karega` for `Phalitamsha`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0006` / `0.0051`
  - incumbent ψ(flat/dep): `0.0004` / `0.0147`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Har Dil Jo Pyar Karega` for `Chaowa Pawa (2009 film)`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0006` / `0.0051`
  - incumbent ψ(flat/dep): `0.0002` / `0.0015`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Rani Mukerji` for `Phalitamsha`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0015` / `0.0059`
  - incumbent ψ(flat/dep): `0.0004` / `0.0147`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Rani Mukerji` for `Chaowa Pawa (2009 film)`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0015` / `0.0059`
  - incumbent ψ(flat/dep): `0.0002` / `0.0015`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Goopy Gyne Bagha Byne` for `Phalitamsha`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0005` / `0.0075`
  - incumbent ψ(flat/dep): `0.0004` / `0.0147`
- question: `Which film was released first, Aas Ka Panchhi or Phoolwari?`
  - action: swap in `Goopy Gyne Bagha Byne` for `Chaowa Pawa (2009 film)`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0005` / `0.0075`
  - incumbent ψ(flat/dep): `0.0002` / `0.0015`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `...Baby One More Time (album)` for `2 Chainz`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.003` / `0.003`
  - incumbent ψ(flat/dep): `0.1351` / `0.1351`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `...Baby One More Time (album)` for `Alex da Kid`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.003` / `0.003`
  - incumbent ψ(flat/dep): `0.0831` / `0.0831`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `Alicia Keys` for `2 Chainz`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.1894` / `0.1894`
  - incumbent ψ(flat/dep): `0.1351` / `0.1351`
- question: `What is the place of birth of the performer of song Changed It?`
  - action: swap in `Alicia Keys` for `Alex da Kid`
  - oracle ΔEM / ΔF1: `0.0` / `0.5455`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.1894` / `0.1894`
  - incumbent ψ(flat/dep): `0.0831` / `0.0831`
