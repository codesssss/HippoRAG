# Action-Level Audit: musique

- action margin: `0.05`
- selected actions: `197`
- oracle-positive selected swaps: `148`
- oracle-negative sampled swaps: `49`
- dryrun executed swaps in audit: `16`
- judge executed swaps in audit: `2`

## Oracle-Positive Delta Audit

| Variant | Count | Median Δ | Nonpositive | <= margin |
|---|---:|---:|---:|---:|
| flat_top2 | 148 | 0.0 | 145 (97.97) | 148 (100.0) |
| dep_top2 | 148 | 0.0 | 145 (97.97) | 148 (100.0) |
| flat_max | 148 | 0.0 | 148 (100.0) | 148 (100.0) |
| flat_full | 148 | -0.0027 | 134 (90.54) | 148 (100.0) |

## Ranking Recall

| Ranking | Top-1 positive hit rate | Top-3 positive hit rate | Queries |
|---|---:|---:|---:|
| ranking_flat_max | 44.93 | 55.07 | 69 |
| ranking_flat_full | 44.93 | 56.52 | 69 |
| ranking_flat_top2 | 47.83 | 55.07 | 69 |
| ranking_dep_top2 | 47.83 | 55.07 | 69 |

## Dep vs Flat

- dep-triggered queries: `30`
- flat/dep ranking changed: `2` (`6.67`)
- flat/dep top-1 action changed: `1` (`3.33`)

## Sample Failure Cases

- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `List of UEFA club competition winners` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0002` / `0.0004`
  - incumbent ψ(flat/dep): `0.07` / `0.0988`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `List of UEFA club competition winners` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0002` / `0.0004`
  - incumbent ψ(flat/dep): `0.0965` / `0.1629`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Manchester City F.C.` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0031` / `0.0071`
  - incumbent ψ(flat/dep): `0.07` / `0.0988`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Manchester City F.C.` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0031` / `0.0071`
  - incumbent ψ(flat/dep): `0.0965` / `0.1629`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Argentina v England (1986 FIFA World Cup)` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0034` / `0.0062`
  - incumbent ψ(flat/dep): `0.07` / `0.0988`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Argentina v England (1986 FIFA World Cup)` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0034` / `0.0062`
  - incumbent ψ(flat/dep): `0.0965` / `0.1629`
- question: `What year did the publisher of Labyrinth end?`
  - action: swap in `The Lord of the Rings` for `Spectrum HoloByte`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0122` / `0.0122`
  - incumbent ψ(flat/dep): `0.0056` / `0.0056`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  - action: swap in `Riverside Plaza` for `Gulf of Mexico`
  - oracle ΔEM / ΔF1: `0.0` / `0.8`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0` / `0.0001`
  - incumbent ψ(flat/dep): `0.0603` / `0.1052`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  - action: swap in `Riverside Plaza` for `Gulf of Mexico`
  - oracle ΔEM / ΔF1: `0.0` / `0.4`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0` / `0.0001`
  - incumbent ψ(flat/dep): `0.0984` / `0.1741`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  - action: swap in `Ohio River` for `Gulf of Mexico`
  - oracle ΔEM / ΔF1: `0.0` / `0.3333`
  - flat Δ / dep Δ: `0.0` / `0.0`
  - candidate ψ(flat/dep): `0.0004` / `0.0006`
  - incumbent ψ(flat/dep): `0.0603` / `0.1052`
