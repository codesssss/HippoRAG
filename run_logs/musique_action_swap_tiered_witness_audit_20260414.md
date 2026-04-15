# Action-Level Audit: musique

- selected actions: `197`
- oracle-positive selected swaps: `148`
- oracle-negative sampled swaps: `49`
- dryrun executed swaps in audit: `16`
- judge executed swaps in audit: `2`
- heuristic tiers queries: `29`
- flat fallback queries: `40`

## Oracle-Positive Gain Audit

| Group | Count | Median gain | Nonpositive |
|---|---:|---:|---:|
| oracle_positive | 148 | 0.0 | 148 (100.0) |
| oracle_negative | 49 | 0.0 | 49 (100.0) |

## Ranking Recall

- top-1 oracle-positive hit rate: `57.97` over `69` queries
- top-3 oracle-positive hit rate: `57.97` over `69` queries

## Sample Failure Cases

- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `List of UEFA club competition winners` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `List of UEFA club competition winners` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Manchester City F.C.` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Manchester City F.C.` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Argentina v England (1986 FIFA World Cup)` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  - action: swap in `Argentina v England (1986 FIFA World Cup)` for `FC Barcelona`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `What year did the publisher of Labyrinth end?`
  - action: swap in `The Lord of the Rings` for `Spectrum HoloByte`
  - oracle ΔEM / ΔF1: `1.0` / `1.0`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  - action: swap in `Riverside Plaza` for `Gulf of Mexico`
  - oracle ΔEM / ΔF1: `0.0` / `0.8`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  - action: swap in `Riverside Plaza` for `Gulf of Mexico`
  - oracle ΔEM / ΔF1: `0.0` / `0.4`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
- question: `Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?`
  - action: swap in `Ohio River` for `Gulf of Mexico`
  - oracle ΔEM / ΔF1: `0.0` / `0.3333`
  - bottleneck tier / facet: `None` / `None`
  - gain vs loss: `0.0` vs `0.0` + `0.0`
  - net score: `0.0`
