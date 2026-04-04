# G1 Residual Precision Failure Audit

- Query count: 40
- Selector: `bridge_beam` / score mode `bridge`
- Positive changed queries: 3
- Negative changed queries: 2

## Key Findings

- G1 changed 5 queries: 3 positive and 2 negative.
- 2/2 negative queries end in fully saturated beam states (path_connectivity, reachable_doc_ratio, query_reachability, query_coverage all ~1.0).
- 1/3 positive queries are also saturated, so saturation alone cannot serve as a safe precision gate.
- max_local_structure overlaps between positive and negative queries, so current absolute scores do not cleanly separate true bridges from topical noise.
- max_local_novelty overlaps between positive and negative queries, so current absolute scores do not cleanly separate true bridges from topical noise.
- max_local_closure overlaps between positive and negative queries, so current absolute scores do not cleanly separate true bridges from topical noise.
- support_mean overlaps between positive and negative queries, so current absolute scores do not cleanly separate true bridges from topical noise.
- state_score overlaps between positive and negative queries, so current absolute scores do not cleanly separate true bridges from topical noise.
- The remaining hard failure family is not missing-bridge capacity anymore. It is high-structure topical expansion: the beam can assign bridge-like local scores and high state scores to documents that stay inside a dense seed/query cluster.
- The next precision signal should be incremental rather than absolute: measure whether a candidate adds chain utility beyond the current state, instead of only requiring high structure/novelty/closure in isolation.

## Feature Overlap

| Metric | Positive Range | Negative Range | Overlap |
|---|---:|---:|---|
| state_score | 0.4874..0.9262 | 0.8936..0.9023 | True |
| path_connectivity | 0.5..1.0 | 1.0..1.0 | True |
| reachable_doc_ratio | 0.5..1.0 | 1.0..1.0 | True |
| query_reachability | 1.0..1.0 | 1.0..1.0 | True |
| query_coverage | 0.25..1.0 | 1.0..1.0 | True |
| support_mean | 0.4191..0.9228 | 0.8678..0.871 | True |
| suffix_base_mean | 0.1388..0.4142 | 0.0124..0.1232 | False |
| max_local_structure | 1.0..1.0 | 1.0..1.0 | True |
| max_local_novelty | 0.9706..1.0 | 0.9286..0.9706 | True |
| max_local_closure | 0.5957..0.8541 | 0.6813..0.6965 | True |
| avg_local_structure | 0.5..0.9481 | 0.9748..1.0 | False |
| avg_local_novelty | 0.8853..0.9643 | 0.9117..0.952 | True |
| avg_local_closure | 0.2979..0.7859 | 0.6724..0.6759 | True |

## Positive Queries

### In what year was the group who performed Attics To Eden formed?

- Delta F1: 1.0
- Answers: `Not mentioned.` -> `2005.`
- Gate: `offrank_bridge_signal_detected`
- Saturated state: `False`
- Best state metrics: score=0.5563, path=0.5, reachable=0.5, qreach=1.0, qcov=1.0, support=0.4191, suffix_base=0.1388
- Selected titles: ['Attics to Eden', 'Peter Green Splinter Group (album)', 'Attic Entertainment Software', 'Friday Night Lights (Attic Lights album)', 'Madina Lake']
- Beam steps:
  - step 4: pool=3, structure=0.0, novelty=1.0, closure=0.0, selection=0.0, state=0.2582, rank=2
  - step 5: pool=6, structure=1.0, novelty=0.9167, closure=0.5957, selection=1.0, state=0.5563, rank=1

### What is the position of the 1st governor general of India?

- Delta F1: 0.8
- Answers: `Governor of the Presidency of Fort William (Bengal).` -> `Governor-General of India.`
- Gate: `offrank_bridge_signal_detected`
- Saturated state: `True`
- Best state metrics: score=0.9262, path=1.0, reachable=1.0, qreach=1.0, qcov=1.0, support=0.9228, suffix_base=0.4142
- Selected titles: ['Governor-General of India', 'Governor-General of India', 'Impeachment of Warren Hastings', 'Nawabs of Bengal and Murshidabad', 'Warren Hastings']
- Beam steps:
  - step 4: pool=12, structure=1.0, novelty=0.9706, closure=0.8541, selection=1.0, state=0.9223, rank=1
  - step 5: pool=4, structure=0.8961, novelty=0.8, closure=0.7178, selection=0.8961, state=0.9262, rank=1

### Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?

- Delta F1: 0.8
- Answers: `Gulf of Mexico.` -> `Mississippi River.`
- Gate: `offrank_bridge_signal_detected`
- Saturated state: `False`
- Best state metrics: score=0.4874, path=0.5, reachable=0.5, qreach=1.0, qcov=0.25, support=0.4403, suffix_base=0.1835
- Selected titles: ['Southeast Library', 'Colorado River (Texas)', 'Gulf of Mexico', 'Oklahoma City', 'Riverside Plaza']
- Beam steps:
  - step 4: pool=4, structure=0.0, novelty=1.0, closure=0.0, selection=0.0, state=0.1856, rank=2
  - step 5: pool=29, structure=1.0, novelty=0.9286, closure=0.7013, selection=1.0, state=0.4874, rank=1


## Negative Queries

### When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?

- Delta F1: -1.0
- Answers: `June 1982.` -> `Not mentioned.`
- Gate: `offrank_bridge_signal_detected`
- Saturated state: `True`
- Best state metrics: score=0.8936, path=1.0, reachable=1.0, qreach=1.0, qcov=1.0, support=0.871, suffix_base=0.1232
- Selected titles: ['Lionel Messi', 'FC Barcelona', 'FC Barcelona', 'List of Spanish football champions', 'List of UEFA club competition winners']
- Beam steps:
  - step 4: pool=26, structure=0.9496, novelty=0.9286, closure=0.6705, selection=0.9496, state=0.8943, rank=1
  - step 5: pool=57, structure=1.0, novelty=0.8947, closure=0.6813, selection=1.0, state=0.8936, rank=1

### How did did the people fare during the reign of the abolisher of sati partha in India?

- Delta F1: -0.1333
- Answers: `The practice of sati was abolished, ending its prevalence.` -> `Freed from forced immolation.`
- Gate: `offrank_bridge_signal_detected`
- Saturated state: `True`
- Best state metrics: score=0.9023, path=1.0, reachable=1.0, qreach=1.0, qcov=1.0, support=0.8678, suffix_base=0.0124
- Selected titles: ['Sati (practice)', 'Bengal Sati Regulation, 1829', 'Sati (film)', 'Nawabs of Bengal and Murshidabad', 'Warren Hastings']
- Beam steps:
  - step 4: pool=9, structure=1.0, novelty=0.9706, closure=0.6965, selection=1.0, state=0.9077, rank=1
  - step 5: pool=54, structure=1.0, novelty=0.9333, closure=0.6483, selection=1.0, state=0.9023, rank=1
