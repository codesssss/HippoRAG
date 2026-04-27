# NREV Day-0 Audit

## Executive Summary

- Closed-book `2/100` is not reliable as a scientific claim: a short-answer no-context prompt improves the first-20 sample, so the original stratification is prompt/parsing sensitive.
- The earlier `51/100` figure was strongest-null dominance, not `l_minus > l_plus`; the actual stored raw-component count is `39/100`.
- `T_minus` had a real implementation defect: matched replacement allowed same-title duplicates, so several "replacement" perturbations preserved the evidence cell instead of destroying it.
- Toy likelihood magnitudes and token-prefix stability look sane; no obvious prompt-logprob span bug was found.
- After fixing same-title replacement and rerunning first-30, full NREV improves to AUC `0.6644`, which is marginal, not alive. It does not reach the `>=0.70` audit keep-alive threshold and remains below `l_plus` alone.

## Audit 1: Closed-Book Sanity

- `own_knowledge`: correct@F1>=0.5 `4/20`, mean F1 `0.1855`, abstain-like `0`
- `direct_short`: correct@F1>=0.5 `7/20`, mean F1 `0.3150`, abstain-like `0`

| # | Gold | Own-Knowledge Parsed | F1 | Direct Parsed | F1 | Raw Own-Knowledge |
|---:|---|---|---:|---|---:|---|
| 0 | 20 March 851 | 922. | 0.000 | Lothair II's mother, Theodora of France, died in 922. | 0.000 | <think>  </think>  Lothair II, also known as Lothair of France, was the son of Hugh Capet, the founder of the Capetia... |
| 1 | Phoolwari | Aas Ka Panchhi. | 0.000 | Aas Ka Panchhi | 0.000 | <think>  </think>  To determine which film was released first between *Aas Ka Panchhi* and *Phoolwari*, we need to ch... |
| 2 | Port of Spain | Chicago, Illinois. | 0.000 | Atlanta, Georgia | 0.000 | <think>  </think>  The song "Changed It" is performed by the American rapper and singer-songwriter Lil Durk. Lil Durk... |
| 3 | no | No. | 1.000 | Yes. | 0.000 | <think>  </think>  Marufabad is a village in Iran, while Nasamkhrali is a village in Georgia. Therefore, they are not... |
| 4 | God'S Gift To Women | To determine which film has the director who is older, we need to i... | 0.143 | God's Gift to Women | 1.000 | <think>  </think>  To determine which film has the director who is older, we need to identify the directors of both f... |
| 5 | The Heart Of Doreon | José Luis García Sánchez was born on April 2 | 0.000 | The Heart Of Doreon | 1.000 | <think>  </think>  To determine which film's director was born first, we need to identify the directors of "El Tonto"... |
| 6 | Andy Summers | Aivar Kuusmaa. | 0.000 | Aivar Kuusmaa | 0.000 | <think>  </think>  Aivar Kuusmaa was born on April 16, 1946, while Andy Summers was born on September 22, 1947. There... |
| 7 | Ailéan mac Ruaidhrí | Ruaidhrí mac Tairrdelbach Ua Conchobair. | 0.500 | Ruaidhrí mac Tairrdelbach Ua Conchobair | 0.500 | <think>  </think>  Ragnall Mac Ruaidhrí was a 13th-century Scottish nobleman. His paternal grandfather was Ruaidhrí m... |
| 8 | yes | Since both directors are from France, | 0.000 | No. | 0.000 | <think>  </think>  To answer the question, we need to identify the countries of origin for the directors of *Intervie... |
| 9 | Chinese | American. | 0.000 | American | 0.000 | <think>  </think>  The director of the film *Blood Street* is John McNaughton. He is an American filmmaker.   Answer:... |
| 10 | Mexico City | New York City. | 0.400 | Mexico City, Mexico | 0.800 | <think>  </think>  The director of the film *Gaby: A True Story* is James Toback. He was born in New York City.   Ans... |
| 11 | yes | Yes. | 1.000 | Yes. | 1.000 | <think>  </think>  Vasilyevsky Island and Preobrazheniya Island are both located in Saint Petersburg, Russia. Therefo... |
| 12 | America | American. | 0.000 | American | 0.000 | <think>  </think>  The song "When The Stars Go Blue" is performed by the American country music artist Lee Greenwood.... |
| 13 | Dean Miller | The song "Me and Bobby McGee" was performed by Janis Joplin. Janis ... | 0.000 | Kris Kristofferson | 0.000 | <think>  </think>  The song "Me and Bobby McGee" was performed by Janis Joplin. Janis Joplin's child was a daughter n... |
| 14 | Delft | The Hague. | 0.000 | Netherlands | 0.000 | <think>  </think>  Maurice, Prince of Orange, was the son of William the Silent, also known as William of Orange. Wil... |
| 15 | Polish-Lithuanian Commonwealth | Poland. | 0.000 | Poland | 0.000 | <think>  </think>  Aleksander Koniecpolski was a Polish-Lithuanian nobleman and military commander. His father, Jan K... |
| 16 | 10 August 1960 | April 13, 2022. | 0.000 | April 13, 2023 | 0.000 | <think>  </think>  The director of the film *Madame La Presidente* is Jean-Pierre Mocky. He died on April 13, 2022.  ... |
| 17 | no | Since Buechler and Howling are from | 0.000 | No. | 1.000 | <think>  </think>  To answer the question, we need to identify the nationalities of the directors of *Wrong Turn 5: B... |
| 18 | You Can No Longer Remain Silent | *You Can No Longer Remain Silent* (1971) was directed by **Peter Gl... | 0.667 | You Can No Longer Remain Silent | 1.000 | <think>  </think>  To determine which film's director died first, we need to identify the directors of *The Goose Wom... |
| 19 | Montreal, Quebec | France. | 0.000 | France | 0.000 | <think>  </think>  The director of the film *The Private Life of Cinema* is Jean-Luc Godard. He was born in France.  ... |

## Audit 2: Likelihood Scale

- gold `l_minus > l_plus`: `39`
- gold `l_minus < l_plus`: `61`
- mean gold `l_minus - l_plus`: `-0.0877`

| # | Gold | l_plus | l_minus | l0 | l_minus-l_plus | Wrong | wrong l_plus | wrong l_minus |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 70 | yes | -12.5421 | -10.3714 | -8.4593 | 2.1708 | No | -0.1862 | -0.4316 |
| 95 | American | -8.0778 | -7.1875 | -6.8066 | 0.8903 | DJ Clue | -1.4221 | -1.8400 |
| 20 | Oxford | -5.2412 | -4.5061 | -6.1343 | 0.7351 | 1792 | -1.8593 | -1.9219 |
| 6 | Andy Summers | -1.6359 | -0.9636 | -0.8579 | 0.6723 | Andy Summers born 1942 | -2.5254 | -2.3466 |
| 76 | Pirates Of The Sky | -0.9569 | -0.2942 | -0.0437 | 0.6627 | Pirates | -1.1834 | -1.8968 |
| 24 | Dying God | -0.7100 | -0.1635 | -0.4077 | 0.5465 | Dying God (2008) | -0.5647 | -0.5982 |
| 79 | Bruce Coulter | -0.7510 | -0.2582 | -0.2008 | 0.4928 | Marjatta Raita was born after Bruce Coulter | -1.1325 | -1.0339 |
| 27 | Yale College | -2.1524 | -1.8063 | -2.5626 | 0.3462 | law | -13.6158 | -12.7500 |
| 84 | yes | -6.7024 | -6.4011 | -7.9609 | 0.3013 | Eustace William Ferguson | -0.9282 | -0.7780 |
| 94 | Athenian | -1.9171 | -1.6173 | -2.3644 | 0.2999 | Croatian | -14.1265 | -14.3323 |

## Audit 3: T-minus Content

| Pair | Removed kind | Count | Mean l_minus component score |
|---|---|---:|---:|
| gold | `critical` | 492 | -2.2724 |
| gold | `padding` | 508 | -2.3437 |
| wrong | `answer_containing` | 496 | -5.0217 |
| wrong | `other` | 504 | -3.3232 |

### Sample T-minus Variants

- query `70`, gold `yes`, l_plus `-12.5421`, l_minus `-10.3714`
  S_gold: [0] Christopher Newton (criminal) (support=True, answer=False); [1] Frances M. Vega (support=True, answer=False); [2] Benjamin Kendrick Pierce (support=False, answer=False); [3] Christopher Newton (support=False, answer=False); [4] Dugès (support=False, answer=False)
  - `delete_0` removed `Christopher Newton (criminal)` (support=True, answer=False)
  - `replace_0` removed `Christopher Newton (criminal)` (support=True, answer=False, repl=Christopher Newton (criminal))
  - `delete_1` removed `Frances M. Vega` (support=True, answer=False)
  - `replace_1` removed `Frances M. Vega` (support=True, answer=False, repl=Frances M. Vega)
- query `95`, gold `American`, l_plus `-8.0778`, l_minus `-7.1875`
  S_gold: [0] I Like Control (support=True, answer=False); [1] DJ Clue (support=True, answer=True); [2] Where Was I (support=False, answer=False); [3] When I Was Young (support=False, answer=False); [4] Shape of My Heart (support=False, answer=False)
  - `delete_0` removed `I Like Control` (support=True, answer=False)
  - `replace_0` removed `I Like Control` (support=True, answer=False, repl=I Like Control)
  - `delete_1` removed `DJ Clue` (support=True, answer=True)
  - `replace_1` removed `DJ Clue` (support=True, answer=True, repl=DJ Clue)
- query `20`, gold `Oxford`, l_plus `-5.2412`, l_minus `-4.5061`
  S_gold: [0] Coulson Wallop (support=True, answer=False); [1] John Wallop, 2nd Earl of Portsmouth (support=True, answer=True); [2] John Stuart, Lord Mount Stuart (support=False, answer=False); [3] George Rodney, 2nd Baron Rodney (support=False, answer=False); [4] Robert Vere Buxton (support=False, answer=True)
  - `delete_0` removed `Coulson Wallop` (support=True, answer=False)
  - `replace_0` removed `Coulson Wallop` (support=True, answer=False, repl=Coulson Wallop)
  - `delete_1` removed `John Wallop, 2nd Earl of Portsmouth` (support=True, answer=True)
  - `replace_1` removed `John Wallop, 2nd Earl of Portsmouth` (support=True, answer=True, repl=John Wallop, 2nd Earl of Portsmouth)
- query `6`, gold `Andy Summers`, l_plus `-1.6359`, l_minus `-0.9636`
  S_gold: [0] Andy Summers (support=True, answer=True); [1] Aivar Kuusmaa (support=True, answer=False); [2] Dugès (support=False, answer=False); [3] Jonny Greenwood (support=False, answer=False); [4] Where Was I (support=False, answer=False)
  - `delete_0` removed `Andy Summers` (support=True, answer=True)
  - `replace_0` removed `Andy Summers` (support=True, answer=True, repl=Andy Summers)
  - `delete_1` removed `Aivar Kuusmaa` (support=True, answer=False)
  - `replace_1` removed `Aivar Kuusmaa` (support=True, answer=False, repl=Aivar Kuusmaa)
- query `76`, gold `Pirates Of The Sky`, l_plus `-0.9569`, l_minus `-0.2942`
  S_gold: [0] Pirates of the Sky (support=True, answer=True); [1] A Romance of Seville (support=True, answer=False); [2] The Trail of the Lonesome Pine (1923 film) (support=False, answer=False); [3] Our Emden (support=False, answer=False); [4] Romance on the Range (support=False, answer=False)
  - `delete_0` removed `Pirates of the Sky` (support=True, answer=True)
  - `replace_0` removed `Pirates of the Sky` (support=True, answer=True, repl=Pirates of the Sky)
  - `delete_1` removed `A Romance of Seville` (support=True, answer=False)
  - `replace_1` removed `A Romance of Seville` (support=True, answer=False, repl=A Romance of Seville)

## Audit 4: Prompt / Token / Toy Checks

### Toy Likelihood

| Question | Answer | evidence score | closed score | prefix stable |
|---|---|---:|---:|---|
| What is the capital of the United Kingdom? | London | -0.3514 | -1.0575 | True / True |
| Who wrote Hamlet? | William Shakespeare | -0.3990 | -0.1477 | True / True |
| What is the largest planet in the Solar System? | Jupiter | -0.0960 | -0.9925 | True / True |

### Per-Token Breakdown Example

- query: `70`
- answer: `yes`
- prefix stable: `True`
- score: `-11.3943`

| Token | Logprob |
|---|---:|
| ` yes` | -11.3943 |

### Prompt Format Samples

- query `70` `delete_0`: plus chars/tokens `3060/673`, minus chars/tokens `1038/266`
  plus tail: `istopher Newton( born 11 June 1936) is a Canadian director and actor and served as artistic director of the Shaw Festival from 1980- 2002.  Wikipedia Title: Dugès Several people share the surname Dugès:  Question: Are...`
  minus tail: `istopher Newton( born 11 June 1936) is a Canadian director and actor and served as artistic director of the Shaw Festival from 1980- 2002.  Wikipedia Title: Dugès Several people share the surname Dugès:  Question: Are...`
- query `95` `delete_0`: plus chars/tokens `859/231`, minus chars/tokens `444/117`
  plus tail: `cer, and radio personality.  Wikipedia Title: Where Was I " Where Was I?" may refer to:  Wikipedia Title: When I Was Young When I Was Young may refer to:  Wikipedia Title: Shape of My Heart Shape of My Heart may refer...`
  minus tail: `cer, and radio personality.  Wikipedia Title: Where Was I " Where Was I?" may refer to:  Wikipedia Title: When I Was Young When I Was Young may refer to:  Wikipedia Title: Shape of My Heart Shape of My Heart may refer...`
- query `20` `delete_0`: plus chars/tokens `7498/2046`, minus chars/tokens `5628/1564`
  plus tail: `ters of the book. From 1945, he was deputy chairman of Martins Bank and chairman of its London board. He married Irene Marguerite Pix, widow of Sir Richard Levinge, 10th baronet, in 1916. He had no issue. He died in I...`
  minus tail: `ters of the book. From 1945, he was deputy chairman of Martins Bank and chairman of its London board. He married Irene Marguerite Pix, widow of Sir Richard Levinge, 10th baronet, in 1916. He had no issue. He died in I...`
- query `6` `delete_0`: plus chars/tokens `3522/851`, minus chars/tokens `3071/745`
  plus tail: `lude two collaborations with director Lynne Ramsay. He has collaborated several times with the Israeli composer Shye Ben Tzur, including on the 2015 album "Junun".  Wikipedia Title: Where Was I " Where Was I?" may ref...`
  minus tail: `lude two collaborations with director Lynne Ramsay. He has collaborated several times with the Israeli composer Shye Ben Tzur, including on the 2015 album "Junun".  Wikipedia Title: Where Was I " Where Was I?" may ref...`
- query `76` `delete_0`: plus chars/tokens `1710/442`, minus chars/tokens `1364/352`
  plus tail: `en". It was made at the Emelka Studios in Munich. The film's sets were designed by the art directors Botho Hoefer and Ludwig Reiber.  Wikipedia Title: Romance on the Range Romance on the Range may refer to:  Question:...`
  minus tail: `en". It was made at the Emelka Studios in Munich. The film's sets were designed by the art directors Botho Hoefer and Ludwig Reiber.  Wikipedia Title: Romance on the Range Romance on the Range may refer to:  Question:...`

## Post-hoc Support-only T-minus Recompute

- limit: `30`

| Score | AUC | Paired Win Rate |
|---|---:|---:|
| `l_plus` | 0.6656 | 0.6667 |
| `rev` | 0.4767 | 0.4000 |
| `nrev_no_alt` | 0.4844 | 0.4667 |
| `nrev_no_l0` | 0.6433 | 0.5667 |
| `nrev_full` | 0.6444 | 0.6000 |

## Fixed Same-title Replacement Rerun

Matched replacement was fixed to exclude any candidate whose title already appears in the current evidence set. This prevents cases such as replacing `Pirates of the Sky` with another `Pirates of the Sky`.

First-30 rerun:

- report: `reports/nrev/day0_audit_rerun30/day0_sanity.md`
- closed-book correct/wrong: `8 / 22`
- decision under registered gates: `MARGINAL_REDESIGN_T_MINUS_ONCE`

| Score | AUC | 95% CI | Paired Win Rate |
|---|---:|---:|---:|
| `l_plus` | 0.6656 | [0.5400, 0.7944] | 0.6667 |
| `rev` | 0.5278 | [0.4078, 0.6356] | 0.4667 |
| `nrev_no_alt` | 0.5222 | [0.3867, 0.6356] | 0.5000 |
| `nrev_no_l0` | 0.6600 | [0.4911, 0.8167] | 0.5667 |
| `nrev_full` | 0.6644 | [0.4744, 0.8244] | 0.6000 |

Interpretation:

- The same-title replacement bug materially hurt REV/NREV, but fixing it does not make NREV pass.
- Full NREV remains below the audit keep-alive threshold `0.70` on first-30 and does not beat `l_plus`.
- This supports "one possible T-minus redesign only" rather than full NREV implementation.

## Audit Decision

NREV_AUDIT_MARGINAL_TMINUS_REDESIGN_ONLY
