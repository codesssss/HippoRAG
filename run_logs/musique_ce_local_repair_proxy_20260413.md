# CE Local Repair Proxy Analysis (musique)

- baseline report: `outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json`
- repair report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_local_repair_qatopk5_20260413smoke.json`
- swap report: `run_logs/musique_swap_value_smoke_20260411.json`
- qa_top_k: `5`
- aligned queries: `100`

## Proxy Sanity

- queries with positive req swap: `18`
- positive req swap rate: `18.0`
- mean missing anchor count: `0.52`
- mean scaffold component count: `65.73`
- queries with positive connector gain: `16`
- queries with positive anchor gain: `4`
- queries with positive both: `2`
- anchor-only rate: `2.0`
- connector-only rate: `14.0`

## Repair Effect

- repair applied query rate: `18.0`
- applied swap positive EM rate: `5.5556`
- applied swap positive F1 rate: `16.6667`
- applied swap avg CE drop vs scaffold: `3.2454`
- applied swap mean ΔEM / ΔF1: `0.0556` / `0.0285`

## Swap Overlap

- oracle positive queries: `14`
- proxy positive and oracle positive: `3`
- overlap rate: `3.0`

## Top Repair Cases

- question: `Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?`
  repair: `Modern history` -> replace `Korean War` (connector `2`, anchor `0`, CE drop `5.6719`)
  delta EM/F1: `1.0` / `1.0`
- question: `How did did the people fare during the reign of the abolisher of sati partha in India?`
  repair: `Delhi` -> replace `Indian Rebellion of 1857` (connector `1`, anchor `0`, CE drop `1.0391`)
  delta EM/F1: `0.0` / `0.1407`
- question: `What is an example of a railroad line in the country first to invade Manchuria?`
  repair: `Allies of World War II` -> replace `Pacific War` (connector `10`, anchor `0`, CE drop `-0.2891`)
  delta EM/F1: `0.0` / `0.0385`
- question: `When did the person chosen to be president of the confederacy end his fight in the Mexican-American war?`
  repair: `Military leadership in the American Civil War` -> replace `Jefferson Davis` (connector `1`, anchor `0`, CE drop `1.1768`)
  delta EM/F1: `0.0` / `0.0`
- question: `How many times did the plague occur in the birth place of Concerto in C Major Op 3 6's composer?`
  repair: `Giuseppe Demachi` -> replace `Black Death` (connector `1`, anchor `0`, CE drop `2.0742`)
  delta EM/F1: `0.0` / `0.0`
- question: `What month did the Tripartite discussions begin between Britain, France, and the country where, despite being headquartered in the nation called the nobilities commonwealth, the top-ranking Warsaw Pact operatives originated?`
  repair: `Cold War` -> replace `Warsaw Pact` (connector `6`, anchor `0`, CE drop `2.3457`)
  delta EM/F1: `0.0` / `0.0`
- question: `How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?`
  repair: `Estonia` -> replace `British Empire` (connector `1`, anchor `0`, CE drop `2.5859`)
  delta EM/F1: `0.0` / `0.0`
- question: `Who was second pick in the 1999 draft of the league that has a competition where they give out the MLB MVP award after it?`
  repair: `Houston Astros` -> replace `NBA high school draftees` (connector `10`, anchor `1`, CE drop `2.6035`)
  delta EM/F1: `0.0` / `0.0`
- question: `When did Nissan, the Acura Legend maker and the Scion Fuse manufacturer open US assembly plants?`
  repair: `IPod` -> replace `Nissan Rogue` (connector `6`, anchor `0`, CE drop `2.6895`)
  delta EM/F1: `0.0` / `0.0`
- question: `what is meaning of the word that is a majority religion of the area that became India when the country origin of Mizraab was created in Arabic dictionary?`
  repair: `History of science` -> replace `Hindus` (connector `0`, anchor `1`, CE drop `2.7363`)
  delta EM/F1: `0.0` / `0.0`
