# CE Local Repair Proxy Analysis (hotpotqa)

- baseline report: `outputs_step0_general_hotpotqa/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260407.json`
- repair report: `outputs_step0_general_hotpotqa/eval_reports/width_match_bridge_append_plus_ce_local_repair_qatopk5_20260413smoke.json`
- swap report: `None`
- qa_top_k: `5`
- aligned queries: `100`

## Proxy Sanity

- queries with positive req swap: `8`
- positive req swap rate: `8.0`
- mean missing anchor count: `0.86`
- mean scaffold component count: `57.69`
- queries with positive connector gain: `6`
- queries with positive anchor gain: `2`
- queries with positive both: `0`
- anchor-only rate: `2.0`
- connector-only rate: `6.0`

## Repair Effect

- repair applied query rate: `8.0`
- applied swap positive EM rate: `0.0`
- applied swap positive F1 rate: `0.0`
- applied swap avg CE drop vs scaffold: `4.7931`
- applied swap mean ΔEM / ΔF1: `0.0` / `0.0`

## Swap Overlap

- oracle positive queries: `0`
- proxy positive and oracle positive: `0`
- overlap rate: `0.0`

## Top Repair Cases

- question: `The fourth episode The Simpsons' seventh season has what kind of theme?`
  repair: `The Food Wife` -> replace `The Itchy &amp; Scratchy &amp; Poochie Show` (connector `1`, anchor `0`, CE drop `1.3945`)
  delta EM/F1: `0.0` / `0.0`
- question: `What 1944 Bollywood film was the mother of Bollywood actor Govinda in?`
  repair: `Adnan Siddiqui` -> replace `Naach Govinda Naach` (connector `0`, anchor `1`, CE drop `2.3789`)
  delta EM/F1: `0.0` / `0.0`
- question: `What is the current home arena of the NHL team Chris Summers plays for?`
  repair: `1966–67 NHL season` -> replace `San Antonio Rampage` (connector `1`, anchor `0`, CE drop `3.3164`)
  delta EM/F1: `0.0` / `0.0`
- question: `Of four Harry S. Truman Supreme Court candidates, who was the 53rd United States Secretary of the Treasury and the 13th Chief Justice of the United States?`
  repair: `Supreme Court of Nepal` -> replace `Taft Court` (connector `1`, anchor `0`, CE drop `3.457`)
  delta EM/F1: `0.0` / `0.0`
- question: `What movie did Chris Duesterdiek work on that was directed by Seth Rogen and Evan Goldberg?`
  repair: `Bigfoot (TV series)` -> replace `Pineapple Express (film)` (connector `1`, anchor `0`, CE drop `4.9863`)
  delta EM/F1: `0.0` / `0.0`
- question: `This Experts Network sports analysts was inducted into the Pro Football Hall of Fame in 2000 and played in the NFL for how many seasons?`
  repair: `1982 Detroit Lions season` -> replace `LaDainian Tomlinson` (connector `0`, anchor `1`, CE drop `5.4429`)
  delta EM/F1: `0.0` / `0.0`
- question: `Where is the ice hockey team based that Zdeno Chára currently serving as captain of?`
  repair: `1966–67 NHL season` -> replace `Al Iafrate` (connector `3`, anchor `0`, CE drop `8.6211`)
  delta EM/F1: `0.0` / `0.0`
- question: `In what year was the Golden State NBA player, who was part of the Cavaliers-Warriors rivalry, named NBA Finals Most Valuable Player?`
  repair: `2003 NBA draft` -> replace `Kevin Durant` (connector `1`, anchor `0`, CE drop `8.748`)
  delta EM/F1: `0.0` / `0.0`
