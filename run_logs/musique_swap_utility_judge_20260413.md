# Swap-Utility Judge (musique)

- baseline report: `outputs_step0_general_musique/eval_reports/width_match_baseline_top10_plus_ce_qatopk5_20260409smoke.json`
- candidate report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_qatopk5_20260409smoke.json`
- judge: `gpt-5.4` via `responses`
- judge base_url: `https://api.shenfengwl.fun`
- judge reasoning_effort: `medium`
- replace_bottom_n: `2`

## Job Generation

- aligned queries: `100`
- queries with swap jobs: `69`
- total swap jobs: `328`
- max_queries: `100`
- queries_without_appended_candidates: `31`
- replace_bottom_n: `2`

## Judge Summary

- parse failure count / rate: `13` / `3.9634`
- helpful / neutral / harmful: `11` / `60` / `244`
- helpful / neutral / harmful rates: `3.3537` / `18.2927` / `74.3902`

## Query Gate

- queries with helpful swap: `7` / `69`
- queries with high-confidence helpful swap: `7` / `69`

## Oracle Alignment

- matched jobs: `158`
- helpful precision EM / F1: `20.0` / `20.0`
- helpful recall EM / F1: `6.6667` / `4.1667`
- helpful mean oracle delta EM / F1: `0.0` / `0.1333`
- judge blocks high-CE nonpositive swaps: `14` / `14`

## Top Helpful Swaps

- question: `What city is the star of Sous les pieds des femmes from?`
  candidate: `Claudia Cardinale` replace `Les Bonnes Femmes` (slot `4`, CE rank `2`)
  confidence / type: `98.0` / `bridge_relation`
  reason: `The candidate directly identifies the star of "Sous les pieds des femmes" (Claudia Cardinale) and gives her birthplace as La Goulette, a neighborhood of Tunis, which is the missing bridge needed to answer the city question, while the incumbent is unrelated.`
- question: `When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?`
  candidate: `Argentina v England (1986 FIFA World Cup)` replace `FC Barcelona` (slot `4`, CE rank `12`)
  confidence / type: `88.0` / `entity_grounding`
  reason: `The candidate usefully grounds the likely comparison target as Maradona, which is more relevant to the question than the incumbent’s unrelated Kubala signing detail, while the key signing date for Barcelona is still preserved elsewhere in the set.`
- question: `When did the majority party in the House of Representatives gain control of the body which approves members of the Cabinet?`
  candidate: `Treaty of Versailles` replace `Speaker of the United States House of Representatives` (slot `4`, CE rank `12`)
  confidence / type: `88.0` / `bridge_relation`
  reason: `The candidate adds the missing bridge that the Cabinet-approving body is the Senate and that the Republican Party gained control of it after the 1918 election, while the incumbent only gives generic information about the House Speaker.`
- question: `What was the form of the language Auctor is in, used in the era of the Frankish king who created the Holy Roman Empire, later known as?`
  candidate: `Anno Domini` replace `Holy Roman Empire` (slot `5`, CE rank `13`)
  confidence / type: `86.0` / `bridge_relation`
  reason: `The candidate adds the missing bridge that the relevant "era" associated with Charlemagne was Anno Domini, which is more directly useful for linking Auctor’s Latin to the time period in question than the incumbent’s generic expansion-history sentence.`
- question: `Who played the girlfriend of who plays marty mcfly's daughter in back to the future 2?`
  candidate: `Elisabeth Shue` replace `Back to the Future Part II` (slot `4`, CE rank `7`)
  confidence / type: `84.0` / `entity_grounding`
  reason: `The candidate usefully grounds the actress associated with Back to the Future Part II, which helps connect the existing evidence about Marty McFly’s girlfriend Jennifer Parker to the likely answer, whereas the incumbent is only generic film metadata.`
- question: `When was the last time Darren Carter's team beat the 1894-95 FA Cup winner?`
  candidate: `1982 European Cup Final` replace `Everton F.C.` (slot `4`, CE rank `11`)
  confidence / type: `83.0` / `entity_grounding`
  reason: `The candidate at least introduces Aston Villa—the actual 1894–95 FA Cup winner—while the incumbent Everton page is unrelated to the multi-hop chain needed to connect Darren Carter's team to that cup winner.`
- question: `What was the language Auctor comes from during the era of the man crowned first Holy Roman Emperor later known as?`
  candidate: `Anno Domini` replace `Holy Roman Empire` (slot `4`, CE rank `11`)
  confidence / type: `81.0` / `bridge_relation`
  reason: `The candidate adds a likely missing bridge for the query term “era” by connecting Charlemagne to the Anno Domini era, while the incumbent only gives broad territorial context about Charlemagne that is less directly useful for answering that Auctor comes from Latin.`
