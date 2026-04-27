# BSGS Oracle Diagnostic Ablation

This diagnostic tests whether oracle pool exposure, oracle binding rewrites,
or oracle likelihoods can rescue the node-marginal BSGS operator after the
main Week-1 route failed on MuSiQue.

## Config
- dataset: `musique`
- n: `200`
- top_k: `5`
- absorbing_enabled: `False`

## Variant Summary

| Variant | Title Recall | Paragraph-Idx Recall | Full/Partial/None Title | Full/Partial/None Idx | Entropy | Dup Title |
|---|---:|---:|---:|---:|---:|---:|
| `base` | `0.2050` | `0.1608` | `9/74/117` | `2/71/127` | `2.5407` | `0.2930` |
| `oracle_pool` | `0.2454` | `0.2113` | `11/86/103` | `5/86/109` | `2.6282` | `0.3360` |
| `oracle_pool_binding` | `0.4958` | `0.4758` | `35/140/25` | `32/141/27` | `2.5464` | `0.3480` |
| `oracle_pool_binding_likelihood` | `0.5854` | `0.5746` | `47/153/0` | `43/157/0` | `3.3023` | `0.7170` |

## Interpretation Guide
- `oracle_pool` exposes complete gold support paragraphs as candidate propositions.
- `oracle_pool_binding` additionally rewrites `#1/#2` with prior gold slot answers.
- `oracle_pool_binding_likelihood` additionally assigns oracle high/low likelihood by slot support paragraph.
- If the last variant remains weak, the failure is not slot quality, binding, or verifier calibration; it is the graph/state transition.

## Debug Examples

### base
- `2hop__13548_13529` title_recall=`1.0000` idx_recall=`0.5000` selected=['FC Barcelona', 'List of international goals scored by Lionel Messi', 'FC Barcelona', 'Iker Muniain', 'List of La Liga top scorers'] gold=['FC Barcelona', 'FC Barcelona']
- `3hop1__9285_5188_23307` title_recall=`0.3333` idx_recall=`0.0000` selected=['Events leading to the attack on Pearl Harbor', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact'] gold=['Szlachta', 'Molotov–Ribbentrop Pact', 'Warsaw Pact']
- `2hop__766973_770570` title_recall=`0.0000` idx_recall=`0.0000` selected=['Eritrea', 'Italian Eritrea', "Sant Martí d'Empúries", "Sant Martí d'Empúries", 'Långe Erik'] gold=['Montebello, New York', 'Erik Hort']
- `2hop__170823_120171` title_recall=`0.0000` idx_recall=`0.0000` selected=['Blast Corps', 'Blast Corps', 'Blast Corps', 'Blast Corps', 'List of Little House on the Prairie books'] gold=['Acornsoft', 'Labyrinth (1984 video game)']
- `2hop__511454_120259` title_recall=`0.0000` idx_recall=`0.0000` selected=['Mary, mother of Jesus', '1909 World Figure Skating Championships', 'Our Lady of Guadaloupe Church', 'Capital punishment in the United Kingdom', 'Capital punishment in the United Kingdom'] gold=['Mercia', 'Spalding Priory']

### oracle_pool
- `2hop__13548_13529` title_recall=`1.0000` idx_recall=`1.0000` selected=['FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona'] gold=['FC Barcelona', 'FC Barcelona']
- `3hop1__9285_5188_23307` title_recall=`0.6667` idx_recall=`0.6667` selected=['Molotov–Ribbentrop Pact', 'Molotov–Ribbentrop Pact', 'Events leading to the attack on Pearl Harbor', 'Warsaw Pact', 'Warsaw Pact'] gold=['Szlachta', 'Molotov–Ribbentrop Pact', 'Warsaw Pact']
- `2hop__766973_770570` title_recall=`0.0000` idx_recall=`0.0000` selected=['Eritrea', 'Italian Eritrea', 'Långe Erik', "Sant Martí d'Empúries", "Sant Martí d'Empúries"] gold=['Montebello, New York', 'Erik Hort']
- `2hop__170823_120171` title_recall=`0.0000` idx_recall=`0.0000` selected=['Blast Corps', 'Blast Corps', 'Blast Corps', 'Blast Corps', 'List of Little House on the Prairie books'] gold=['Acornsoft', 'Labyrinth (1984 video game)']
- `2hop__511454_120259` title_recall=`0.0000` idx_recall=`0.0000` selected=['Mary, mother of Jesus', '1909 World Figure Skating Championships', 'Our Lady of Guadaloupe Church', 'Capital punishment in the United Kingdom', 'Capital punishment in the United Kingdom'] gold=['Mercia', 'Spalding Priory']

### oracle_pool_binding
- `2hop__13548_13529` title_recall=`1.0000` idx_recall=`1.0000` selected=['FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'List of international goals scored by Lionel Messi'] gold=['FC Barcelona', 'FC Barcelona']
- `3hop1__9285_5188_23307` title_recall=`1.0000` idx_recall=`1.0000` selected=['Molotov–Ribbentrop Pact', 'Warsaw Pact', 'Molotov–Ribbentrop Pact', 'Events leading to the attack on Pearl Harbor', 'Szlachta'] gold=['Szlachta', 'Molotov–Ribbentrop Pact', 'Warsaw Pact']
- `2hop__766973_770570` title_recall=`0.0000` idx_recall=`0.0000` selected=['Eritrea', 'Italian Eritrea', 'Långe Erik', "Sant Martí d'Empúries", "Sant Martí d'Empúries"] gold=['Montebello, New York', 'Erik Hort']
- `2hop__170823_120171` title_recall=`0.5000` idx_recall=`0.5000` selected=['Labyrinth (1984 video game)', 'Labyrinth (1984 video game)', 'Labyrinth (1984 video game)', 'Blast Corps', 'Blast Corps'] gold=['Acornsoft', 'Labyrinth (1984 video game)']
- `2hop__511454_120259` title_recall=`0.5000` idx_recall=`0.5000` selected=['Mary, mother of Jesus', '1909 World Figure Skating Championships', 'Our Lady of Guadaloupe Church', 'Mercia', 'Mercia'] gold=['Mercia', 'Spalding Priory']

### oracle_pool_binding_likelihood
- `2hop__13548_13529` title_recall=`1.0000` idx_recall=`0.5000` selected=['FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona'] gold=['FC Barcelona', 'FC Barcelona']
- `3hop1__9285_5188_23307` title_recall=`0.3333` idx_recall=`0.3333` selected=['Szlachta', 'Szlachta', 'Szlachta', 'Szlachta', 'Szlachta'] gold=['Szlachta', 'Molotov–Ribbentrop Pact', 'Warsaw Pact']
- `2hop__766973_770570` title_recall=`1.0000` idx_recall=`1.0000` selected=['Erik Hort', 'Erik Hort', 'Montebello, New York', 'Montebello, New York', 'Montebello, New York'] gold=['Montebello, New York', 'Erik Hort']
- `2hop__170823_120171` title_recall=`0.5000` idx_recall=`0.5000` selected=['Acornsoft', 'Acornsoft', 'Acornsoft', 'Acornsoft', 'Acornsoft'] gold=['Acornsoft', 'Labyrinth (1984 video game)']
- `2hop__511454_120259` title_recall=`1.0000` idx_recall=`1.0000` selected=['Mercia', 'Mercia', 'Spalding Priory', 'Spalding Priory', 'Spalding Priory'] gold=['Mercia', 'Spalding Priory']
