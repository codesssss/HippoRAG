# BSGS Oracle-Slot Mechanism Analysis

- Status: `completed`
- Examples: `1000`

## Overall
- supporting_paragraph_recall: `0.2268`
- avg_belief_entropy: `2.4719`
- avg_duplicate_title_rate: `0.2654`

## Support Coverage Bins

| Bin | Count | Fraction | Avg Recall | Avg Entropy |
|---|---:|---:|---:|---:|
| full_support_covered | `54` | `0.0540` | `1.0000` | `2.7557` |
| partial_support_covered | `400` | `0.4000` | `0.4319` | `2.5009` |
| no_support_covered | `546` | `0.5460` | `0.0000` | `2.4226` |

## Failure Samples

- `no_support_covered` `2hop__766973_770570` recall=`0.0` selected=['Eritrea', 'Italian Eritrea', "Sant Martí d'Empúries", "Sant Martí d'Empúries", 'Långe Erik'] gold=['Montebello, New York', 'Erik Hort']
- `no_support_covered` `2hop__170823_120171` recall=`0.0` selected=['Blast Corps', 'Blast Corps', 'Blast Corps', 'Blast Corps', 'List of Little House on the Prairie books'] gold=['Acornsoft', 'Labyrinth (1984 video game)']
- `no_support_covered` `2hop__511454_120259` recall=`0.0` selected=['Mary, mother of Jesus', '1909 World Figure Skating Championships', 'Our Lady of Guadaloupe Church', 'Capital punishment in the United Kingdom', 'Capital punishment in the United Kingdom'] gold=['Mercia', 'Spalding Priory']
- `no_support_covered` `4hop2__71753_648517_70784_79935` recall=`0.0` selected=['Umm Ubays', 'Wollaston Peninsula', 'Wollaston Peninsula', 'Wollaston Peninsula', 'Wollaston Peninsula'] gold=['History of Saudi Arabia', 'Israel', 'Battle of Qurah and Umm al Maradim', 'Geography of Saudi Arabia']
- `no_support_covered` `4hop1__94201_642284_131926_89261` recall=`0.0` selected=['Seama, New Mexico', 'Paguate, New Mexico', 'Belview, Virginia', 'One Tree Hill (season 3)', 'One Tree Hill (season 3)'] gold=['Riverside Plaza', 'Mississippi River', 'Minneapolis', 'Southeast Library']
- `no_support_covered` `4hop1__152562_5274_458768_33633` recall=`0.0` selected=['Metropolis International', 'Terry Nelson (musician)', 'Terry Nelson (musician)', 'Terry Nelson (musician)', 'Terry Nelson (musician)'] gold=['Sony Music', 'Vilaiyaadu Mankatha', 'The Right Stuff Records', 'Santa Monica, California']
- `no_support_covered` `3hop1__305282_282081_73772` recall=`0.0` selected=['Giovanni Cifolelli', 'Battle of Abbeville', 'Battle of Preveza', 'Battle of Preveza', 'Fundamental Rights, Directive Principles and Fundamental Duties of India'] gold=['III (Stanton Moore album)', "Flyin' the Koop", 'Battle of New Orleans']
- `no_support_covered` `2hop__619265_45326` recall=`0.0` selected=['Cartoon Wars Part I', 'Glee (season 5)', 'Glee (season 5)', 'List of The Mindy Project episodes', 'List of The Mindy Project episodes'] gold=['The Bag or the Bat', 'List of Ray Donovan episodes']
- `partial_support_covered` `3hop1__9285_5188_23307` recall=`0.3333333333333333` selected=['Events leading to the attack on Pearl Harbor', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact'] gold=['Szlachta', 'Molotov–Ribbentrop Pact', 'Warsaw Pact']
- `partial_support_covered` `2hop__655505_110949` recall=`0.5` selected=['Metsweding District Municipality', 'Dominic Toretto', 'List of The Blacklist characters', 'Till dom ensamma', 'Till dom ensamma'] gold=['Till dom ensamma', 'Mauro Scocco']
