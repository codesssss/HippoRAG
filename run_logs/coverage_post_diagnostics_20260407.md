# Coverage Post-Run Diagnostics

## Headline

- Focus 1: append=0 same-pool coverage vs cross-encoder.
- Focus 2: coverage-vs-CE / coverage-vs-baseline overlap behavior.
- Focus 3: append=3 under coverage versus append=0 under coverage.

## Overall

| Dataset | K | Baseline EM/F1 | append0+CE EM/F1 | append0+Cov EM/F1 | append3+Cov EM/F1 |
|---|---:|---:|---:|---:|---:|
| musique | 5 | 0.27/0.3348 | 0.31/0.3698 | 0.33/0.4007 | 0.28/0.3538 |
| musique | 7 | 0.32/0.3924 | 0.34/0.4034 | 0.33/0.3907 | 0.31/0.3881 |
| musique | 10 | 0.34/0.4235 | 0.39/0.4412 | 0.39/0.4464 | 0.33/0.4084 |
| hotpotqa | 5 | 0.58/0.6967 | 0.62/0.7173 | 0.54/0.6454 | 0.51/0.6179 |
| hotpotqa | 7 | 0.58/0.6953 | 0.60/0.7136 | 0.58/0.6783 | 0.60/0.6956 |
| hotpotqa | 10 | 0.60/0.7295 | 0.59/0.6981 | 0.59/0.6981 | 0.57/0.6715 |
| 2wikimultihopqa | 5 | 0.36/0.4008 | 0.42/0.4553 | 0.39/0.4426 | 0.38/0.4385 |
| 2wikimultihopqa | 7 | 0.42/0.4642 | 0.44/0.5152 | 0.42/0.4740 | 0.43/0.4835 |
| 2wikimultihopqa | 10 | 0.45/0.5118 | 0.46/0.4992 | 0.45/0.4942 | 0.48/0.5203 |

## Paired Gain/Tie/Loss

| Dataset | K | Cov vs CE EM G/T/L | Cov vs CE F1 G/T/L | Avg ΔEM | Avg ΔF1 |
|---|---:|---:|---:|---:|---:|
| musique | 5 | 7/88/5 | 13/80/7 | 0.0200 | 0.0310 |
| musique | 7 | 4/91/5 | 8/81/11 | -0.0100 | -0.0127 |
| musique | 10 | 1/98/1 | 4/92/4 | 0.0000 | 0.0052 |
| hotpotqa | 5 | 1/90/9 | 4/86/10 | -0.0800 | -0.0719 |
| hotpotqa | 7 | 2/94/4 | 3/92/5 | -0.0200 | -0.0352 |
| hotpotqa | 10 | 0/100/0 | 0/100/0 | 0.0000 | 0.0000 |
| 2wikimultihopqa | 5 | 6/85/9 | 9/80/11 | -0.0300 | -0.0127 |
| 2wikimultihopqa | 7 | 4/90/6 | 5/86/9 | -0.0200 | -0.0412 |
| 2wikimultihopqa | 10 | 0/99/1 | 0/99/1 | -0.0100 | -0.0050 |

## Overlap

| Dataset | K | append0 avg overlap CE | append0 avg overlap baseline | append0 changed | append3 avg overlap baseline | append3 avg appended selected |
|---|---:|---:|---:|---:|---:|---:|
| musique | 5 | 3.28 | 3.10 | 100/100 | 2.64 | 1.06 |
| musique | 7 | 5.11 | 5.15 | 100/100 | 4.20 | 1.43 |
| musique | 10 | 5.10 | 10.00 | 99/100 | 8.40 | 1.60 |
| hotpotqa | 5 | 3.56 | 3.30 | 100/100 | 3.10 | 0.36 |
| hotpotqa | 7 | 6.05 | 5.37 | 100/100 | 5.08 | 0.49 |
| hotpotqa | 10 | 10.00 | 10.00 | 100/100 | 9.41 | 0.59 |
| 2wikimultihopqa | 5 | 3.74 | 3.06 | 100/100 | 2.85 | 0.46 |
| 2wikimultihopqa | 7 | 6.26 | 5.10 | 100/100 | 4.81 | 0.56 |
| 2wikimultihopqa | 10 | 9.90 | 10.00 | 100/100 | 9.39 | 0.61 |

## Append Effect

| Dataset | K | append3 vs append0 EM G/T/L | append3 vs append0 F1 G/T/L | queries with appended selected | appended selected and hurt |
|---|---:|---:|---:|---:|---:|
| musique | 5 | 3/89/8 | 4/84/12 | 64 | 12 |
| musique | 7 | 2/94/4 | 7/88/5 | 68 | 5 |
| musique | 10 | 1/92/7 | 5/85/10 | 69 | 10 |
| hotpotqa | 5 | 1/95/4 | 1/95/4 | 30 | 4 |
| hotpotqa | 7 | 3/96/1 | 3/96/1 | 36 | 1 |
| hotpotqa | 10 | 1/96/3 | 1/95/4 | 38 | 4 |
| 2wikimultihopqa | 5 | 1/97/2 | 2/94/4 | 31 | 4 |
| 2wikimultihopqa | 7 | 1/99/0 | 3/95/2 | 37 | 2 |
| 2wikimultihopqa | 10 | 5/93/2 | 6/91/3 | 38 | 3 |

## Representative Failures

### musique K=5

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 1 | 3 | How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died? |
| -1.0000 | 3 | 2 | What is the position of the 1st governor general of India? |
| -1.0000 | 3 | 2 | When did the country the top-ranking Warsaw Pact operatives came from, despite it being headquartered in the country where A Generation is set, agree to a unified Germany inside NATO? |
| -1.0000 | 3 | 2 | When did the person chosen to be president of the confederacy end his fight in the Mexican-American war? |
| -1.0000 | 1 | 2 | When was the region immediately north of the region where the country in which Aluf can be found is located and the Persian Gulf established? |

### musique K=7

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 3 | 3 | How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died? |
| -1.0000 | 2 | 4 | What county has a city that is adjacent to the capital city of Erskine College's state? |
| -1.0000 | 3 | 3 | Who had the lowest batting average in the league where the team with the most games in the series after which the MLB MVP is awarded played? |
| -0.4286 | 1 | 4 | Who stars in the video "One Last Time" by the performer of Baby I? |
| -0.0123 | 2 | 3 | How did did the people fare during the reign of the abolisher of sati partha in India? |

### musique K=10

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 3 | 7 | What is the position of the 1st governor general of India? |
| -1.0000 | 3 | 7 | Who is the child of the president under whom prohibition occurred? |
| -1.0000 | 1 | 9 | Who is the spouse of Young Man Luther's author? |
| -0.9524 | 3 | 7 | What is the meaning of the name of the city where the Yongle emperor greeted the person to whom the edict was addressed? |
| -0.5000 | 3 | 7 | What county has a city that is adjacent to the capital city of Erskine College's state? |

### hotpotqa K=5

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 1 | 4 | In what year was the Golden State NBA player, who was part of the Cavaliers-Warriors rivalry, named NBA Finals Most Valuable Player? |
| -1.0000 | 1 | 3 | What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal? |
| -1.0000 | 2 | 2 | What is the current home arena of the NHL team Chris Summers plays for? |
| -0.5000 | 2 | 2 | Where is the ice hockey team based that Zdeno Chára currently serving as captain of? |

### hotpotqa K=7

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 3 | 4 | The 53rd National Hockey League All-Star Game took place at the indoor arena that was completed in what year? |

### hotpotqa K=10

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 1 | 9 | At what intersection was the former home of the wooden roller coaster now located at Six Flags Great America in Gurnee, Illinois located? |
| -1.0000 | 2 | 8 | Which of Damon Stoudamire's cousins once played college basketball for the University of Kentucky? |
| -0.9994 | 1 | 9 | In what year was the composer of "Anthem" born? |
| -0.6667 | 3 | 7 | This Experts Network sports analysts was inducted into the Pro Football Hall of Fame in 2000 and played in the NFL for how many seasons? |

### 2wikimultihopqa K=5

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -0.7143 | 1 | 3 | Are both movies, Naked Tango and Algiers (Film), from the same country? |
| -0.6000 | 1 | 4 | Which film has the director born later, Christ Walking On The Water or 45 Fathers? |
| -0.5455 | 3 | 1 | What is the place of birth of the performer of song Changed It? |
| -0.2105 | 1 | 2 | Which film has the director who was born first, Tombstone Rashomon or Waiting For The Clouds? |

### 2wikimultihopqa K=7

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -0.2222 | 2 | 3 | Which film has the director who is older than the other, Airheads or Return To Cabin By The Lake?  |
| -0.0455 | 2 | 4 | Which film has the director born later, Romance On The Run or The Palace Of Angels? |

### 2wikimultihopqa K=10

| ΔF1 append3-append0 | appended selected | overlap vs baseline | Question |
|---:|---:|---:|---|
| -1.0000 | 1 | 9 | Are both movies, Naked Tango and Algiers (Film), from the same country? |
| -1.0000 | 2 | 8 | When did Lothair Ii's mother die? |
| -0.2045 | 3 | 7 | What is the place of birth of the performer of song Changed It? |

