# ETv4 MuSiQue 3-Hop Gold-Loss Case Study

Source report: `run_logs/etv4_full1000_optimized_equivalence_20260511/musique/reports/musique_evidence_transition_graphragv4_fact_witnessed_sto_retrieval.json`
OpenIE: `/mnt/nvme/code/HippoRAG/run_logs/evidence_transition_graphragv4_clean_mainline_multi_anchor_strict_musique100_20260511/musique/index/openie_results_ner_gpt-4o-mini.json`

```text
| Item | Value |
|------|-------|
| 3-hop loss cases | 29 |
| Loss definition | ETv4 gold hit count < same-entry dense gold hit count |
| Sample seed | 20260511 |
| Sample size | 10 |
```

## Sample Cases

### Case 1: query_index=222

Question: When is the opening day of the season of the league that Jim Wilson's team plays for?

```text
Gold docs:       [3539, 3542, 3553]
Dense top5:      [8689, 3553, 3554, 3548, 3542]  hit=2  R@5=0.666667
ETv4 top5:       [3553, 3555, 8689, 3554, 3548]  hit=1  R@5=0.333333
Dropped gold:    [3542]
Removed dense:   [3542]
Inserted ETv4:   [3555]
```

Gold docs:
- D3539 `Cleveland Indians` :: The Cleveland Indians are an American professional baseball team based in Cleveland, Ohio. The Indians compete in Major League Baseball (MLB) as a member club of the American Leagu
- D3542 `2018 Major League Baseball season` :: The 2018 Major League Baseball season began on March 29, 2018, and is scheduled to end on September 30. The Postseason will begin on October 2. The 2018 World Series is set to begi
- D3553 `Jim Wilson (first baseman)` :: He was released by the Indians following the 1986 season. After a brief tour in the Minnesota Twins organization, Wilson signed as a free agent with the Seattle Mariners on March 1

Dropped gold docs:
- D3542 `2018 Major League Baseball season` :: The 2018 Major League Baseball season began on March 29, 2018, and is scheduled to end on September 30. The Postseason will begin on October 2. The 2018 World Series is set to begi

Inserted ETv4 docs:
- D3555 `Cleveland Bulldogs` :: The Cleveland Bulldogs were a team that played in Cleveland, Ohio in the National Football League. They were originally called the Indians in 1923, not to be confused with the Clev

### Case 2: query_index=346

Question: When did the party Oklahoma's US Senators come from take control of the body determining the rules of the US House and US Senate?

```text
Gold docs:       [1625, 5207, 911]
Dense top5:      [5207, 1625, 913, 1613, 911]  hit=3  R@5=1.0
ETv4 top5:       [679, 808, 5207, 1625, 913]  hit=2  R@5=0.666667
Dropped gold:    [911]
Removed dense:   [1613, 911]
Inserted ETv4:   [679, 808]
```

Gold docs:
- D1625 `Standing Rules of the United States Senate` :: The Standing Rules of the Senate are the parliamentary procedures adopted by the United States Senate that govern its procedure. The Senate's power to establish rules derives from 
- D5207 `Oklahoma` :: Following the 2000 census, the Oklahoma delegation to the U.S. House of Representatives was reduced from six to five representatives, each serving one congressional district. For t
- D911 `2014 United States Senate elections` :: The Republicans regained the majority of the Senate in the 114th Congress, which started in January 2015; the Republicans had not controlled the Senate since January 2007. They had

Dropped gold docs:
- D911 `2014 United States Senate elections` :: The Republicans regained the majority of the Senate in the 114th Congress, which started in January 2015; the Republicans had not controlled the Senate since January 2007. They had

Inserted ETv4 docs:
- D679 `Treaty of Versailles` :: After the Versailles conference, Democratic President Woodrow Wilson claimed that ``at last the world knows America as the savior of the world! ''However, the Republican Party, led
- D808 `2002 United States House of Representatives elections` :: The Elections for the United States House of Representatives on 5 November 2002 was in the middle of President George W. Bush's first term. Although it was a midterm election, the 

### Case 3: query_index=365

Question: What is the average winter daytime temperature in the region containing Richmond, in the state where WXBX is located?

```text
Gold docs:       [3777, 5455, 3768]
Dense top5:      [5455, 9997, 3768, 1508, 3777]  hit=3  R@5=1.0
ETv4 top5:       [5455, 2045, 9997, 3768, 1508]  hit=2  R@5=0.666667
Dropped gold:    [3777]
Removed dense:   [3777]
Inserted ETv4:   [2045]
```

Gold docs:
- D3777 `North Carolina` :: In winter, the Piedmont is colder than the coast, with temperatures usually averaging in the upper 40s–lower 50s °F (8–12 °C) during the day and often dropping below the freezing p
- D5455 `WXBX` :: WXBX is an Oldies formatted broadcast radio station (95.3 FM) licensed to Rural Retreat, Virginia, serving the Wytheville and Wythe County, Virginia area. WXBX is owned and operate
- D3768 `Richmond, Virginia` :: Richmond is located at 37°32′N 77°28′W﻿ / ﻿37.533°N 77.467°W﻿ / 37.533; -77.467 (37.538, −77.462). According to the United States Census Bureau, the city has a total area of 62 squ

Dropped gold docs:
- D3777 `North Carolina` :: In winter, the Piedmont is colder than the coast, with temperatures usually averaging in the upper 40s–lower 50s °F (8–12 °C) during the day and often dropping below the freezing p

Inserted ETv4 docs:
- D2045 `Virginia` :: Virginia has a total area of , including of water, making it the 35th-largest state by area. Virginia is bordered by Maryland and Washington, D.C. to the north and east; by the Atl

### Case 4: query_index=397

Question: When was the last time Gary Rowett's team beat the 1894-95 FA Cup winner?

```text
Gold docs:       [5811, 1402, 1395]
Dense top5:      [5811, 1395, 1397, 1402, 1157]  hit=3  R@5=1.0
ETv4 top5:       [1395, 6606, 5811, 1394, 1397]  hit=2  R@5=0.666667
Dropped gold:    [1402]
Removed dense:   [1402, 1157]
Inserted ETv4:   [6606, 1394]
```

Gold docs:
- D5811 `Gary Rowett` :: As a player, he was a defender, and played in the Premier League for Everton, Derby County, Leicester City and Charlton Athletic. He also played in the Football League for Cambridg
- D1402 `Second City derby` :: Date Venue Home team Score Competition Round Attendance 5 November 1887 Wellington Road Aston Villa 4 -- 0 FA Cup 2nd Round 23 March 1901 Muntz Street Small Heath 0 -- 0 FA Cup Qua
- D1395 `1894–95 FA Cup` :: The Trophy was stolen from a display in the shop window of W. Shillcock (a football fitter) in Newton Row, Birmingham, after the Final and never recovered despite a £10 reward. Acc

Dropped gold docs:
- D1402 `Second City derby` :: Date Venue Home team Score Competition Round Attendance 5 November 1887 Wellington Road Aston Villa 4 -- 0 FA Cup 2nd Round 23 March 1901 Muntz Street Small Heath 0 -- 0 FA Cup Qua

Inserted ETv4 docs:
- D6606 `Lewis Hunt` :: Lewis James Hunt (born 25 August 1982 in Birmingham, West Midlands) is an English footballer who last played for Sutton United.
- D1394 `Premier League` :: Premier League Founded 20 February 1992 Country England (19 teams) Other club (s) from Wales (1 team) Confederation UEFA Number of teams 20 Level on pyramid Relegation to EFL Champ

### Case 5: query_index=487

Question: The native country of emu birds started conscription. This occurred during the war depicted in The Things They Carried. What year did the draft start?

```text
Gold docs:       [5711, 6893, 6895]
Dense top5:      [6893, 5711, 6889, 8347, 6895]  hit=3  R@5=1.0
ETv4 top5:       [6893, 2199, 5711, 6889, 8347]  hit=2  R@5=0.666667
Dropped gold:    [6895]
Removed dense:   [6895]
Inserted ETv4:   [2199]
```

Gold docs:
- D5711 `Conscription in Australia` :: In 1964 compulsory National Service for 20 - year - old males was introduced under the National Service Act (1964). The selection of conscripts was made by a sortition or lottery d
- D6893 `The Things They Carried` :: The Things They Carried (1990) is a collection of linked short stories by American novelist Tim O'Brien, about a platoon of American soldiers fighting on the ground in the Vietnam 
- D6895 `Bird migration` :: Bird migration is not limited to birds that can fly. Most species of penguin (Spheniscidae) migrate by swimming. These routes can cover over 1,000 km (620 mi). Dusky grouse Dendrag

Dropped gold docs:
- D6895 `Bird migration` :: Bird migration is not limited to birds that can fly. Most species of penguin (Spheniscidae) migrate by swimming. These routes can cover over 1,000 km (620 mi). Dusky grouse Dendrag

Inserted ETv4 docs:
- D2199 `Powers of the United States Congress` :: The Constitution also gives Congress an important role in national defense, including the exclusive power to declare war, to raise and maintain the armed forces, and to make rules 

### Case 6: query_index=545

Question: In what region of the country where Lam Dong is located is John Phan's birthplace?

```text
Gold docs:       [4984, 4983, 7487]
Dense top5:      [4983, 7487, 6296, 4992, 4984]  hit=3  R@5=1.0
ETv4 top5:       [4983, 5055, 7487, 6296, 4992]  hit=2  R@5=0.666667
Dropped gold:    [4984]
Removed dense:   [4984]
Inserted ETv4:   [5055]
```

Gold docs:
- D4984 `South Central Coast` :: South Central Coast (Vietnamese: Duyên hải Nam Trung Bộ) is one of the regions of Vietnam. It consists of the independent municipality of Đà Nẵng and seven other provinces. The two
- D4983 `John Phan` :: Bon "John" Phan (born October 10, 1974 in Da Nang, Vietnam) is a Vietnamese-American professional poker player based in Stockton, California who is a two time World Series of Poker
- D7487 `Lâm Đồng Province` :: Lâm Đồng () is a province located in the Central Highlands () region of Vietnam. Its capital is Da Lat. Lâm Đồng borders Khánh Hòa Province and Ninh Thuận Province to the east, Đồn

Dropped gold docs:
- D4984 `South Central Coast` :: South Central Coast (Vietnamese: Duyên hải Nam Trung Bộ) is one of the regions of Vietnam. It consists of the independent municipality of Đà Nẵng and seven other provinces. The two

Inserted ETv4 docs:
- D5055 `San Diego` :: Stockton and Kearny went on to recover Los Angeles and force the capitulation of Alta California with the "Treaty of Cahuenga" on January 13, 1847. As a result of the Mexican–Ameri

### Case 7: query_index=607

Question: What does seal stand for in the operator of the list of destroyer classes of the operator of the USS Tringa seals?

```text
Gold docs:       [399, 387, 8142]
Dense top5:      [399, 8142, 401, 400, 387]  hit=3  R@5=1.0
ETv4 top5:       [8142, 1043, 399, 401, 400]  hit=2  R@5=0.666667
Dropped gold:    [387]
Removed dense:   [387]
Inserted ETv4:   [1043]
```

Gold docs:
- D399 `United States Navy SEALs` :: The United States Navy's ``Sea, Air, and Land ''Teams, commonly abbreviated as the Navy SEALs, are the U.S. Navy's primary special operations force and a component of the Naval Spe
- D387 `List of destroyer classes of the United States Navy` :: The first major warship produced by the U.S. Navy after World War II (and in the Cold War) were "frigates"—the ships were originally designated destroyer leaders but reclassified i
- D8142 `USS Tringa (ASR-16)` :: USS "Tringa" (ASR-16) was a Chanticleer-class submarine rescue ship of the United States Navy. She was laid down on 12 July 1945 at Savannah, Georgia, by the Savannah Machine & Fou

Dropped gold docs:
- D387 `List of destroyer classes of the United States Navy` :: The first major warship produced by the U.S. Navy after World War II (and in the Cold War) were "frigates"—the ships were originally designated destroyer leaders but reclassified i

Inserted ETv4 docs:
- D1043 `Savannah, Georgia` :: Savannah (/ səˈvænə /) is the oldest city in the U.S. state of Georgia and is the county seat of Chatham County. Established in 1733 on the Savannah River, the city of Savannah bec

### Case 8: query_index=803

Question: In what region of the country where Ha Hoa is located is the city where Zone 5 Military Museum is found?

```text
Gold docs:       [4991, 5419, 4984]
Dense top5:      [5419, 4991, 7205, 4992, 4984]  hit=3  R@5=1.0
ETv4 top5:       [5419, 3182, 4991, 7205, 4992]  hit=2  R@5=0.666667
Dropped gold:    [4984]
Removed dense:   [4984]
Inserted ETv4:   [3182]
```

Gold docs:
- D4991 `Hạ Hòa District` :: Hạ Hòa is a rural district of Phú Thọ Province in the Northeast region of Vietnam. As of 2003, the district had a population of 108,556. The district covers an area of 340 km². The
- D5419 `Zone 5 Military Museum, Danang` :: The Zone 5 Military Museum (Bao Tang Khu 5) is a military museum located at 3 Duy Tân, Da Nang, Vietnam. It covers all Vietnamese resistance to foreign occupation from the Chinese 
- D4984 `South Central Coast` :: South Central Coast (Vietnamese: Duyên hải Nam Trung Bộ) is one of the regions of Vietnam. It consists of the independent municipality of Đà Nẵng and seven other provinces. The two

Dropped gold docs:
- D4984 `South Central Coast` :: South Central Coast (Vietnamese: Duyên hải Nam Trung Bộ) is one of the regions of Vietnam. It consists of the independent municipality of Đà Nẵng and seven other provinces. The two

Inserted ETv4 docs:
- D3182 `John Kerry` :: On April 22, 1971, Kerry appeared before a U.S. Senate committee hearing on proposals relating to ending the war. The day after this testimony, Kerry participated in a demonstratio

### Case 9: query_index=921

Question: When did the maker of the Acura Legend, the largest auto company in the world, and Nissan open US assembly plants?

```text
Gold docs:       [1638, 1627, 10989]
Dense top5:      [1638, 1627, 1645, 10989, 1979]  hit=3  R@5=1.0
ETv4 top5:       [1631, 1287, 1638, 1627, 1645]  hit=2  R@5=0.666667
Dropped gold:    [10989]
Removed dense:   [10989, 1979]
Inserted ETv4:   [1631, 1287]
```

Gold docs:
- D1638 `Acura Legend` :: The Acura Legend is a mid-size luxury/executive car manufactured by Honda. It was sold in the U.S., Canada, and parts of China under Honda's luxury brand, Acura, from 1985 to 1995,
- D1627 `1973 oil crisis` :: Some buyers lamented the small size of the first Japanese compacts, and both Toyota and Nissan (then known as Datsun) introduced larger cars such as the Toyota Corona Mark II, the 
- D10989 `Automotive industry` :: Rank Group Country Vehicles Toyota Japan 10,213,486 Volkswagen Group Germany 10,126,281 Hyundai South Korea 7,889,538 General Motors United States 7,793,066 5 Ford United States 6,

Dropped gold docs:
- D10989 `Automotive industry` :: Rank Group Country Vehicles Toyota Japan 10,213,486 Volkswagen Group Germany 10,126,281 Hyundai South Korea 7,889,538 General Motors United States 7,793,066 5 Ford United States 6,

Inserted ETv4 docs:
- D1631 `Infiniti J30` :: The Infiniti J30, or Nissan Leopard J Ferie in Japan, was a rear wheel drive luxury car. The J30 went into production on April 7, 1992 as a 1993 model to replace the M30 (which was
- D1287 `United States` :: The United States of America (USA), commonly known as the United States (U.S.) or America, is a federal republic composed of 50 states, a federal district, five major self - govern

### Case 10: query_index=999

Question: What UK label was bought by the broadcast company that, along with ABC and the original network of Undercovers, is one of the major broadcasters based in New York?

```text
Gold docs:       [1877, 1875, 11655]
Dense top5:      [1877, 11655, 1887, 156, 1875]  hit=3  R@5=1.0
ETv4 top5:       [11655, 10185, 1877, 1887, 156]  hit=2  R@5=0.666667
Dropped gold:    [1875]
Removed dense:   [1875]
Inserted ETv4:   [10185]
```

Gold docs:
- D1877 `New York City` :: The television industry developed in New York and is a significant employer in the city's economy. The three major American broadcast networks are all headquartered in New York: AB
- D1875 `Sony Music` :: In 1964, CBS established its own UK distribution with the acquisition of Oriole Records. EMI continued to distribute Epic and Okeh label material on the Columbia label in the UK un
- D11655 `Undercovers (TV series)` :: Undercovers is an American action spy television series created by J. J. Abrams and Josh Reims that aired NBC from September 22 to December 29, 2010. They were executive producers 

Dropped gold docs:
- D1875 `Sony Music` :: In 1964, CBS established its own UK distribution with the acquisition of Oriole Records. EMI continued to distribute Epic and Okeh label material on the Columbia label in the UK un

Inserted ETv4 docs:
- D10185 `Westworld (TV series)` :: Nolan and Joy serve as executive producers, along with J.J. Abrams, Jerry Weintraub, and Bryan Burk. The first season was broadcast between October 2 and December 4, 2016; it compr
