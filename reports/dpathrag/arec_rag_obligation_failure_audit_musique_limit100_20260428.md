# AREC Generated Obligation Failure Audit

- Dataset: `musique`
- Rows sampled: `30`
- Sampling: generated top20 fixed-pool harms first, then unchanged, then improved.
- Label options: `correct_useful, duplicate_or_paraphrase, answer_leak, vacuous, wrong_entity_or_wrong_relation, other`

## Aggregate Heuristic Flags

| Flag | Count |
|---|---:|
| answer_string_present | 27 |
| duplicate_or_paraphrase_candidate | 8 |

## Cases

### 1. qid=31 hop=2

Question: Who is played by the director of The Good Shepherd in The Godfather?

Gold answers: `['Vito Andolini', 'Vito Andolini Corleone', 'Vito Corleone']`

Gold titles: `['The Godfather Part II', 'The Good Shepherd (film)']`

Initial answer: `Robert De Niro`

Initial top5: `['The Good Shepherd (film)', 'The Godfather Part II', 'Tom Hagen', 'Vito Corleone', 'Mary Corleone']`

Generated top20 projection: `['Vito Corleone', 'The Godfather (film series)', 'Al Neri', 'John Cazale', 'Nino Rota']`

Support delta: recall 1.0000 -> 0.0000; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The director of The Good Shepherd is Francis Ford Coppola. | none | TODO |
| In The Godfather, the character of Michael Corleone is played by Robert De Niro. | answer_string_present | TODO |

### 2. qid=76 hop=2

Question: Who is the father of Empress Wang's husband?

Gold answers: `['Yang Xingmi']`

Gold titles: `['Empress Dowager Wang (Rui)', 'Empress Wang (Yang Pu)']`

Initial answer: `Wang Xianzhi`

Initial top5: `['Empress Dowager Wang (Rui)', "Wang Shen'ai", 'Empress Wang (Yang Pu)', 'Empress Dowager Huang', 'Empress Dowager Xu']`

Generated top20 projection: `['Empress Dowager Huang', 'Jin Feishan', 'Han dynasty', 'Princess Pingyang (Han dynasty)', 'Lady Li (Wang Jipeng)']`

Support delta: recall 1.0000 -> 0.0000; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Empress Wang's husband is Wang Xianzhi. | answer_string_present | TODO |
| Wang Xianzhi is the father of Empress Wang. | answer_string_present | TODO |

### 3. qid=16 hop=3

Question: When was the Palau de la Generalitat constructed in the city where Martin from the region where Perdiguera is located died?

Gold answers: `['15th century', 'built in the 15th century']`

Gold titles: `['Gothic architecture', 'Martin of Aragon', 'Perdiguera']`

Initial answer: `15th century`

Initial top5: `['Martin of Aragon', 'Palau Güell', 'Perdiguera', 'Josep Domènech i Estapà', 'Gothic architecture']`

Generated top20 projection: `['Gothic architecture', 'Paco Godia', 'Palau Güell', 'Jordi Solé Tura', 'Joan Llimona']`

Support delta: recall 1.0000 -> 0.3333; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Martin from the region where Perdiguera is located died in the city of Barcelona. | none | TODO |
| Perdiguera is located in the region of Catalonia. | none | TODO |
| The Palau de la Generalitat was constructed in the 15th century in Barcelona. | answer_string_present | TODO |

### 4. qid=83 hop=4

Question: How long are the city council terms in the second largest city in the state where Yuma is located?

Gold answers: `['four-year']`

Gold titles: `['Tucson, Arizona', 'Tucson, Arizona', 'Yuma County Library District', 'Yuma, Colorado']`

Initial answer: `Four years.`

Initial top5: `['Yuma, Colorado', 'Tucson, Arizona', 'Yuma County Library District', 'Richmond, Virginia', 'Arizona']`

Generated top20 projection: `['List of counties in Arizona', 'Second City derby', 'Governor of Colima', 'Tucson, Arizona', 'Tucson, Arizona']`

Support delta: recall 1.0000 -> 0.3333; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Yuma is located in the state of Arizona. | none | TODO |
| The second largest city in Arizona is Phoenix. | none | TODO |
| The city council term length in Phoenix is four years. | answer_string_present | TODO |

### 5. qid=19 hop=2

Question: Who is the mascot of the university related to Randy Conrads?

Gold answers: `['Benny Beaver']`

Gold titles: `['Benny Beaver', 'Randy Conrads']`

Initial answer: `Benny Beaver`

Initial top5: `['Randy Conrads', 'The Oregon Duck', 'Smokey (mascot)', 'Benny Beaver', 'Sebastian the Ibis']`

Generated top20 projection: `['Robert Wayman', 'Alma mater', 'John Darsee', 'Benny Beaver', 'Matt Burch']`

Support delta: recall 1.0000 -> 0.5000; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Randy Conrads is associated with the University of Nevada, Las Vegas (UNLV). | none | TODO |
| The mascot of the University of Nevada, Las Vegas (UNLV) is Benny Beaver. | answer_string_present | TODO |

### 6. qid=52 hop=2

Question: Who did the person who married their half sister in the bible marry after the death of sarah?

Gold answers: `['Keturah']`

Gold titles: `['Incest in the Bible', 'Keturah']`

Initial answer: `Keturah`

Initial top5: `['Incest in the Bible', 'Sarah', 'Keturah', 'Rebecca', 'Hetepheres II']`

Generated top20 projection: `['Incest in the Bible', 'Sarah', 'John FitzAlan, 2nd Baron Arundel', 'Ahmose-Henuttamehu', 'Chileab']`

Support delta: recall 1.0000 -> 0.5000; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Abraham married Keturah after the death of Sarah. | answer_string_present | TODO |
| Keturah is mentioned in the Bible as one of Abraham's wives after Sarah's death. | answer_string_present | TODO |
| Abraham married his half-sister Sarah, according to the Bible. | none | TODO |

### 7. qid=54 hop=2

Question: Who was Beyonce's husband talking about in the the song Cry?

Gold answers: `['three different relationships he had in the past']`

Gold titles: `['Beyoncé', 'Song Cry']`

Initial answer: `Jay-Z`

Initial top5: `['Song Cry', 'Beyoncé', 'Beyoncé', 'Someone like You (Adele song)', 'Beyoncé']`

Generated top20 projection: `['Beyoncé', 'Beyoncé', 'Beyoncé', 'Beyoncé', 'Beyoncé']`

Support delta: recall 1.0000 -> 0.5000; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Beyoncé's song 'Cry' references her husband Jay-Z. | answer_string_present | TODO |
| Jay-Z is a well-known rapper and record producer who was married to Beyoncé from 2008 to 2014. | answer_string_present | TODO |
| The song 'Cry' was released in 2011 as part of Beyoncé's album '4'. | none | TODO |

### 8. qid=58 hop=2

Question: Who was the spouse of a leading speaker against slavery and publisher of an antislavery newspaper?

Gold answers: `['Helen Pitts Douglass']`

Gold titles: `['Helen Pitts Douglass', 'The North Star (anti-slavery newspaper)']`

Initial answer: `Helen Pitts Douglass`

Initial top5: `['Helen Pitts Douglass', 'The North Star (anti-slavery newspaper)', 'William Cooper Nell', 'José Ortega Spottorno', 'Anne-Antoinette Diderot']`

Generated top20 projection: `['Helen Pitts Douglass', 'Annie Besant', 'Lawrence Brainerd', 'Mary Birdsall', 'Thomas S. Hinde']`

Support delta: recall 1.0000 -> 0.5000; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Helen Pitts Douglass was a leading speaker against slavery. | answer_string_present | TODO |
| Helen Pitts Douglass was the publisher of an antislavery newspaper. | answer_string_present | TODO |
| Helen Pitts Douglass was married to a prominent figure known for opposing slavery. | answer_string_present | TODO |

### 9. qid=20 hop=3

Question: What does seal stand for in the U.S. military branch that operates destroyers, as well as the USS Edsall?

Gold answers: `['Sea, Air, and Land']`

Gold titles: `['List of destroyer classes of the United States Navy', 'USS Edsall (DE-129)', 'United States Navy SEALs']`

Initial answer: `SEAL stands for Sea, Air, and Land in the U.S. military branch that operates destroyers, as well as the USS Edsall.`

Initial top5: `['USS Edsall (DE-129)', 'United States Navy SEALs', 'United States Navy SEALs', 'List of destroyer classes of the United States Navy', 'USS Rhodes (DE-384)']`

Generated top20 projection: `['USS Edsall (DE-129)', 'United States Navy SEALs', 'USS Brownson (DD-868)', 'USS Kirkpatrick (DE-318)', 'USS Catbird (AM-68)']`

Support delta: recall 1.0000 -> 0.6667; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The U.S. military branch that operates destroyers is the United States Navy. | none | TODO |
| The USS Edsall is a ship that belongs to the United States Navy. | none | TODO |
| The term SEAL is an acronym that stands for Sea, Air, and Land. | none | TODO |
| The United States Navy's SEAL units are known as the Naval Special Warfare Command (NAVSPECWARCOM). | none | TODO |

### 10. qid=36 hop=3

Question: Who is the child of the president who suffered a major defeat when the body which approves members of the American cabinet refused to ratify the Versailles treaty?

Gold answers: `['Jessie Woodrow Wilson', 'Jessie Woodrow Wilson Sayre']`

Gold titles: `['Cabinet of the United States', 'Jessie Woodrow Wilson Sayre', 'Treaty of Versailles']`

Initial answer: `Jessie Woodrow Wilson Sayre`

Initial top5: `['Treaty of Versailles', 'William Howard Taft', 'Vance C. McCormick', 'Cabinet of the United States', 'Jessie Woodrow Wilson Sayre']`

Generated top20 projection: `['Jessie Woodrow Wilson Sayre', 'Treaty of Versailles', 'Gerald Ford', 'Martin Van Buren', 'Imperialism']`

Support delta: recall 1.0000 -> 0.6667; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Jessie Woodrow Wilson Sayre is the child of Woodrow Wilson. | answer_string_present | TODO |
| Woodrow Wilson was the president of the United States. | none | TODO |
| Woodrow Wilson suffered a major defeat when the Senate refused to ratify the Treaty of Versailles. | none | TODO |

### 11. qid=93 hop=3

Question: When did Nissan, the Acura Legend maker and the Scion Fuse manufacturer open US assembly plants?

Gold answers: `['1981']`

Gold titles: `['1973 oil crisis', 'Acura Legend', 'Scion Fuse']`

Initial answer: `Honda, Toyota, and Nissan opened US assembly plants in the early 1980s.`

Initial top5: `['Scion Fuse', 'Acura Legend', '1973 oil crisis', 'Scion (automobile)', 'Toyota']`

Generated top20 projection: `['Toyota', '1973 oil crisis', 'Acura Legend', 'Acura RDX', 'Nissan Rogue']`

Support delta: recall 1.0000 -> 0.6667; SC 1.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Nissan opened a US assembly plant in the early 1980s. | duplicate_or_paraphrase_candidate | TODO |
| Honda opened a US assembly plant in the early 1980s. | duplicate_or_paraphrase_candidate | TODO |
| Toyota opened a US assembly plant in the early 1980s. | duplicate_or_paraphrase_candidate | TODO |

### 12. qid=57 hop=3

Question: What is the name of the famous bridge in the birth city of the composer of Scanderbeg?

Gold answers: `['Ponte di Rialto', 'Rialto Bridge']`

Gold titles: `['Orlando furioso (Vivaldi, 1714)', 'Rialto Bridge', 'Scanderbeg (opera)']`

Initial answer: `Bridge of Sighs`

Initial top5: `['Scanderbeg (opera)', 'Antonio Vivaldi', 'Bridge of Sighs', 'Konrad Adenauer Bridge', 'Rialto Bridge']`

Generated top20 projection: `['Bridge of Sighs', 'List of bridges of Pittsburgh', 'Giuseppe Demachi', 'Paris', 'Nino Rota']`

Support delta: recall 0.6667 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The composer of Scanderbeg was born in Venice. | none | TODO |
| The Bridge of Sighs is a famous bridge in Venice. | answer_string_present | TODO |

### 13. qid=74 hop=3

Question: What is the direction of flow of the body of water by the city where Write This Down was formed?

Gold answers: `['Minnesota', 'rises in northern Minnesota and meanders slowly southwards']`

Gold titles: `['Minneapolis', 'Mississippi River', 'Write This Down (band)']`

Initial answer: `The direction of flow of the body of water by the city where Write This Down was formed is east.`

Initial top5: `['Write This Down (band)', 'New York City', 'New York City', 'Pambula River', 'Minneapolis']`

Generated top20 projection: `['Tomaga River', 'Calabar River', 'Allegheny River', 'Pambula River', 'Lisala']`

Support delta: recall 0.6667 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Write This Down was formed in a city located near a body of water. | none | TODO |
| The body of water near the city where Write This Down was formed flows in an eastern direction. | none | TODO |

### 14. qid=9 hop=2

Question: What is the Till dom ensamma performer's birth date?

Gold answers: `['11 September 1962']`

Gold titles: `['Mauro Scocco', 'Till dom ensamma']`

Initial answer: `The evidence does not provide the birth date of the Till dom ensamma performer.`

Initial top5: `['Till dom ensamma', 'När hela världen ser på', 'Långa nätter', 'Stig Olin', 'Styrbjörn Holm']`

Generated top20 projection: `['Lucy Fallon', 'Du får göra som du vill', 'Styrbjörn Holm', 'Långa nätter', 'Jag ångrar ingenting (song)']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The Till dom ensamma performer's birth date is listed in a reliable biographical source. | none | TODO |
| A credible source confirms the birth date of the Till dom ensamma performer. | none | TODO |
| The Till dom ensamma performer's birth date is documented in an official record. | none | TODO |

### 15. qid=10 hop=2

Question: How many episodes are in season 5 of the series with The Bag or the Bat?

Gold answers: `['12']`

Gold titles: `['List of Ray Donovan episodes', 'The Bag or the Bat']`

Initial answer: `13`

Initial top5: `['List of Orange Is the New Black episodes', 'The Bag or the Bat', 'List of How I Met Your Mother episodes', 'Arrested Development (season 5)', 'Cheatty Cheatty Bang Bang']`

Generated top20 projection: `['Real Time with Bill Maher', 'List of The Walking Dead episodes', 'List of The Originals episodes', 'List of How I Met Your Mother episodes', 'List of Doc Martin episodes']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The series with The Bag or the Bat is titled 'The Office (US)' | none | TODO |
| Season 5 of 'The Office (US)' contains 13 episodes | answer_string_present | TODO |

### 16. qid=12 hop=2

Question: What is the birthplace of the person after whom São José dos Campos was named?

Gold answers: `['Nazareth']`

Gold titles: `['Sisters of St Joseph of Nazareth', 'São José dos Campos']`

Initial answer: `Pedro Leopoldo, State of Minas Gerais`

Initial top5: `['São José dos Campos', 'Chico Xavier', 'Casa Natal del General Santander', 'São José dos Quatro Marcos', 'Roberto Irineu Marinho']`

Generated top20 projection: `['Chico Xavier', 'Leonhard Seppala', 'San Miguel de Allende', 'Grover Cleveland', 'Niels Henrik Abel']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| São José dos Campos was named after a person named Pedro Leopoldo. | none | TODO |
| Pedro Leopoldo was born in the State of Minas Gerais. | none | TODO |

### 17. qid=15 hop=2

Question: What company succeeded the owner of Empire Sports Network?

Gold answers: `['Time Warner', 'Time Warner Cable']`

Gold titles: `['Empire Sports Network', 'Windjammer Communications']`

Initial answer: `<think> </think> {"answer": "The company that succeeded the owner of Empire Sports Network was Fox Entertainment Group (`

Initial top5: `['Empire Sports Network', 'Fox Sports 1', 'Fox Sports 1', 'Fox Sports West and Prime Ticket', 'WFAN-FM']`

Generated top20 projection: `['Fox Sports 1', 'Fox Sports 1', 'The Wall Street Journal', 'New York Giants Radio Network', 'Fox Sports West and Prime Ticket']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Empire Sports Network was acquired by Fox Entertainment Group. | duplicate_or_paraphrase_candidate | TODO |
| Fox Entertainment Group was the successor company to the owner of Empire Sports Network. | duplicate_or_paraphrase_candidate | TODO |
| The owner of Empire Sports Network was sold to Fox Entertainment Group. | duplicate_or_paraphrase_candidate | TODO |

### 18. qid=22 hop=2

Question: Where was the person who wrote about the rioting being a dividing factor in Birmingham educated?

Gold answers: `['University of Glasgow']`

Gold titles: `['James Watt', 'Priestley Riots']`

Initial answer: `Birmingham University`

Initial top5: `['Priestley Riots', 'Richard Cory-Wright', 'Ernest Barnes', 'Ian Quayle Jones', 'Out of the Ashes (book)']`

Generated top20 projection: `['Rob Larter', 'Moinuddin Ahmed Chowdhury', 'Richard Cory-Wright', 'Edmund Burke', '1967 Detroit riot']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The person who wrote about the rioting being a dividing factor in Birmingham attended Birmingham University. | answer_string_present | TODO |

### 19. qid=51 hop=4

Question: Who won the Indy Car Race in the largest populated city of the state where the performer of Mingus Three is from?

Gold answers: `['Mario Andretti']`

Gold titles: `['Charles Mingus', 'Desert Diamond West Valley Phoenix Grand Prix', 'Mingus Three', 'Tucson, Arizona']`

Initial answer: `The winner of the Indy Car Race in Indianapolis, the largest populated city in Indiana, is not provided in the evidence.`

Initial top5: `['Mingus Three', 'Charles Mingus', 'Indiana', 'Saint Paul, Minnesota', 'A Modern Jazz Symposium of Music and Poetry']`

Generated top20 projection: `['2018 Indianapolis 500', 'Indiana', 'Detroit', 'Jazz Portraits: Mingus in Wonderland', 'Oklahoma City']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The performer of the song 'Mingus Three' is from Indiana. | none | TODO |
| Indianapolis is the largest populated city in Indiana. | none | TODO |
| There is an Indy Car Race held in Indianapolis. | none | TODO |

### 20. qid=70 hop=2

Question: What does the river that Darling Mills Creek turns into flow into?

Gold answers: `['Port Jackson', 'Sydney Harbour']`

Gold titles: `['Darling Mills Creek', 'Parramatta River']`

Initial answer: `The river that Darling Mills Creek turns into flows into the Tasman Sea.`

Initial top5: `['Darling Mills Creek', 'Murrah River', 'Pambula River', 'Doubtful Creek', 'Mamaroneck River']`

Generated top20 projection: `['Tarbell Brook', 'Tomaga River', 'Black Creek (Toronto)', 'Staaten River', 'List of crossings of the Kiskiminetas River']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Darling Mills Creek eventually becomes part of the Clarence River. | none | TODO |
| The Clarence River flows into the Tasman Sea. | none | TODO |

### 21. qid=81 hop=2

Question: Who is the spouse of the actor that plays Jarvis in the Avengers Age of Ultron?

Gold answers: `['Jennifer Connelly']`

Gold titles: `['A Beautiful Mind (soundtrack)', 'Paul Bettany']`

Initial answer: `Rebecca Romijn`

Initial top5: `['Paul Bettany', 'Avengers: Age of Ultron', 'Andy Serkis', "Jerry O'Connell", 'Samuel L. Jackson']`

Generated top20 projection: `["Jerry O'Connell", 'Ashley Wilkes', 'George Wickham', 'List of Marvel Cinematic Universe films', 'Black Widow (Natasha Romanova)']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The actor who plays Jarvis in the Avengers: Age of Ultron is James Badge Dale. | none | TODO |
| James Badge Dale's spouse is Rebecca Romijn. | answer_string_present | TODO |

### 22. qid=82 hop=2

Question: What is the location of formation of the film company distributing The Boss?

Gold answers: `['Fort Lee']`

Gold titles: `['The Boss (1915 film)', 'World Film Company']`

Initial answer: `The film company distributing The Boss is DreamWorks Animation, which was formed in the United States.`

Initial top5: `['The Boss Baby', 'The Boss Baby', 'Universal Pictures', 'World Film Company', 'A Boss in the Living Room']`

Generated top20 projection: `['The Boss Baby', 'The Boss Baby', 'The Post (film)', 'The Post (film)', 'UHF (film)']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| DreamWorks Animation is the film company distributing The Boss. | none | TODO |
| DreamWorks Animation was formed in the United States. | none | TODO |

### 23. qid=84 hop=2

Question: Who is the child of the president under whom prohibition occurred?

Gold answers: `['Jessie Woodrow Wilson', 'Jessie Woodrow Wilson Sayre']`

Gold titles: `['Jessie Woodrow Wilson Sayre', 'Prohibition in the United States']`

Initial answer: `Chester Alan Arthur II`

Initial top5: `['Prohibition in the United States', 'Prohibition in the United States', 'Eighteenth Amendment to the United States Constitution', 'William Howard Taft', 'Chester Alan Arthur II']`

Generated top20 projection: `['Director of the National Security Agency', 'Chester Alan Arthur II', 'Martin Van Buren', 'State of the Union', 'Treaty of Versailles']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Chester Alan Arthur II was the child of Chester Alan Arthur. | answer_string_present | TODO |
| Chester Alan Arthur served as the 21st President of the United States. | none | TODO |
| Prohibition in the United States occurred during the presidency of Warren G. Harding. | none | TODO |

### 24. qid=97 hop=2

Question: Who is the child of the Victim of Romance performer?

Gold answers: `['Chynna Phillips']`

Gold titles: `['Victim of Romance', 'Wilson Phillips']`

Initial answer: `The evidence does not provide information about the child of the Victim of Romance performer.`

Initial top5: `['Victim of Romance', 'Louis Chedid', 'Stig Olin', 'List of show business families', 'David Profumo']`

Generated top20 projection: `['Missing You (Diana Ross song)', 'Madonna (entertainer)', 'Neil Reid', 'The King of Rome', 'Philippine de Rothschild']`

Support delta: recall 0.5000 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The Victim of Romance performer is a known public figure. | none | TODO |
| There is public information available about the personal life of the Victim of Romance performer. | none | TODO |
| The Victim of Romance performer has children listed in publicly accessible records. | none | TODO |
| The identity of the child of the Victim of Romance performer is known and documented. | none | TODO |

### 25. qid=1 hop=3

Question: What month did the Tripartite discussions begin between Britain, France, and the country where, despite being headquartered in the nation called the nobilities commonwealth, the top-ranking Warsaw Pact operatives originated?

Gold answers: `['mid-June']`

Gold titles: `['Molotov–Ribbentrop Pact', 'Szlachta', 'Warsaw Pact']`

Initial answer: `June`

Initial top5: `['Molotov–Ribbentrop Pact', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact']`

Generated top20 projection: `['Warsaw Pact', 'Allies of World War II', 'Warsaw Pact', 'Warsaw Pact', 'Warsaw Pact']`

Support delta: recall 0.6667 -> 0.3333; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The Tripartite discussions between Britain, France, and the country where the top-ranking Warsaw Pact operatives originated began in June. | answer_string_present | TODO |
| The country where the top-ranking Warsaw Pact operatives originated is the same nation where the nobilities commonwealth is headquartered. | none | TODO |
| The Warsaw Pact was a military alliance formed by the Soviet Union and its Eastern European satellite states. | none | TODO |
| The Tripartite discussions refer to the meetings between Britain, France, and the United States during the Cold War. | none | TODO |

### 26. qid=32 hop=3

Question: How long are the council terms of the city that shares a county with Helvetia?

Gold answers: `['four-year']`

Gold titles: `['Helvetia, Arizona', 'Pima County Natural Resources, Parks and Recreation', 'Tucson, Arizona']`

Initial answer: `Four years.`

Initial top5: `['Helvetia, Arizona', 'Tucson, Arizona', 'Richmond, Virginia', 'San Diego', 'Białystok']`

Generated top20 projection: `['Richmond, Virginia', 'Tucson, Arizona', 'Governor of Colima', 'Lake Helena', 'Ann Arbor, Michigan']`

Support delta: recall 0.6667 -> 0.3333; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The city that shares a county with Helvetia is named [City Name]. | none | TODO |
| The council term length for [City Name] is four years. | answer_string_present | TODO |

### 27. qid=40 hop=3

Question: The Commission of Truth and Friendship established governments for Bustami's birth country and a second country that had who as president after declaring independence?

Gold answers: `['Francisco Guterres']`

Gold titles: `['Bustami', 'East Timor', 'Indonesia–Timor Leste Commission of Truth and Friendship']`

Initial answer: `David Dacko`

Initial top5: `['Indonesia–Timor Leste Commission of Truth and Friendship', 'Independence Day (United States)', 'Central African Republic', 'Estanislau da Silva', 'Leaders of the Vietnam War']`

Generated top20 projection: `['David Dacko', 'Father of the Nation', 'Martin Van Buren', 'Korean War', 'Josip Broz Tito']`

Support delta: recall 0.3333 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The Commission of Truth and Friendship established a government for the country where Bustami was born. | none | TODO |
| The Commission of Truth and Friendship also established a government for a second country that declared independence. | none | TODO |
| The second country that declared independence had David Dacko as its president. | answer_string_present | TODO |

### 28. qid=46 hop=3

Question: what is meaning of the word that is a majority religion of the area that became India when the country origin of Mizraab was created in Arabic dictionary?

Gold answers: `['IN', 'IND', 'India', 'Republic of India', 'in', 'the country of India']`

Gold titles: `['Hindus', 'Mizraab', 'Partition of India']`

Initial answer: `Hinduism`

Initial top5: `['Hindus', 'Mizraab', 'Culture of India', 'Islam in India', '2011 Census of India']`

Generated top20 projection: `['2011 Census of India', 'Christian', 'Partition of India', 'Madrasa', 'Jews']`

Support delta: recall 0.6667 -> 0.3333; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The majority religion in the area that became India is Hinduism. | answer_string_present | TODO |
| The word 'Mizraab' originates from the Arabic dictionary and refers to a specific term related to religious or cultural practices. | none | TODO |
| The area that became India was predominantly inhabited by people who practiced Hinduism before the country's formation. | answer_string_present | TODO |

### 29. qid=50 hop=3

Question: When did the country the top-ranking Warsaw Pact operatives came from, despite it being headquartered in the country where A Generation is set, agree to a unified Germany inside NATO?

Gold answers: `['1990', 'May 1990']`

Gold titles: `['A Generation', 'German reunification', 'Warsaw Pact']`

Initial answer: `1970`

Initial top5: `['Warsaw Pact', 'Warsaw Pact', 'A Generation', 'Warsaw Pact', 'Warsaw Pact']`

Generated top20 projection: `['Warsaw Pact', 'Germany', 'Warsaw Pact', 'Warsaw Pact', 'Member states of NATO']`

Support delta: recall 0.6667 -> 0.3333; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| The country from which the top-ranking Warsaw Pact operatives came is East Germany. | none | TODO |
| The country where the novel 'A Generation' is set is West Germany. | none | TODO |
| East Germany agreed to a unified Germany inside NATO in 1970. | answer_string_present | TODO |

### 30. qid=64 hop=3

Question: Who was a prominent figure at the radio division of the network that created the version of The Biggest Loser set in the country where Seria is?

Gold answers: `['Walter Sabo']`

Gold titles: `['Adult contemporary music', 'Seria, Belait', 'The Biggest Loser Brunei: Lose It All']`

Initial answer: `Ali Vincent`

Initial top5: `['The Biggest Loser Brunei: Lose It All', 'Ali Vincent', 'The Biggest Loser (season 2)', 'The Biggest Loser (season 1)', 'The Biggest Loser (season 16)']`

Generated top20 projection: `['Big Brother (American TV series)', 'WMMI', 'Love Around', 'DXAQ-AM', 'Taxi Orange']`

Support delta: recall 0.3333 -> 0.0000; SC 0.0000 -> 0.0000

Manual query label: `TODO`

| Obligation | Heuristic flags | Manual label |
|---|---|---|
| Ali Vincent was a prominent figure at the radio division of the network that created the version of The Biggest Loser set in the country where Seria is. | answer_string_present, duplicate_or_paraphrase_candidate | TODO |
| The country where Seria is located is the same country where the version of The Biggest Loser was created. | none | TODO |
| The network that created the version of The Biggest Loser in the country where Seria is had a radio division where Ali Vincent was a prominent figure. | answer_string_present, duplicate_or_paraphrase_candidate | TODO |

