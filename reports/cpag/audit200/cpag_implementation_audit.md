# CPAG Implementation Audit

## Summary

- Sample queries: `200`
- Avg doc_id overlap@20: `8.84`
- Avg title overlap@20: `8.835`
- PropRAG top1 manual RRF ranks: `[1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 3, 1, 2, 1, 1, 1, 16, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 5, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 6, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 6, 2, 2, 1, 2, 1, 1, 1, 1, 12, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 7, 1, 1, 1, 1, 1, 2, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2]`
- All checks pass: `True`
- Failure count: `0`
- Warning count: `15`
- PropRAG top1 RRF rank > 3 count: `12`

## Failures

- None

## Warnings

- qid=1c0dd3b00bdc11eba7f7acde48001122: PropRAG top1 manual RRF rank is 4; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=7cb81afc0bd911eba7f7acde48001122: PropRAG top1 manual RRF rank is 16; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=a1d9a65c0bd911eba7f7acde48001122: PropRAG top1 manual RRF rank is 13; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=6b8402c40bdd11eba7f7acde48001122: PropRAG top1 manual RRF rank is 7; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=e7dc72940baf11ebab90acde48001122: PropRAG top1 manual RRF rank is 5; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=f1a794f20bdc11eba7f7acde48001122: PropRAG top1 manual RRF rank is 8; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=5cbb015f087511ebbd67ac1f6bf848b6: PropRAG top1 manual RRF rank is 6; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=5c7842780baf11ebab90acde48001122: PropRAG top1 manual RRF rank is 6; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=94198d580bb011ebab90acde48001122: PropRAG top1 manual RRF rank is 12; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=3ed2ba7a0bda11eba7f7acde48001122: PropRAG top1 manual RRF rank is 4; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=bbfe9c84087511ebbd67ac1f6bf848b6: score-only manual RRF top5 differs from implementation due to tie-break ordering
- qid=4f8e349e08e111ebbda2ac1f6bf848b6: PropRAG top1 manual RRF rank is 7; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=b8ef144a0bdd11eba7f7acde48001122: PropRAG top1 manual RRF rank is 5; this can happen when cross-pool shared docs outrank PropRAG-only docs
- qid=75668b40094111ebbdaeac1f6bf848b6: score-only manual RRF top5 differs from implementation due to tie-break ordering
- qid=8540ae66088011ebbd6cac1f6bf848b6: score-only manual RRF top5 differs from implementation due to tie-break ordering

## Sample Checks

### 83bf3b5a0bd911eba7f7acde48001122

- Question: When did Lothair Ii's mother die?
- doc_id overlap@20: `14`
- title overlap@20: `14`
- union size: `26`
- cross-pool docs: `14`
- PropRAG top1: `Lothair II`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Lothair II', 'Teutberga', 'Bertha, daughter of Lothair II', 'Waldrada of Lotharingia', 'Ermengarde of Tours']`
- Impl RRF top5: `['Lothair II', 'Teutberga', 'Bertha, daughter of Lothair II', 'Waldrada of Lotharingia', 'Ermengarde of Tours']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Lothair II', 'Teutberga', 'Bertha, daughter of Lothair II', 'Waldrada of Lotharingia', 'Ermengarde of Tours']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Lothair II` sources `['proprag', 'dense']` graph count `2`

### a80d84e7096d11ebbdb0ac1f6bf848b6

- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Aas Ka Panchhi`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Aas Ka Panchhi', 'Phoolwari', 'Aansoo Ban Gaye Phool', 'Kishore Sahu', 'The Warriors']`
- Impl RRF top5: `['Aas Ka Panchhi', 'Phoolwari', 'Aansoo Ban Gaye Phool', 'Kishore Sahu', 'The Warriors']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Aas Ka Panchhi', 'Phoolwari', 'Aansoo Ban Gaye Phool', 'John Jaffer Janardhanan', 'Kishore Sahu']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Aas Ka Panchhi` sources `['proprag', 'dense']` graph count `2`

### 2dc690ba0bdc11eba7f7acde48001122

- Question: What is the place of birth of the performer of song Changed It?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Changed It`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Changed It', 'You Changed Me', 'Andrew Allen (singer)', 'Place of birth', 'Carly Rae Jepsen']`
- Impl RRF top5: `['Changed It', 'You Changed Me', 'Andrew Allen (singer)', 'Place of birth', 'Carly Rae Jepsen']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Changed It', 'Carly Rae Jepsen', 'You Changed Me', "Did It On'em", 'Andrew Allen (singer)']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Changed It` sources `['proprag', 'dense']` graph count `2`

### 462bb642099211ebbdb0ac1f6bf848b6

- Question: Are Marufabad and Nasamkhrali both located in the same country?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Nasamkhrali`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Marufabad', 'Nasamkhrali', 'Where Was I', 'Qaleh-ye Bakhtiar', 'Qaleh-ye Zaras']`
- Impl RRF top5: `['Marufabad', 'Nasamkhrali', 'Where Was I', 'Qaleh-ye Bakhtiar', 'Qaleh-ye Zaras']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Marufabad', 'Nasamkhrali', 'Ampelakia, Evros', 'Qaleh-ye Bakhtiar', 'Radisele']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Marufabad` sources `['proprag', 'dense']` graph count `2`

### a1cdb240085811ebbd5bac1f6bf848b6

- Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Aldri annet enn bråk`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Aldri annet enn bråk', 'Edith Carlmar', "God's Gift to Women", 'Ingmar Bergman', 'Bjarne Henning-Jensen']`
- Impl RRF top5: `['Aldri annet enn bråk', 'Edith Carlmar', "God's Gift to Women", 'Ingmar Bergman', 'Bjarne Henning-Jensen']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Aldri annet enn bråk', 'Edith Carlmar', "God's Gift to Women", 'Altid ballade', 'Michael Curtiz']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Aldri annet enn bråk` sources `['proprag', 'dense']` graph count `2`

### 6718770a087311ebbd66ac1f6bf848b6

- Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `The Heart of Doreon`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Heart of Doreon', 'El Tonto', 'Heart (1987 film)', 'Juan Bustillo Oro', 'Leopoldo Torre Nilsson']`
- Impl RRF top5: `['The Heart of Doreon', 'El Tonto', 'Heart (1987 film)', 'Juan Bustillo Oro', 'Leopoldo Torre Nilsson']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Heart of Doreon', 'El Tonto', 'Heart (1987 film)', 'Robert North Bradbury', 'Juan Bustillo Oro']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Heart of Doreon` sources `['proprag', 'dense']` graph count `2`

### 7f7046d308f711ebbdaaac1f6bf848b6

- Question: Who was born first out of Aivar Kuusmaa and Andy Summers?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Andy Summers`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Andy Summers', 'Aivar Kuusmaa', 'Dugès', 'Jonny Greenwood', 'Svante Stensson Sture']`
- Impl RRF top5: `['Andy Summers', 'Aivar Kuusmaa', 'Dugès', 'Jonny Greenwood', 'Svante Stensson Sture']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Andy Summers', 'Aivar Kuusmaa', 'Dugès', 'Jonny Greenwood', 'Where Was I']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Andy Summers` sources `['proprag', 'dense']` graph count `2`

### 9fe6a6760baf11ebab90acde48001122

- Question: Who is Raghnall Mac Ruaidhrí's paternal grandfather?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Raghnall Mac Ruaidhrí`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Raghnall Mac Ruaidhrí', 'Ruaidhrí Mac Ruaidhrí', 'Dáire Drechlethan', 'Takayama Tomoteru', 'Comgall mac Domangairt']`
- Impl RRF top5: `['Raghnall Mac Ruaidhrí', 'Ruaidhrí Mac Ruaidhrí', 'Dáire Drechlethan', 'Takayama Tomoteru', 'Comgall mac Domangairt']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Raghnall Mac Ruaidhrí', 'Ruaidhrí Mac Ruaidhrí', 'Dáire Drechlethan', 'Takayama Tomoteru', 'Amy of Garmoran']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Raghnall Mac Ruaidhrí` sources `['proprag', 'dense']` graph count `2`

### 551e024408a611ebbd7fac1f6bf848b6

- Question: Do both films Interview With A Hitman and The Last Coupon have the directors from the same country?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `The Last Coupon`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Interview with a Hitman', 'The Last Coupon', 'Perry Bhandal', 'Spring Handicap', 'The Umbrella Coup']`
- Impl RRF top5: `['Interview with a Hitman', 'The Last Coupon', 'Perry Bhandal', 'Spring Handicap', 'The Umbrella Coup']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Interview with a Hitman', 'The Last Coupon', 'Perry Bhandal', 'Spring Handicap', 'The Umbrella Coup']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Interview with a Hitman` sources `['proprag', 'dense']` graph count `2`

### 33f51d7e0bde11eba7f7acde48001122

- Question: What nationality is the director of film Blood Street?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Blood Street`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Blood Street', 'Jackie Kong', 'Wisit Sasanatieng', 'Claude Weisz', 'Lasse Hallström']`
- Impl RRF top5: `['Blood Street', 'Jackie Kong', 'Wisit Sasanatieng', 'Claude Weisz', 'Lasse Hallström']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Blood Street', 'Lasse Hallström', 'Leo Fong', 'Tiger Cage (film)', 'Wisit Sasanatieng']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Blood Street` sources `['proprag', 'dense']` graph count `2`

### 914b452c0bdc11eba7f7acde48001122

- Question: What is the place of birth of the director of film Gaby: A True Story?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Luis Mandoki`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Gaby: A True Story', 'Luis Mandoki', 'Claude Weisz', 'Rod Hardy', 'Sepideh Farsi']`
- Impl RRF top5: `['Gaby: A True Story', 'Luis Mandoki', 'Claude Weisz', 'Rod Hardy', 'Sepideh Farsi']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Gaby: A True Story', 'Luis Mandoki', 'Claude Weisz', 'Rod Hardy', 'Pál Gábor']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Gaby: A True Story` sources `['proprag', 'dense']` graph count `2`

### d6898f78089511ebbd75ac1f6bf848b6

- Question: Are Vasilyevsky Island and Preobrazheniya Island located in the same country?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Preobrazheniya Island`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Preobrazheniya Island', 'Vasilyevsky Island', 'Abdul-Vahed Niyazov', 'Ingmarsö', 'Izmalkovo, Lipetsk Oblast']`
- Impl RRF top5: `['Preobrazheniya Island', 'Vasilyevsky Island', 'Abdul-Vahed Niyazov', 'Ingmarsö', 'Izmalkovo, Lipetsk Oblast']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Preobrazheniya Island', 'Vasilyevsky Island', 'Abdul-Vahed Niyazov', 'Michał Przysiężny', 'Ingmarsö']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Preobrazheniya Island` sources `['proprag', 'dense']` graph count `2`

### 006d81bc0bde11eba7f7acde48001122

- Question: What nationality is the performer of song When The Stars Go Blue?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `When the Stars Go Blue`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['When the Stars Go Blue', 'When I Was Young', 'Where Was I', 'Leaving You', 'Here to Stay']`
- Impl RRF top5: `['When the Stars Go Blue', 'When I Was Young', 'Where Was I', 'Leaving You', 'Here to Stay']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['When the Stars Go Blue', 'Something Worth Leaving Behind (song)', 'Forever Everyday', "You've Got a Good Love Comin' (song)", 'When I Was Young']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `When the Stars Go Blue` sources `['proprag', 'dense']` graph count `2`

### f5e3b9ca0bdb11eba7f7acde48001122

- Question: Who is the child of the performer of song Me And Bobby Mcgee?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Me and Bobby McGee`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Me and Bobby McGee', 'Fred Foster', 'Janis Joplin', 'Daughter of Destiny', "List of people named O'Grady"]`
- Impl RRF top5: `['Me and Bobby McGee', 'Fred Foster', 'Janis Joplin', 'Daughter of Destiny', "List of people named O'Grady"]`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Me and Bobby McGee', 'Fred Foster', 'Janis Joplin', 'Roger Miller', 'Semi-Tough']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Me and Bobby McGee` sources `['proprag', 'dense']` graph count `2`

### 265daf200bdc11eba7f7acde48001122

- Question: Where was the place of death of Maurice, Prince Of Orange's father?
- doc_id overlap@20: `14`
- title overlap@20: `14`
- union size: `26`
- cross-pool docs: `14`
- PropRAG top1: `Maurice, Prince of Orange`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Maurice, Prince of Orange', 'William the Silent', 'William I, Count of Nassau-Dillenburg', 'William of Nassau (1601–1627)', 'Louis of Nassau, Lord of De Lek and Beverweerd']`
- Impl RRF top5: `['Maurice, Prince of Orange', 'William the Silent', 'William I, Count of Nassau-Dillenburg', 'William of Nassau (1601–1627)', 'Louis of Nassau, Lord of De Lek and Beverweerd']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Maurice, Prince of Orange', 'Louis of Nassau, Lord of De Lek and Beverweerd', 'The Private Life of Louis XIV', 'William of Nassau (1601–1627)', 'William the Silent']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Maurice, Prince of Orange` sources `['proprag', 'dense']` graph count `2`

### 1c0dd3b00bdc11eba7f7acde48001122

- Question: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Stanisław Koniecpolski`
- PropRAG top1 manual RRF rank: `4`
- Manual RRF top5: `['Aleksander Koniecpolski (1555–1609)', 'Aleksander Koniecpolski (1620–1659)', 'Takayama Tomoteru', 'Stanisław Koniecpolski', 'Dugès']`
- Impl RRF top5: `['Aleksander Koniecpolski (1555–1609)', 'Aleksander Koniecpolski (1620–1659)', 'Takayama Tomoteru', 'Stanisław Koniecpolski', 'Dugès']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Aleksander Koniecpolski (1555–1609)', 'Stanisław Koniecpolski', 'Takayama Tomoteru', 'Aleksander Koniecpolski (1620–1659)', 'Marek Sobieski']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Aleksander Koniecpolski (1555–1609)` sources `['proprag', 'dense']` graph count `2`

### bab3b6d00bda11eba7f7acde48001122

- Question: What is the date of death of the director of film Madame La Presidente?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Madame la Presidente`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Madame la Presidente', 'Édouard Molinaro', 'Claude Sautet', 'Jean Grémillon', 'Claude Autant-Lara']`
- Impl RRF top5: `['Madame la Presidente', 'Édouard Molinaro', 'Claude Sautet', 'Jean Grémillon', 'Claude Autant-Lara']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Madame la Presidente', 'Édouard Molinaro', 'Frank Lloyd', 'Claude Sautet', 'Jean Grémillon']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Madame la Presidente` sources `['proprag', 'dense']` graph count `2`

### 84b691d8086a11ebbd5fac1f6bf848b6

- Question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Wrong Turn 5: Bloodlines`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Dark River (2017 film)', 'Wrong Turn 5: Bloodlines', "Declan O'Brien", 'Wrong Turn: The Foundation', 'Dark River (1990 film)']`
- Impl RRF top5: `['Dark River (2017 film)', 'Wrong Turn 5: Bloodlines', "Declan O'Brien", 'Wrong Turn: The Foundation', 'Dark River (1990 film)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dark River (2017 film)', 'Wrong Turn 5: Bloodlines', "Declan O'Brien", 'Wrong Turn: The Foundation', 'Rufus Norris']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Dark River (2017 film)` sources `['proprag', 'dense']` graph count `2`

### dcbee4b608b011ebbd85ac1f6bf848b6

- Question: Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `The Goose Woman`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Goose Woman', 'You Can No Longer Remain Silent', 'The Past of Mary Holmes', 'Michael Powell', 'Helmut Käutner']`
- Impl RRF top5: `['The Goose Woman', 'You Can No Longer Remain Silent', 'The Past of Mary Holmes', 'Michael Powell', 'Helmut Käutner']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Goose Woman', 'You Can No Longer Remain Silent', 'The Past of Mary Holmes', 'Michael Powell', 'Robert A. Stemmle']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Goose Woman` sources `['proprag', 'dense']` graph count `2`

### 2d8f3ebe0bda11eba7f7acde48001122

- Question: Where was the director of film The Private Life Of Cinema born?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `The Private Life of Cinema`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Private Life of Cinema', 'Peter Greenaway', 'Michael Powell', 'Alexander Korda', 'Bertrand Tavernier']`
- Impl RRF top5: `['The Private Life of Cinema', 'Peter Greenaway', 'Michael Powell', 'Alexander Korda', 'Bertrand Tavernier']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Private Life of Cinema', 'Michael Powell', 'Peter Greenaway', 'Denys Desjardins', 'Alexander Korda']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Private Life of Cinema` sources `['proprag', 'dense']` graph count `2`

### e6e38bc40bdd11eba7f7acde48001122

- Question: Where did Coulson Wallop's father study?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Coulson Wallop`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Coulson Wallop', 'John Wallop, 2nd Earl of Portsmouth', 'Takayama Tomoteru', 'Anacyndaraxes', 'Alexander Baring, 4th Baron Ashburton']`
- Impl RRF top5: `['Coulson Wallop', 'John Wallop, 2nd Earl of Portsmouth', 'Takayama Tomoteru', 'Anacyndaraxes', 'Alexander Baring, 4th Baron Ashburton']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Coulson Wallop', 'John Wallop, 2nd Earl of Portsmouth', 'John Stuart, Lord Mount Stuart', 'George Rodney, 2nd Baron Rodney', 'Takayama Tomoteru']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Coulson Wallop` sources `['proprag', 'dense']` graph count `2`

### 0bedb7e80bdc11eba7f7acde48001122

- Question: Why did John Middleton Murry's wife die?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `John Middleton Murry`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['John Middleton Murry', 'John Middleton Murry Jr.', 'Katherine Mansfield', 'Anne Estelle Rice', 'Problem of pain']`
- Impl RRF top5: `['John Middleton Murry', 'John Middleton Murry Jr.', 'Katherine Mansfield', 'Anne Estelle Rice', 'Problem of pain']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['John Middleton Murry', 'John Middleton Murry Jr.', 'Katherine Mansfield', 'Anne Estelle Rice', 'Problem of pain']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `John Middleton Murry` sources `['proprag', 'dense']` graph count `2`

### 86508dc60bdc11eba7f7acde48001122

- Question: Where was the composer of film Billy Elliot born?
- doc_id overlap@20: `17`
- title overlap@20: `17`
- union size: `23`
- cross-pool docs: `17`
- PropRAG top1: `Stephen Warbeck`
- PropRAG top1 manual RRF rank: `3`
- Manual RRF top5: `['Billy Elliot', 'Lee Hall (playwright)', 'Stephen Warbeck', 'Billy Elliot the Musical Live', 'Walter Ulfig']`
- Impl RRF top5: `['Billy Elliot', 'Lee Hall (playwright)', 'Stephen Warbeck', 'Billy Elliot the Musical Live', 'Walter Ulfig']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Billy Elliot', 'Stephen Warbeck', 'Lee Hall (playwright)', 'Billy Elliot the Musical Live', 'Walter Ulfig']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Billy Elliot` sources `['proprag', 'dense']` graph count `2`

### e311a3ba0bdc11eba7f7acde48001122

- Question: What is the place of birth of Lisbeth Palme's husband?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Lisbeth Palme`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Lisbeth Palme', 'Olof Palme', 'Place of birth', 'Sven Ulric Palme', 'Lars Eliasson']`
- Impl RRF top5: `['Lisbeth Palme', 'Olof Palme', 'Place of birth', 'Sven Ulric Palme', 'Lars Eliasson']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Lisbeth Palme', 'Olof Palme', 'Lars Eliasson', 'Place of birth', 'Sven Ulric Palme']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Lisbeth Palme` sources `['proprag', 'dense']` graph count `2`

### bdd526c4091811ebbdaeac1f6bf848b6

- Question: Which film came out first, 3 Dots or Dying God?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Dying God`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['3 Dots', 'Dying God', 'Dots', 'Great God Gold', "God's Comedy"]`
- Impl RRF top5: `['3 Dots', 'Dying God', 'Dots', 'Great God Gold', "God's Comedy"]`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['3 Dots', 'Dying God', 'Dots', 'Killed the Family and Went to the Movies', 'Great God Gold']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `3 Dots` sources `['proprag', 'dense']` graph count `2`

### 435f65fa0baf11ebab90acde48001122

- Question: Who is the father-in-law of Sisowath Kossamak?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Sisowath Kossamak`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sisowath Kossamak', 'Norodom Suramarit', 'Obata Toramori', 'Anacyndaraxes', 'Takayama Tomoteru']`
- Impl RRF top5: `['Sisowath Kossamak', 'Norodom Suramarit', 'Obata Toramori', 'Anacyndaraxes', 'Takayama Tomoteru']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sisowath Kossamak', 'Norodom Suramarit', 'Obata Toramori', 'Anacyndaraxes', 'Takayama Tomoteru']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Sisowath Kossamak` sources `['proprag', 'dense']` graph count `2`

### 5cc01ab60bb011ebab90acde48001122

- Question: Who is Mugain's mother-in-law?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Mugain`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Mugain', 'Minamoto no Chikako', 'Maria Thins', 'Dáire Drechlethan', 'Doria Ragland']`
- Impl RRF top5: `['Mugain', 'Minamoto no Chikako', 'Maria Thins', 'Dáire Drechlethan', 'Doria Ragland']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Mugain', 'Minamoto no Chikako', 'Conchobar mac Nessa', 'Vera Miletić', 'Maria Thins']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Mugain` sources `['proprag', 'dense']` graph count `2`

### 2084260e0bde11eba7f7acde48001122

- Question: Where did Theodore Salisbury Woolsey's father study?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Theodore Salisbury Woolsey`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Theodore Salisbury Woolsey', 'Theodore Dwight Woolsey', 'Theodore Salisbury Woolsey Jr.', 'Takayama Tomoteru', 'Anacyndaraxes']`
- Impl RRF top5: `['Theodore Salisbury Woolsey', 'Theodore Dwight Woolsey', 'Theodore Salisbury Woolsey Jr.', 'Takayama Tomoteru', 'Anacyndaraxes']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Theodore Salisbury Woolsey', 'Theodore Dwight Woolsey', 'Theodore Salisbury Woolsey Jr.', 'Sidney Mason Stone', 'Hywel Teifi Edwards']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Theodore Salisbury Woolsey` sources `['proprag', 'dense']` graph count `2`

### 7cb81afc0bd911eba7f7acde48001122

- Question: What is the place of birth of the director of film The Return Of Swamp Thing?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Jim Wynorski`
- PropRAG top1 manual RRF rank: `16`
- Manual RRF top5: `['The Return of Swamp Thing', 'Swamp Thing (film)', 'Rod Hardy', 'Claude Weisz', 'Dick Durock']`
- Impl RRF top5: `['The Return of Swamp Thing', 'Swamp Thing (film)', 'Rod Hardy', 'Claude Weisz', 'Dick Durock']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Return of Swamp Thing', 'Jim Wynorski', 'Rod Hardy', 'Swamp Thing (film)', 'Claude Weisz']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Return of Swamp Thing` sources `['proprag', 'dense']` graph count `2`

### a1d9a65c0bd911eba7f7acde48001122

- Question: Where was the place of death of the performer of song I Can'T See Myself Leaving You?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `28`
- cross-pool docs: `11`
- PropRAG top1: `Aretha Franklin`
- PropRAG top1 manual RRF rank: `13`
- Manual RRF top5: `["I Can't See Myself Leaving You", 'Leaving You', 'Where Was I', 'Where Are You', 'Here to Stay']`
- Impl RRF top5: `["I Can't See Myself Leaving You", 'Leaving You', 'Where Was I', 'Where Are You', 'Here to Stay']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `["I Can't See Myself Leaving You", 'Aretha Franklin', 'Leaving You', 'Where Was I', 'Chuck Berry']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `I Can't See Myself Leaving You` sources `['proprag', 'dense']` graph count `2`

### f34d188608cb11ebbd93ac1f6bf848b6

- Question: Which film has the director who was born later, Playing It Wild or I'Ll Be Going Now?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Playing It Wild`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Playing It Wild', "I'll Be Going Now", 'Victor Sjöström', 'Edith Carlmar', 'Wild Rovers']`
- Impl RRF top5: `['Playing It Wild', "I'll Be Going Now", 'Victor Sjöström', 'Edith Carlmar', 'Wild Rovers']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Playing It Wild', "I'll Be Going Now", 'William Duncan (actor)', 'Victor Sjöström', 'Francis J. Grandon']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Playing It Wild` sources `['proprag', 'dense']` graph count `2`

### 3a3c2efe0bdc11eba7f7acde48001122

- Question: What nationality is Beatrice I, Countess Of Burgundy's husband?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Otto I, Count of Burgundy`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Otto I, Count of Burgundy', 'Beatrice I, Countess of Burgundy', 'Beatrice of Navarre, Duchess of Burgundy', 'Sibylla of Burgundy, Duchess of Burgundy', 'Beatrice of Viennois']`
- Impl RRF top5: `['Otto I, Count of Burgundy', 'Beatrice I, Countess of Burgundy', 'Beatrice of Navarre, Duchess of Burgundy', 'Sibylla of Burgundy, Duchess of Burgundy', 'Beatrice of Viennois']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Beatrice I, Countess of Burgundy', 'Otto I, Count of Burgundy', 'Beatrice of Navarre, Duchess of Burgundy', 'Frederick V, Duke of Swabia', 'Beatrice of Viennois']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Beatrice I, Countess of Burgundy` sources `['proprag', 'dense']` graph count `2`

### 9aa8008408de11ebbd9eac1f6bf848b6

- Question: Which film has the director born later, Christ Walking On The Water or 45 Fathers?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Christ Walking on the Water`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Christ Walking on the Water', '45 Fathers', 'Bertrand Tavernier', 'Werner Herzog', 'Christian E. Christiansen']`
- Impl RRF top5: `['Christ Walking on the Water', '45 Fathers', 'Bertrand Tavernier', 'Werner Herzog', 'Christian E. Christiansen']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Christ Walking on the Water', '45 Fathers', 'Bertrand Tavernier', 'Georges Méliès', 'James Tinling']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Christ Walking on the Water` sources `['proprag', 'dense']` graph count `2`

### 0dfe41f60bdc11eba7f7acde48001122

- Question: Where was the performer of song Come Dance With Me (Song) born?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Come Dance with Me (song)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Come Dance with Me (song)', 'Place of birth', 'Where Was I', 'Dance with Me (Zoli Ádok song)', 'Dance with Me (Kelly Clarkson song)']`
- Impl RRF top5: `['Come Dance with Me (song)', 'Place of birth', 'Where Was I', 'Dance with Me (Zoli Ádok song)', 'Dance with Me (Kelly Clarkson song)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Come Dance with Me (song)', 'Dance with Me (Zoli Ádok song)', 'Place of birth', 'The Tender Trap (film)', 'Where Was I']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Come Dance with Me (song)` sources `['proprag', 'dense']` graph count `2`

### 3ce92df80bde11eba7f7acde48001122

- Question: Where does the director of film Talk About A Stranger work at?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Talk About a Stranger`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Talk About a Stranger', 'Stranger in the House', 'Stranger in Town', 'Anthony Mann', 'G. Marthandan']`
- Impl RRF top5: `['Talk About a Stranger', 'Stranger in the House', 'Stranger in Town', 'Anthony Mann', 'G. Marthandan']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Talk About a Stranger', 'Anthony Mann', 'Stranger in the Mirror', 'The Stranger', 'Stranger in the House']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Talk About a Stranger` sources `['proprag', 'dense']` graph count `2`

### 3e3cd62a086211ebbd5dac1f6bf848b6

- Question: Who was born later, Gideon Johnson Pillow or Holm Jølsen?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Gideon Johnson Pillow`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Gideon Johnson Pillow', 'Holm Jølsen', 'Ola Isene', 'J. Neely Johnson', 'Thaddeus P. Mott']`
- Impl RRF top5: `['Gideon Johnson Pillow', 'Holm Jølsen', 'Ola Isene', 'J. Neely Johnson', 'Thaddeus P. Mott']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Gideon Johnson Pillow', 'Holm Jølsen', 'J. Neely Johnson', 'Ola Isene', 'Thaddeus P. Mott']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Gideon Johnson Pillow` sources `['proprag', 'dense']` graph count `2`

### b672c1ee0bdd11eba7f7acde48001122

- Question: What is the place of birth of the composer of film Inherent Vice (Film)?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Inherent Vice (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Inherent Vice (film)', 'Jonny Greenwood', 'Walter Ulfig', 'Thomas Morse', 'Abe Meyer']`
- Impl RRF top5: `['Inherent Vice (film)', 'Jonny Greenwood', 'Walter Ulfig', 'Thomas Morse', 'Abe Meyer']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Inherent Vice (film)', 'Jonny Greenwood', 'Walter Ulfig', 'Melora Walters', 'Thomas Morse']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Inherent Vice (film)` sources `['proprag', 'dense']` graph count `2`

### 01309e5008bd11ebbd89ac1f6bf848b6

- Question: Which film whose director is younger, Dangerously They Live or Salad By The Roots?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Salad by the Roots`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Salad by the Roots', 'Dangerously They Live', 'Werner Herzog', 'Julie Dash', 'Roger Corman']`
- Impl RRF top5: `['Salad by the Roots', 'Dangerously They Live', 'Werner Herzog', 'Julie Dash', 'Roger Corman']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Salad by the Roots', 'Dangerously They Live', 'Werner Herzog', 'Georges Lautner', 'Julie Dash']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Salad by the Roots` sources `['proprag', 'dense']` graph count `2`

### c5a00e860bda11eba7f7acde48001122

- Question: Where did Sylvia Burka's husband die?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Sylvia Burka`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sylvia Burka', 'Valley of Death', 'Where Was I', 'Stan Marks', 'Victor Janson']`
- Impl RRF top5: `['Sylvia Burka', 'Valley of Death', 'Where Was I', 'Stan Marks', 'Victor Janson']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sylvia Burka', 'Victor Janson', 'Jocelyn Lovell', 'Valley of Death', 'Where Was I']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Sylvia Burka` sources `['proprag', 'dense']` graph count `2`

### 434dea7708d211ebbd95ac1f6bf848b6

- Question: Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Pamela Ann Rymer`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Pamela Ann Rymer', 'Ludwig Elsbett', 'Dugès', 'Pamela Bach', 'Archduchess Maria Elisabeth of Austria (1737–1740)']`
- Impl RRF top5: `['Pamela Ann Rymer', 'Ludwig Elsbett', 'Dugès', 'Pamela Bach', 'Archduchess Maria Elisabeth of Austria (1737–1740)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Pamela Ann Rymer', 'Ludwig Elsbett', 'Dugès', 'Pamela A. Barker', 'Pamela Harris (judge)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Pamela Ann Rymer` sources `['proprag', 'dense']` graph count `2`

### 02b62fce087211ebbd64ac1f6bf848b6

- Question: Which film whose director is younger, Phalitamsha or Gladiators Seven?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Gladiators Seven`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Gladiators Seven', 'Phalitamsha', 'Hrishikesh Mukherjee', 'Alberto De Martino', 'Mario Bava']`
- Impl RRF top5: `['Gladiators Seven', 'Phalitamsha', 'Hrishikesh Mukherjee', 'Alberto De Martino', 'Mario Bava']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Gladiators Seven', 'Phalitamsha', 'Alberto De Martino', 'The Warriors', 'Puttanna Kanagal']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Gladiators Seven` sources `['proprag', 'dense']` graph count `2`

### b0ac0fc8087b11ebbd68ac1f6bf848b6

- Question: Which film has the director who was born later, Henry Goes Arizona or The Blue Collar Worker And The Hairdresser In A Whirl Of Sex And Politics?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `The Blue Collar Worker and the Hairdresser in a Whirl of Sex and Politics`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Blue Collar Worker and the Hairdresser in a Whirl of Sex and Politics', 'Henry Goes Arizona', 'The Boss and the Worker', 'Henry King (director)', 'Henry Hathaway']`
- Impl RRF top5: `['The Blue Collar Worker and the Hairdresser in a Whirl of Sex and Politics', 'Henry Goes Arizona', 'The Boss and the Worker', 'Henry King (director)', 'Henry Hathaway']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Blue Collar Worker and the Hairdresser in a Whirl of Sex and Politics', 'Henry Goes Arizona', 'Henry King (director)', 'Lina Wertmüller', 'The Boss and the Worker']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Blue Collar Worker and the Hairdresser in a Whirl of Sex and Politics` sources `['proprag', 'dense']` graph count `2`

### c9a769c608be11ebbd8aac1f6bf848b6

- Question: Who was born first, Hywel Teifi Edwards or Felix Schuster?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Felix Schuster`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Felix Schuster', 'Hywel Teifi Edwards', 'Dugès', 'Harold D. Schuster', 'Alexander Fuks']`
- Impl RRF top5: `['Felix Schuster', 'Hywel Teifi Edwards', 'Dugès', 'Harold D. Schuster', 'Alexander Fuks']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Felix Schuster', 'Hywel Teifi Edwards', 'Alexander Fuks', 'Dugès', 'Harold D. Schuster']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Felix Schuster` sources `['proprag', 'dense']` graph count `2`

### 3f04140e08c311ebbd8dac1f6bf848b6

- Question: Which film has the director who was born first, Tombstone Rashomon or Waiting For The Clouds?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Tombstone Rashomon`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Tombstone Rashomon', 'Waiting for the Clouds', 'Bertrand Tavernier', 'Anthony Mann', 'Sergio Leone']`
- Impl RRF top5: `['Tombstone Rashomon', 'Waiting for the Clouds', 'Bertrand Tavernier', 'Anthony Mann', 'Sergio Leone']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Tombstone Rashomon', 'Waiting for the Clouds', 'Bertrand Tavernier', 'Alex Cox', 'Sergio Leone']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Tombstone Rashomon` sources `['proprag', 'dense']` graph count `2`

### 50e4a65c0bde11eba7f7acde48001122

- Question: Where was the performer of song B Boy (Song) detained?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `B Boy (song)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['B Boy (song)', 'Where Was I', 'J-Flexx', 'Juice Aleem', 'Let Me Out']`
- Impl RRF top5: `['B Boy (song)', 'Where Was I', 'J-Flexx', 'Juice Aleem', 'Let Me Out']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['B Boy (song)', 'Where Was I', 'Meek Mill', 'J-Flexx', 'Aretha Franklin']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `B Boy (song)` sources `['proprag', 'dense']` graph count `2`

### b64f541e088d11ebbd70ac1f6bf848b6

- Question: Which film has the director who was born later, Illusions (1982 Film) or It'S A Wonderful Afterlife?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Illusions (1982 film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Illusions (1982 film)', "It's a Wonderful Afterlife", 'Gurinder Chadha', 'Georges Méliès', 'Goldy Notay']`
- Impl RRF top5: `['Illusions (1982 film)', "It's a Wonderful Afterlife", 'Gurinder Chadha', 'Georges Méliès', 'Goldy Notay']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Illusions (1982 film)', 'Goldy Notay', "It's a Wonderful Afterlife", 'Gurinder Chadha', 'Georges Méliès']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Illusions (1982 film)` sources `['proprag', 'dense']` graph count `2`

### f998c58c091b11ebbdaeac1f6bf848b6

- Question: Are both movies, Naked Tango and Algiers (Film), from the same country?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Algiers (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Algiers (film)', 'Naked Tango', 'The Lights of Buenos Aires', 'To the Heart', 'Manuel Romero']`
- Impl RRF top5: `['Algiers (film)', 'Naked Tango', 'The Lights of Buenos Aires', 'To the Heart', 'Manuel Romero']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Algiers (film)', 'Naked Tango', 'The Lights of Buenos Aires', 'List of Uruguayan films', 'To the Heart']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Algiers (film)` sources `['proprag', 'dense']` graph count `2`

### 38da155f087511ebbd67ac1f6bf848b6

- Question: Who lived longer, Sir Thomas Wheate, 1St Baronet or Albert Édouard Le Brethon De Caligny?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Albert Édouard Le Brethon de Caligny`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Albert Édouard Le Brethon de Caligny', 'Sir Thomas Wheate, 1st Baronet', 'Sir Thomas Wheate, 2nd Baronet', 'Sir Thomas Beaumont, 1st Baronet', 'Sir Thomas Southwell, 1st Baronet']`
- Impl RRF top5: `['Albert Édouard Le Brethon de Caligny', 'Sir Thomas Wheate, 1st Baronet', 'Sir Thomas Wheate, 2nd Baronet', 'Sir Thomas Beaumont, 1st Baronet', 'Sir Thomas Southwell, 1st Baronet']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Albert Édouard Le Brethon de Caligny', 'Sir Thomas Wheate, 1st Baronet', 'Sir Thomas Wheate, 2nd Baronet', 'Sir Thomas Beaumont, 1st Baronet', 'Dugès']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Albert Édouard Le Brethon de Caligny` sources `['proprag', 'dense']` graph count `2`

### 49b22d0a0bde11eba7f7acde48001122

- Question: Where did the composer of film The Straw Hat die?
- doc_id overlap@20: `14`
- title overlap@20: `14`
- union size: `26`
- cross-pool docs: `14`
- PropRAG top1: `The Straw Hat`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Straw Hat', 'Abe Meyer', 'Bert Grund', 'Oscar Straus (composer)', 'Amedeo Escobar']`
- Impl RRF top5: `['The Straw Hat', 'Abe Meyer', 'Bert Grund', 'Oscar Straus (composer)', 'Amedeo Escobar']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Straw Hat', 'Abe Meyer', 'Thulasi (1987 film)', 'Bert Grund', 'Vijaya Bhaskar']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Straw Hat` sources `['proprag', 'dense']` graph count `2`

### 78b10fb00bdc11eba7f7acde48001122

- Question: Where does the director of film A Nest Of Noblemen work at?
- doc_id overlap@20: `2`
- title overlap@20: `2`
- union size: `38`
- cross-pool docs: `2`
- PropRAG top1: `A Nest of Noblemen`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['A Nest of Noblemen', 'Ah Nian', 'G. Marthandan', 'Vladimir Gardin', 'Ringo-en no shōjo']`
- Impl RRF top5: `['A Nest of Noblemen', 'Ah Nian', 'G. Marthandan', 'Vladimir Gardin', 'Ringo-en no shōjo']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['A Nest of Noblemen', 'Vladimir Gardin', 'Ah Nian', 'Alfred Hitchcock', 'Édouard Niermans (director)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `A Nest of Noblemen` sources `['proprag', 'dense']` graph count `2`

### 879d21e208fc11ebbdadac1f6bf848b6

- Question: Were both Pietro Salini and Domenico Ravenna, born in the same place?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Pietro Salini`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Pietro Salini', 'Domenico Ravenna', 'Pierfelice Ravenna', 'Dugès', 'Gino Ravenna']`
- Impl RRF top5: `['Pietro Salini', 'Domenico Ravenna', 'Pierfelice Ravenna', 'Dugès', 'Gino Ravenna']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Pietro Salini', 'Domenico Ravenna', 'Pierfelice Ravenna', 'Dugès', 'Quintus Salvidienus Rufus']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Pietro Salini` sources `['proprag', 'dense']` graph count `2`

### 62a9cd640bb011ebab90acde48001122

- Question: Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `John Ernest II, Duke of Saxe-Weimar`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['John Ernest II, Duke of Saxe-Weimar', 'John George II, Duke of Saxe-Eisenach', 'John William III, Duke of Saxe-Eisenach', 'William Ernest, Duke of Saxe-Weimar', 'William, Duke of Saxe-Weimar']`
- Impl RRF top5: `['John Ernest II, Duke of Saxe-Weimar', 'John George II, Duke of Saxe-Eisenach', 'John William III, Duke of Saxe-Eisenach', 'William Ernest, Duke of Saxe-Weimar', 'William, Duke of Saxe-Weimar']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['John Ernest II, Duke of Saxe-Weimar', 'John George II, Duke of Saxe-Eisenach', 'John William III, Duke of Saxe-Eisenach', 'William Ernest, Duke of Saxe-Weimar', 'Anna Dorothea, Abbess of Quedlinburg']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `John Ernest II, Duke of Saxe-Weimar` sources `['proprag', 'dense']` graph count `2`

### cda459160bda11eba7f7acde48001122

- Question: Who is the father of the director of film Palo Alto (2013 Film)?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Gia Coppola`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Gia Coppola', 'Palo Alto (2013 film)', 'Jack Kilmer', 'Lars Eliasson', 'Zoe Levin']`
- Impl RRF top5: `['Gia Coppola', 'Palo Alto (2013 film)', 'Jack Kilmer', 'Lars Eliasson', 'Zoe Levin']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Gia Coppola', 'Palo Alto (2013 film)', 'Zoe Levin', 'Jack Kilmer', "Stella's Favor"]`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Gia Coppola` sources `['proprag', 'dense']` graph count `2`

### 1dfaa6200bdd11eba7f7acde48001122

- Question: Who is the child of the director of film Los Pagares De Mendieta?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Efren Reyes Sr.`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Los Pagares de Mendieta', 'Efren Reyes Sr.', 'Gracia Querejeta', 'Leopoldo Torres Ríos', 'Juan Bustillo Oro']`
- Impl RRF top5: `['Los Pagares de Mendieta', 'Efren Reyes Sr.', 'Gracia Querejeta', 'Leopoldo Torres Ríos', 'Juan Bustillo Oro']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Efren Reyes Sr.', 'Los Pagares de Mendieta', 'Leopoldo Torre Nilsson', 'Leopoldo Torres Ríos', 'King Baggot (cinematographer)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Efren Reyes Sr.` sources `['proprag', 'dense']` graph count `2`

### ea151f3908e011ebbda2ac1f6bf848b6

- Question: Who lived longer, Hubert Mordek or Levon Ashotovich Grigorian?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Levon Ashotovich Grigorian`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Levon Ashotovich Grigorian', 'Hubert Mordek', 'Peter Gravesen', 'Dugès', 'Albert Mkrtchyan']`
- Impl RRF top5: `['Levon Ashotovich Grigorian', 'Hubert Mordek', 'Peter Gravesen', 'Dugès', 'Albert Mkrtchyan']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Levon Ashotovich Grigorian', 'Hubert Mordek', 'Peter Gravesen', 'Alexina Duchamp', 'Michael Glatthaar']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Levon Ashotovich Grigorian` sources `['proprag', 'dense']` graph count `2`

### 8e07f1f00bda11eba7f7acde48001122

- Question: Who is the spouse of the director of film My Three Merry Widows?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `My Three Merry Widows`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['My Three Merry Widows', 'The Very Merry Widows', 'Sherry Hormann', 'Geoff Murphy', 'My Wife Is Being Stupid']`
- Impl RRF top5: `['My Three Merry Widows', 'The Very Merry Widows', 'Sherry Hormann', 'Geoff Murphy', 'My Wife Is Being Stupid']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['My Three Merry Widows', 'Geoff Murphy', 'The Very Merry Widows', 'Fernando Cortés', 'Sherry Hormann']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `My Three Merry Widows` sources `['proprag', 'dense']` graph count `2`

### b7f5f7200bdd11eba7f7acde48001122

- Question: Which country the composer of film Thunder On The Hill is from?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `Thunder on the Hill`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Thunder on the Hill', 'Walter Ulfig', 'Bert Grund', 'Abe Meyer', 'Hans J. Salter']`
- Impl RRF top5: `['Thunder on the Hill', 'Walter Ulfig', 'Bert Grund', 'Abe Meyer', 'Hans J. Salter']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Thunder on the Hill', 'Walter Ulfig', 'Hans J. Salter', 'Bert Grund', 'Abe Meyer']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Thunder on the Hill` sources `['proprag', 'dense']` graph count `2`

### 0dc9691b087911ebbd67ac1f6bf848b6

- Question: Which film has the director born later, Arrête Ton Cinéma or Agni (2004 Film)?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Arrête ton cinéma`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Agni (2004 film)', 'Arrête ton cinéma', 'Bertrand Tavernier', 'Assassination (1964 film)', 'Garçon stupide']`
- Impl RRF top5: `['Agni (2004 film)', 'Arrête ton cinéma', 'Bertrand Tavernier', 'Assassination (1964 film)', 'Garçon stupide']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Agni (2004 film)', 'Arrête ton cinéma', 'Bertrand Tavernier', 'Diane Kurys', 'Garçon stupide']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Agni (2004 film)` sources `['proprag', 'dense']` graph count `2`

### ec70a8a208a311ebbd7cac1f6bf848b6

- Question: Who died earlier, Werner Mensching or Karel Zich?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Karel Zich`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Karel Zich', 'Werner Mensching', 'Karl, Prince of Leiningen', 'Dugès', 'Rita Blumenberg']`
- Impl RRF top5: `['Karel Zich', 'Werner Mensching', 'Karl, Prince of Leiningen', 'Dugès', 'Rita Blumenberg']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Karel Zich', 'Werner Mensching', 'Rita Blumenberg', 'Karl, Prince of Leiningen', 'K-rupt']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Karel Zich` sources `['proprag', 'dense']` graph count `2`

### 0083038e0bde11eba7f7acde48001122

- Question: Where did Prince Ferdinand Of Bavaria's mother die?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Prince Ferdinand of Bavaria`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Prince Ferdinand of Bavaria', 'Joseph Ferdinand of Bavaria', 'Duke Clement Francis of Bavaria', 'Maria Theresa of Austria-Este (1849–1919)', 'Princess Maria Elisabeth of Bavaria']`
- Impl RRF top5: `['Prince Ferdinand of Bavaria', 'Joseph Ferdinand of Bavaria', 'Duke Clement Francis of Bavaria', 'Maria Theresa of Austria-Este (1849–1919)', 'Princess Maria Elisabeth of Bavaria']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Prince Ferdinand of Bavaria', 'Joseph Ferdinand of Bavaria', 'Prince Ferdinand of Hohenzollern', 'Duke Clement Francis of Bavaria', 'Maria Theresa of Austria-Este (1849–1919)']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Prince Ferdinand of Bavaria` sources `['proprag', 'dense']` graph count `2`

### 6b8402c40bdd11eba7f7acde48001122

- Question: Who is the child of the director of film An Event?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Morris Harvey`
- PropRAG top1 manual RRF rank: `7`
- Manual RRF top5: `['An Event', 'Steve Cooreman', 'Anup Sengupta', 'Konrad Wolf', 'Daughter of Destiny']`
- Impl RRF top5: `['An Event', 'Steve Cooreman', 'Anup Sengupta', 'Konrad Wolf', 'Daughter of Destiny']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['An Event', 'Morris Harvey', 'Steve Cooreman', 'Vatroslav Mimica', 'Anup Sengupta']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `An Event` sources `['proprag', 'dense']` graph count `2`

### d9c894ec0bd911eba7f7acde48001122

- Question: What nationality is the director of film Good People (Film)?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Good People (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Good People (film)', 'Jean Grémillon', 'Mark Pellington', 'Lasse Hallström', 'Mohammad Ahmadi']`
- Impl RRF top5: `['Good People (film)', 'Jean Grémillon', 'Mark Pellington', 'Lasse Hallström', 'Mohammad Ahmadi']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Good People (film)', 'Lasse Hallström', 'Henrik Ruben Genz', 'Dugès', 'Jean Grémillon']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Good People (film)` sources `['proprag', 'dense']` graph count `2`

### 4d3ab509099c11ebbdb0ac1f6bf848b6

- Question: Are Gare De La Roche-Sur-Yon and Gare De Namps-Quevauvillers both located in the same country?
- doc_id overlap@20: `13`
- title overlap@20: `13`
- union size: `27`
- cross-pool docs: `13`
- PropRAG top1: `Gare de Namps-Quevauvillers`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Gare de Namps-Quevauvillers', 'Gare de La Roche-sur-Yon', 'Gare de Machecoul', 'Gare de Challans', 'Gare de Bouaye']`
- Impl RRF top5: `['Gare de Namps-Quevauvillers', 'Gare de La Roche-sur-Yon', 'Gare de Machecoul', 'Gare de Challans', 'Gare de Bouaye']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Gare de Namps-Quevauvillers', 'Gare de La Roche-sur-Yon', 'Gare de Challans', 'Gare de Machecoul', 'Gare de Lyon Saint-Exupéry']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Gare de Namps-Quevauvillers` sources `['proprag', 'dense']` graph count `2`

### a9e322b20bdc11eba7f7acde48001122

- Question: What is the place of birth of the director of film Fortunella (Film)?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Fortunella (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Fortunella (film)', 'Sepideh Farsi', 'Peter Greenaway', 'Eduardo De Filippo', 'Raffaello Matarazzo']`
- Impl RRF top5: `['Fortunella (film)', 'Sepideh Farsi', 'Peter Greenaway', 'Eduardo De Filippo', 'Raffaello Matarazzo']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Fortunella (film)', 'Sepideh Farsi', 'Eduardo De Filippo', 'Peter Greenaway', 'Three Lucky Fools']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Fortunella (film)` sources `['proprag', 'dense']` graph count `2`

### 8f783c7a0bda11eba7f7acde48001122

- Question: Where was the husband of Joanna Elisabeth Of Holstein-Gottorp born?
- doc_id overlap@20: `17`
- title overlap@20: `17`
- union size: `23`
- cross-pool docs: `17`
- PropRAG top1: `Joanna Elisabeth of Holstein-Gottorp`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Joanna Elisabeth of Holstein-Gottorp', 'Augusta Marie of Holstein-Gottorp', 'Christian August of Holstein-Gottorp, Prince of Eutin', 'Frederick II, Duke of Holstein-Gottorp', 'Duchess Marie Elisabeth of Saxony']`
- Impl RRF top5: `['Joanna Elisabeth of Holstein-Gottorp', 'Augusta Marie of Holstein-Gottorp', 'Christian August of Holstein-Gottorp, Prince of Eutin', 'Frederick II, Duke of Holstein-Gottorp', 'Duchess Marie Elisabeth of Saxony']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Joanna Elisabeth of Holstein-Gottorp', 'Christian August of Holstein-Gottorp, Prince of Eutin', 'Christian August, Prince of Anhalt-Zerbst', 'Duchess Marie Elisabeth of Saxony', 'Frederick II, Duke of Holstein-Gottorp']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Joanna Elisabeth of Holstein-Gottorp` sources `['proprag', 'dense']` graph count `2`

### b642318c0bdd11eba7f7acde48001122

- Question: Where was the place of death of Abdul-Aziz Bin Muhammad's father?
- doc_id overlap@20: `16`
- title overlap@20: `16`
- union size: `24`
- cross-pool docs: `16`
- PropRAG top1: `Abdul-Aziz bin Muhammad`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Abdul-Aziz bin Muhammad', 'Abdullah ibn Muhammad', 'Az-Zubayr ibn Abd al-Muttalib', 'Muhammad bin Saud', 'Abdul Aziz bin Musaid']`
- Impl RRF top5: `['Abdul-Aziz bin Muhammad', 'Abdullah ibn Muhammad', 'Az-Zubayr ibn Abd al-Muttalib', 'Muhammad bin Saud', 'Abdul Aziz bin Musaid']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Abdul-Aziz bin Muhammad', 'Abdullah ibn Muhammad', 'Muhammad bin Saud', 'Az-Zubayr ibn Abd al-Muttalib', 'Abd al-Aziz ibn al-Walid']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Abdul-Aziz bin Muhammad` sources `['proprag', 'dense']` graph count `2`

### a52e9e1908b711ebbd88ac1f6bf848b6

- Question: Which film has the director who died later, Love, Honor And Oh-Baby! or I Cover The Underworld?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `I Cover the Underworld`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['I Cover the Underworld', 'Love, Honor and Oh-Baby!', 'Love, Honor, and Oh Baby!', 'Charles Vidor', 'Love, Honor and Obey']`
- Impl RRF top5: `['I Cover the Underworld', 'Love, Honor and Oh-Baby!', 'Love, Honor, and Oh Baby!', 'Charles Vidor', 'Love, Honor and Obey']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['I Cover the Underworld', 'Love, Honor and Oh-Baby!', 'Love, Honor, and Oh Baby!', 'Charles Vidor', 'Charles Lamont']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `I Cover the Underworld` sources `['proprag', 'dense']` graph count `2`

### fcdafe320bdb11eba7f7acde48001122

- Question: Where was the director of film The Circus Cyclone born?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `The Circus Cyclone`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Circus Cyclone', 'Claude Weisz', 'Sidney Olcott', 'Hassan Zee', 'Peter Greenaway']`
- Impl RRF top5: `['The Circus Cyclone', 'Claude Weisz', 'Sidney Olcott', 'Hassan Zee', 'Peter Greenaway']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Circus Cyclone', 'Claude Weisz', 'Albert S. Rogell', 'Sidney Olcott', 'The Knockout Kid']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Circus Cyclone` sources `['proprag', 'dense']` graph count `2`

### 7d87730c087211ebbd66ac1f6bf848b6

- Question: Are the movies Wizards Of The Lost Kingdom and Final Exam (1981 Film), from the same country?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Wizards of the Lost Kingdom II`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Final Exam (1981 film)', 'Wizards of the Lost Kingdom II', 'Wizards of the Lost Kingdom', 'Killed the Family and Went to the Movies', 'David A. Prior']`
- Impl RRF top5: `['Final Exam (1981 film)', 'Wizards of the Lost Kingdom II', 'Wizards of the Lost Kingdom', 'Killed the Family and Went to the Movies', 'David A. Prior']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Final Exam (1981 film)', 'Wizards of the Lost Kingdom II', 'Wizards of the Lost Kingdom', 'David A. Prior', 'Killed the Family and Went to the Movies']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Final Exam (1981 film)` sources `['proprag', 'dense']` graph count `2`

### ee5cf4a008a711ebbd7fac1f6bf848b6

- Question: Which museum was established first, Museum Of Croatian Archaeological Monuments or Bayernhof Music Museum?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Museum of Croatian Archaeological Monuments`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Museum of Croatian Archaeological Monuments', 'Bayernhof Music Museum', 'Estonian Theatre and Music Museum', 'Danish Music Museum', 'Telangana State Archaeology Museum']`
- Impl RRF top5: `['Museum of Croatian Archaeological Monuments', 'Bayernhof Music Museum', 'Estonian Theatre and Music Museum', 'Danish Music Museum', 'Telangana State Archaeology Museum']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Museum of Croatian Archaeological Monuments', 'Bayernhof Music Museum', 'Estonian Theatre and Music Museum', 'Ali Baksh Jarnail', 'Danish Music Museum']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Museum of Croatian Archaeological Monuments` sources `['proprag', 'dense']` graph count `2`

### 037da85d08c611ebbd90ac1f6bf848b6

- Question: Are Christopher Newton (Criminal) and Frances M. Vega of the same nationality?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Frances M. Vega`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Frances M. Vega', 'Christopher Hart', 'Dugès', 'Christopher Newton', 'Christopher Newton (criminal)']`
- Impl RRF top5: `['Frances M. Vega', 'Christopher Hart', 'Dugès', 'Christopher Newton', 'Christopher Newton (criminal)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Frances M. Vega', 'Christopher Hart', 'Benjamin Kendrick Pierce', 'Christopher Newton (criminal)', 'Christopher Newton']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Frances M. Vega` sources `['proprag', 'dense']` graph count `2`

### e15d99380bdd11eba7f7acde48001122

- Question: Where did the director of film Dancing In The Rain (Film) die?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Dancing in the Rain (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Dancing in the Rain (film)', 'Dancing in the Rain', 'Louis Mercanton', 'Dino Risi', 'Where Was I']`
- Impl RRF top5: `['Dancing in the Rain (film)', 'Dancing in the Rain', 'Louis Mercanton', 'Dino Risi', 'Where Was I']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dancing in the Rain', 'Dancing in the Rain (film)', 'James Young (director)', 'Boštjan Hladnik', 'Where Was I']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Dancing in the Rain` sources `['proprag', 'dense']` graph count `2`

### f05423560bda11eba7f7acde48001122

- Question: Where did Saw Thanda's husband die?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Saw Thanda`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Saw Thanda', 'Salin Mibaya', 'Where Was I', 'James Randall Marsh', 'Devisingh Ransingh Shekhawat']`
- Impl RRF top5: `['Saw Thanda', 'Salin Mibaya', 'Where Was I', 'James Randall Marsh', 'Devisingh Ransingh Shekhawat']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Saw Thanda', 'Salin Mibaya', 'Where Was I', 'Min Dikkha', 'Gytha Thorkelsdóttir']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Saw Thanda` sources `['proprag', 'dense']` graph count `2`

### 0ce9a92008ed11ebbda7ac1f6bf848b6

- Question: Did the movies Inside The Room and Crude Set Drama, originate from the same country?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Crude Set Drama`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Crude Set Drama', 'Inside the Room', 'The Room (film)', 'The Room', 'In the Room']`
- Impl RRF top5: `['Crude Set Drama', 'Inside the Room', 'The Room (film)', 'The Room', 'In the Room']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Crude Set Drama', 'Inside the Room', 'The Room (film)', 'The Room', 'The Birth of John the Baptist']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Crude Set Drama` sources `['proprag', 'dense']` graph count `2`

### 7d7237ac08a011ebbd78ac1f6bf848b6

- Question: Which film has the director who was born earlier, The Marriage Of Princess Demidoff or The Pocket-Knife?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `The Marriage of Princess Demidoff`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Marriage of Princess Demidoff', 'The Pocket-knife', 'The Marriage of Krechinsky', 'Jonathan Demme', 'Lewis Milestone']`
- Impl RRF top5: `['The Marriage of Princess Demidoff', 'The Pocket-knife', 'The Marriage of Krechinsky', 'Jonathan Demme', 'Lewis Milestone']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Marriage of Princess Demidoff', 'The Pocket-knife', 'The Marriage of Krechinsky', 'Ben Sombogaart', 'Jonathan Demme']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Marriage of Princess Demidoff` sources `['proprag', 'dense']` graph count `2`

### 90c8793c087411ebbd67ac1f6bf848b6

- Question: Which film has the director who is older than the other, Airheads or Return To Cabin By The Lake? 
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Airheads`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Airheads', 'Return to Cabin by the Lake', 'Michael Lehmann', "Return to Return to Nuke 'Em High AKA Volume 2", 'The Return of Swamp Thing']`
- Impl RRF top5: `['Airheads', 'Return to Cabin by the Lake', 'Michael Lehmann', "Return to Return to Nuke 'Em High AKA Volume 2", 'The Return of Swamp Thing']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Airheads', 'Return to Cabin by the Lake', 'Michael Lehmann', 'Po-Chih Leong', "Return to Return to Nuke 'Em High AKA Volume 2"]`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Airheads` sources `['proprag', 'dense']` graph count `2`

### 63c16246095211ebbdaeac1f6bf848b6

- Question: Which film was released first, A Romance Of Seville or Pirates Of The Sky?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `A Romance of Seville`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['A Romance of Seville', 'Pirates of the Sky', 'Pirates of the Coast', 'The Romance of Max', 'Romance on the Range']`
- Impl RRF top5: `['A Romance of Seville', 'Pirates of the Sky', 'Pirates of the Coast', 'The Romance of Max', 'Romance on the Range']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['A Romance of Seville', 'Pirates of the Sky', 'Pirates of the Coast', 'The Trail of the Lonesome Pine (1923 film)', 'Our Emden']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `A Romance of Seville` sources `['proprag', 'dense']` graph count `2`

### 48e6d06e086f11ebbd62ac1f6bf848b6

- Question: Which film has the director died later, Lost In The Stratosphere or Blind Man'S Eyes?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Blind Man's Eyes`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `["Blind Man's Eyes", 'Lost in the Stratosphere', 'Blind Man (film)', "Dead Man's Eyes", 'Harry Revier']`
- Impl RRF top5: `["Blind Man's Eyes", 'Lost in the Stratosphere', 'Blind Man (film)', "Dead Man's Eyes", 'Harry Revier']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `["Blind Man's Eyes", 'Lost in the Stratosphere', 'Blind Man (film)', "Dead Man's Eyes", 'Melville W. Brown']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Blind Man's Eyes` sources `['proprag', 'dense']` graph count `2`

### d18fc6640bda11eba7f7acde48001122

- Question: What nationality is the performer of song Am I Wrong (Étienne De Crécy Song)?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Am I Wrong (Étienne de Crécy song)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Am I Wrong (Étienne de Crécy song)', 'Where Was I', 'Dugès', 'When I Was Young', 'Here to Stay']`
- Impl RRF top5: `['Am I Wrong (Étienne de Crécy song)', 'Where Was I', 'Dugès', 'When I Was Young', 'Here to Stay']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Am I Wrong (Étienne de Crécy song)', 'Étienne de Crécy', 'Where Was I', 'Dugès', 'When I Was Young']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Am I Wrong (Étienne de Crécy song)` sources `['proprag', 'dense']` graph count `2`

### 4a11106f08ea11ebbda7ac1f6bf848b6

- Question: Who was born earlier, Marjatta Raita or Bruce Coulter?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Bruce Coulter`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Bruce Coulter', 'Marjatta Raita', 'Aarno Sulkanen', 'John Edward Bruce', 'Hélène Grenier']`
- Impl RRF top5: `['Bruce Coulter', 'Marjatta Raita', 'Aarno Sulkanen', 'John Edward Bruce', 'Hélène Grenier']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Bruce Coulter', 'Marjatta Raita', 'Aarno Sulkanen', 'Dugès', 'John Edward Bruce']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Bruce Coulter` sources `['proprag', 'dense']` graph count `2`

### a457e11c0bdb11eba7f7acde48001122

- Question: Who is the mother of the director of film Atomised (Film)?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Atomised (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Atomised (film)', 'Steve Cooreman', 'Jacques Doillon', 'Catherine Breillat', 'Claude Weisz']`
- Impl RRF top5: `['Atomised (film)', 'Steve Cooreman', 'Jacques Doillon', 'Catherine Breillat', 'Claude Weisz']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Atomised (film)', 'Jacques Doillon', 'Steve Cooreman', '3096 Days', 'Oskar Roehler']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Atomised (film)` sources `['proprag', 'dense']` graph count `2`

### 632fcc82085611ebbd59ac1f6bf848b6

- Question: Was Angus Wagner or Juan Carlos Falcón born first?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Angus Wagner`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Angus Wagner', 'Juan Carlos Falcón', 'Alberto Falcón', 'José Luis Falcón', 'Dugès']`
- Impl RRF top5: `['Angus Wagner', 'Juan Carlos Falcón', 'Alberto Falcón', 'José Luis Falcón', 'Dugès']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Angus Wagner', 'Juan Carlos Falcón', 'Alberto Falcón', 'Dugès', 'José Luis Falcón']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Angus Wagner` sources `['proprag', 'dense']` graph count `2`

### 076288460bde11eba7f7acde48001122

- Question: What is the date of birth of Henry I Of Ziębice's father?
- doc_id overlap@20: `17`
- title overlap@20: `17`
- union size: `23`
- cross-pool docs: `17`
- PropRAG top1: `Henry I of Ziębice`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Henry I of Ziębice', 'Henry VIII of Legnica', 'Jan I of Żagań', 'Wenceslaus I of Legnica', 'Jan IV of Oświęcim']`
- Impl RRF top5: `['Henry I of Ziębice', 'Henry VIII of Legnica', 'Jan I of Żagań', 'Wenceslaus I of Legnica', 'Jan IV of Oświęcim']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Henry I of Ziębice', 'Henry VIII of Legnica', 'Nicholas the Small', 'Jan I of Żagań', 'Jan IV of Oświęcim']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Henry I of Ziębice` sources `['proprag', 'dense']` graph count `2`

### 402c6e0008c611ebbd90ac1f6bf848b6

- Question: Was Sammy Hagar or Renaud Garcia-Fons born first?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Sammy Hagar`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Renaud Garcia-Fons', 'Sammy Hagar', 'David Lauser', 'Dugès', 'Manuel García Calderón']`
- Impl RRF top5: `['Renaud Garcia-Fons', 'Sammy Hagar', 'David Lauser', 'Dugès', 'Manuel García Calderón']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Renaud Garcia-Fons', 'Sammy Hagar', 'Nguyên Lê', 'David Lauser', 'Dugès']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Renaud Garcia-Fons` sources `['proprag', 'dense']` graph count `2`

### 9523a3df088911ebbd6eac1f6bf848b6

- Question: Do David Elliot (Illustrator) and Eustace William Ferguson share the same nationality?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Eustace William Ferguson`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['David Elliot (illustrator)', 'Eustace William Ferguson', 'David Elliot (footballer)', 'David Em', 'David Elliot Loye']`
- Impl RRF top5: `['David Elliot (illustrator)', 'Eustace William Ferguson', 'David Elliot (footballer)', 'David Em', 'David Elliot Loye']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['David Elliot (illustrator)', 'Eustace William Ferguson', 'David Elliot (footballer)', 'David Em', 'Adam Rex']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `David Elliot (illustrator)` sources `['proprag', 'dense']` graph count `2`

### 5ff291a80bda11eba7f7acde48001122

- Question: Where was the composer of song Back In The U.S.A. born?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Back in the U.S.A.`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Back in the U.S.A.', 'Chuck Berry', 'Where Was I', 'See the USA in Your Chevrolet', 'Leon Carr']`
- Impl RRF top5: `['Back in the U.S.A.', 'Chuck Berry', 'Where Was I', 'See the USA in Your Chevrolet', 'Leon Carr']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Back in the U.S.A.', 'Chuck Berry', 'Where Was I', 'Elvis Presley', 'Leon Carr']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Back in the U.S.A.` sources `['proprag', 'dense']` graph count `2`

### a08aa2a9089111ebbd73ac1f6bf848b6

- Question: Who is older, Cheryl Saban or Dmitry Grigorieff?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Dmitry Grigorieff`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Cheryl Saban', 'Dmitry Grigorieff', 'Tifanie Christun', 'Jeffry Life', 'Dugès']`
- Impl RRF top5: `['Cheryl Saban', 'Dmitry Grigorieff', 'Tifanie Christun', 'Jeffry Life', 'Dugès']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Cheryl Saban', 'Dmitry Grigorieff', 'Jeffry Life', 'Tifanie Christun', 'David Elliot Loye']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Cheryl Saban` sources `['proprag', 'dense']` graph count `2`

### 63240f22089d11ebbd78ac1f6bf848b6

- Question: Which film has the director who was born later, The First Day Of Freedom or Malabimba – The Malicious Whore?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Malabimba – The Malicious Whore`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Malabimba – The Malicious Whore', 'The First Day of Freedom', 'Michael Powell', 'Day of the Cobra', 'Marco Ferreri']`
- Impl RRF top5: `['Malabimba – The Malicious Whore', 'The First Day of Freedom', 'Michael Powell', 'Day of the Cobra', 'Marco Ferreri']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Malabimba – The Malicious Whore', 'The First Day of Freedom', 'Andrea Bianchi', 'Alain Jessua', 'Fernando Di Leo']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Malabimba – The Malicious Whore` sources `['proprag', 'dense']` graph count `2`

### fde95590089c11ebbd78ac1f6bf848b6

- Question: Was Jorge Ledezma or Yuliya Baraley born first?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Yuliya Baraley`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Yuliya Baraley', 'Jorge Ledezma', 'Dugès', 'Abdul-Vahed Niyazov', 'Yulia Yasenok']`
- Impl RRF top5: `['Yuliya Baraley', 'Jorge Ledezma', 'Dugès', 'Abdul-Vahed Niyazov', 'Yulia Yasenok']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Yuliya Baraley', 'Jorge Ledezma', 'Dugès', 'Abdul-Vahed Niyazov', 'José Alejandro Aguilar López']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Yuliya Baraley` sources `['proprag', 'dense']` graph count `2`

### 3bb9c0740bb011ebab90acde48001122

- Question: Who is Sibyl Hathaway's child-in-law?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Sibyl Hathaway`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sibyl Hathaway', 'Dudley Beaumont', 'Ogawa Mataji', 'Robert Vadra', 'Nick Sidi']`
- Impl RRF top5: `['Sibyl Hathaway', 'Dudley Beaumont', 'Ogawa Mataji', 'Robert Vadra', 'Nick Sidi']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sibyl Hathaway', 'Dudley Beaumont', 'Ogawa Mataji', 'Dugès', 'Robert Vadra']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Sibyl Hathaway` sources `['proprag', 'dense']` graph count `2`

### dfb1d2c8085611ebbd59ac1f6bf848b6

- Question: Which film has the director born first, Sins Of Madeleine or Captain Apache?
- doc_id overlap@20: `13`
- title overlap@20: `13`
- union size: `27`
- cross-pool docs: `13`
- PropRAG top1: `Sins of Madeleine`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Captain Apache', 'Sins of Madeleine', 'Alexander Singer', 'Anthony Mann', 'Peter Glenville']`
- Impl RRF top5: `['Captain Apache', 'Sins of Madeleine', 'Alexander Singer', 'Anthony Mann', 'Peter Glenville']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Captain Apache', 'Sins of Madeleine', 'Alexander Singer', 'Anthony Mann', 'Henri Lepage (director)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Captain Apache` sources `['proprag', 'dense']` graph count `2`

### e7dc72940baf11ebab90acde48001122

- Question: Who is Godomar Ii's stepmother?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Gundobad`
- PropRAG top1 manual RRF rank: `5`
- Manual RRF top5: `['Godomar II', 'Duchess Marie of Württemberg', 'Matilda II, Countess of Boulogne', 'Gothelo I, Duke of Lorraine', 'Gundobad']`
- Impl RRF top5: `['Godomar II', 'Duchess Marie of Württemberg', 'Matilda II, Countess of Boulogne', 'Gothelo I, Duke of Lorraine', 'Gundobad']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Godomar II', 'Gundobad', 'Beatrice of Coimbra', 'Duchess Marie of Württemberg', 'Engelbert, Count of Nevers']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Godomar II` sources `['proprag', 'dense']` graph count `2`

### 764446780bde11eba7f7acde48001122

- Question: Where was the director of film The Outlaw Express born?
- doc_id overlap@20: `8`
- title overlap@20: `7`
- union size: `31`
- cross-pool docs: `7`
- PropRAG top1: `Outlaw Express`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Outlaw Express', 'Sidney Olcott', 'Stagecoach Express (film)', 'Rod Hardy', 'George Sherman']`
- Impl RRF top5: `['Outlaw Express', 'Sidney Olcott', 'Stagecoach Express (film)', 'Rod Hardy', 'George Sherman']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Outlaw Express', 'Leo D. Maloney', 'Sidney Olcott', 'Stagecoach Express (film)', 'George Sherman']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Outlaw Express` sources `['proprag', 'dense']` graph count `2`

### bb97d66208ba11ebbd88ac1f6bf848b6

- Question: Which film has the director who died later, 45 Calibre Echo or Bons Baisers De Hong Kong?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `45 Calibre Echo`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['45 Calibre Echo', 'Bons Baisers de Hong Kong', 'Tsui Hark', 'A Better Tomorrow', 'Stanley Kwan']`
- Impl RRF top5: `['45 Calibre Echo', 'Bons Baisers de Hong Kong', 'Tsui Hark', 'A Better Tomorrow', 'Stanley Kwan']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['45 Calibre Echo', 'Bons Baisers de Hong Kong', 'Bruce M. Mitchell', 'Tsui Hark', 'Jeane Manson']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `45 Calibre Echo` sources `['proprag', 'dense']` graph count `2`

### f1a794f20bdc11eba7f7acde48001122

- Question: What nationality is Lamprocles's father?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Alexandros Margaritis`
- PropRAG top1 manual RRF rank: `8`
- Manual RRF top5: `['Lamprocles', 'Obata Toramori', 'Lars Eliasson', 'Yasuichi Oshima', 'Takayama Tomoteru']`
- Impl RRF top5: `['Lamprocles', 'Obata Toramori', 'Lars Eliasson', 'Yasuichi Oshima', 'Takayama Tomoteru']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Lamprocles', 'Alexandros Margaritis', 'Obata Toramori', 'Lars Eliasson', 'Marie-Lambertine Coclers']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Lamprocles` sources `['proprag', 'dense']` graph count `2`

### 47bb70600bde11eba7f7acde48001122

- Question: Which country the performer of song I Like Control is from?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `I Like Control`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['I Like Control', 'Where Was I', 'When I Was Young', 'Little Fire', 'Shape of My Heart']`
- Impl RRF top5: `['I Like Control', 'Where Was I', 'When I Was Young', 'Little Fire', 'Shape of My Heart']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['I Like Control', 'DJ Clue', 'Little Fire', 'Where Was I', 'When I Was Young']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `I Like Control` sources `['proprag', 'dense']` graph count `2`

### 6cbb0ed6089911ebbd77ac1f6bf848b6

- Question: Which film has the director born later, Romance On The Run or The Palace Of Angels?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `Romance on the Run`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Romance on the Run', 'The Palace of Angels', 'Johnny on the Run', 'Jean Negulesco', 'Roy Rowland (film director)']`
- Impl RRF top5: `['Romance on the Run', 'The Palace of Angels', 'Johnny on the Run', 'Jean Negulesco', 'Roy Rowland (film director)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Romance on the Run', 'The Palace of Angels', 'Johnny on the Run', 'The Circus Clown', 'Jean Negulesco']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Romance on the Run` sources `['proprag', 'dense']` graph count `2`

### 594ca25608df11ebbda0ac1f6bf848b6

- Question: Which film has the director born first, Mord Em'Ly or Ek Hi Bhool (1940 Film)?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Ek Hi Bhool (1940 film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Ek Hi Bhool (1940 film)', 'Ek Hi Bhool', "Mord Em'ly", 'Mouna Geethangal', 'Michael Powell']`
- Impl RRF top5: `['Ek Hi Bhool (1940 film)', 'Ek Hi Bhool', "Mord Em'ly", 'Mouna Geethangal', 'Michael Powell']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Ek Hi Bhool (1940 film)', 'Ek Hi Bhool', "Mord Em'ly", 'Vijay Bhatt', 'Michael Powell']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Ek Hi Bhool (1940 film)` sources `['proprag', 'dense']` graph count `2`

### a431a93808d511ebbd98ac1f6bf848b6

- Question: Which film has the director born earlier, The Korean Wedding Chest or True To The Navy?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `True to the Navy`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['True to the Navy', 'The Korean Wedding Chest', 'The Wedding Chest', 'Navy Nurse', 'Lewis Milestone']`
- Impl RRF top5: `['True to the Navy', 'The Korean Wedding Chest', 'The Wedding Chest', 'Navy Nurse', 'Lewis Milestone']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['True to the Navy', 'The Korean Wedding Chest', 'The Wedding Chest', 'Ulrike Ottinger', 'Navy Nurse']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `True to the Navy` sources `['proprag', 'dense']` graph count `2`

### 421d07aa0bb011ebab90acde48001122

- Question: Who is Marianus V Of Arborea's mother?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Marianus V of Arborea`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Marianus V of Arborea', 'Marianus I of Arborea', 'Brancaleone Doria', 'William II of Narbonne', 'Minamoto no Chikako']`
- Impl RRF top5: `['Marianus V of Arborea', 'Marianus I of Arborea', 'Brancaleone Doria', 'William II of Narbonne', 'Minamoto no Chikako']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Marianus V of Arborea', 'Marianus I of Arborea', 'Brancaleone Doria', 'William II of Narbonne', 'Minamoto no Chikako']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Marianus V of Arborea` sources `['proprag', 'dense']` graph count `2`

### 3f48bc400bb011ebab90acde48001122

- Question: Who is the paternal grandfather of Margaret Of Bavaria, Marchioness Of Mantua?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Margaret of Bavaria, Marchioness of Mantua`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Margaret of Bavaria, Marchioness of Mantua', 'Margaret of Bavaria, Electress Palatine', 'Albert III, Duke of Bavaria', 'Margaret Paleologa', 'William of Bavaria-Munich']`
- Impl RRF top5: `['Margaret of Bavaria, Marchioness of Mantua', 'Margaret of Bavaria, Electress Palatine', 'Albert III, Duke of Bavaria', 'Margaret Paleologa', 'William of Bavaria-Munich']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Margaret of Bavaria, Marchioness of Mantua', 'Albrecht, Duke of Bavaria', 'Margaret of Bavaria, Electress Palatine', 'Albert III, Duke of Bavaria', 'Margaret of Cleves, Duchess of Bavaria-Munich']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Margaret of Bavaria, Marchioness of Mantua` sources `['proprag', 'dense']` graph count `2`

### c68eb0be08e611ebbda5ac1f6bf848b6

- Question: Which film has the director who was born later, Felices 140 or Dr. Who And The Daleks?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Dr. Who and the Daleks`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Dr. Who and the Daleks', 'Felices 140', "Daleks' Invasion Earth 2150 A.D.", 'Gordon Flemyng', 'Roberta Tovey']`
- Impl RRF top5: `['Dr. Who and the Daleks', 'Felices 140', "Daleks' Invasion Earth 2150 A.D.", 'Gordon Flemyng', 'Roberta Tovey']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dr. Who and the Daleks', 'Felices 140', 'Gordon Flemyng', "Daleks' Invasion Earth 2150 A.D.", 'Roberta Tovey']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Dr. Who and the Daleks` sources `['proprag', 'dense']` graph count `2`

### ab28bfb6097d11ebbdb0ac1f6bf848b6

- Question: Which film came out first, Mama'S Little Pirate or Beauties On Motor Scooters?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Beauties on Motor Scooters`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Beauties on Motor Scooters', "Mama's Little Pirate", 'Such a Little Pirate', "Mama's Little Girl (disambiguation)", 'Beauties on Bicycles']`
- Impl RRF top5: `['Beauties on Motor Scooters', "Mama's Little Pirate", 'Such a Little Pirate', "Mama's Little Girl (disambiguation)", 'Beauties on Bicycles']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Beauties on Motor Scooters', "Mama's Little Pirate", 'Such a Little Pirate', 'Clown Princes', 'Beauties on Bicycles']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Beauties on Motor Scooters` sources `['proprag', 'dense']` graph count `2`

### 0f97e1f3096f11ebbdb0ac1f6bf848b6

- Question: Which film came out earlier, Two Brides And A Baby or Dinner At The Ritz?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Two Brides and a Baby`
- PropRAG top1 manual RRF rank: `3`
- Manual RRF top5: `['Dinner at the Ritz', 'The Two Brides', 'Two Brides and a Baby', 'Dinner at Eight', 'Dinner at Eight (1989 film)']`
- Impl RRF top5: `['Dinner at the Ritz', 'The Two Brides', 'Two Brides and a Baby', 'Dinner at Eight', 'Dinner at Eight (1989 film)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dinner at the Ritz', 'Two Brides and a Baby', 'The Two Brides', 'Dinner at Eight', 'Harold D. Schuster']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Dinner at the Ritz` sources `['proprag', 'dense']` graph count `2`

### 3ab846860bde11eba7f7acde48001122

- Question: Where did the composer of film Camille (1926 Feature Film) die?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Camille (1926 feature film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Camille (1926 feature film)', 'Henri Verdun', 'Louis Mercanton', 'Bert Grund', 'Oscar Straus (composer)']`
- Impl RRF top5: `['Camille (1926 feature film)', 'Henri Verdun', 'Louis Mercanton', 'Bert Grund', 'Oscar Straus (composer)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Camille (1926 feature film)', 'Henri Verdun', 'Louis Mercanton', 'Claude Pinoteau', 'Bert Grund']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Camille (1926 feature film)` sources `['proprag', 'dense']` graph count `2`

### 8cd3fe6c0baf11ebab90acde48001122

- Question: Who is Colgú Mac Faílbe Flaind's uncle?
- doc_id overlap@20: `13`
- title overlap@20: `13`
- union size: `27`
- cross-pool docs: `13`
- PropRAG top1: `Colgú mac Faílbe Flaind`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Colgú mac Faílbe Flaind', 'Faílbe Flann mac Áedo Duib', 'Finguine mac Cathail', 'Tnúthgal mac Donngaile', 'Domnall mac Caustantín']`
- Impl RRF top5: `['Colgú mac Faílbe Flaind', 'Faílbe Flann mac Áedo Duib', 'Finguine mac Cathail', 'Tnúthgal mac Donngaile', 'Domnall mac Caustantín']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Colgú mac Faílbe Flaind', 'Faílbe Flann mac Áedo Duib', 'Finguine mac Cathail', 'Tnúthgal mac Donngaile', 'Domnall mac Caustantín']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Colgú mac Faílbe Flaind` sources `['proprag', 'dense']` graph count `2`

### 0d9251260baf11ebab90acde48001122

- Question: Who is the sibling-in-law of Jean Tangye?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Jean Tangye`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Jean Tangye', 'Dugès', "Angélique de Saint-Jean Arnauld d'Andilly", 'James Armour (Master mason)', 'Derek Tangye']`
- Impl RRF top5: `['Jean Tangye', 'Dugès', "Angélique de Saint-Jean Arnauld d'Andilly", 'James Armour (Master mason)', 'Derek Tangye']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Jean Tangye', 'Derek Tangye', 'Dugès', 'Mayor of Elizabeth, New Jersey', "List of people named O'Grady"]`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Jean Tangye` sources `['proprag', 'dense']` graph count `2`

### 8cdc97de0bdc11eba7f7acde48001122

- Question: Where was the founder of magazine Track & Field News born?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Track & Field News`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Track & Field News', 'Bert Nelson (publisher)', 'Charles Arthur Tabberer', 'Tsuruichi Hayashi', 'Where Was I']`
- Impl RRF top5: `['Track & Field News', 'Bert Nelson (publisher)', 'Charles Arthur Tabberer', 'Tsuruichi Hayashi', 'Where Was I']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Track & Field News', 'Bert Nelson (publisher)', 'Charles Arthur Tabberer', 'Tsuruichi Hayashi', 'Where Was I']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Track & Field News` sources `['proprag', 'dense']` graph count `2`

### 178a2b9408d611ebbd98ac1f6bf848b6

- Question: Was Shahanuddin Choudhury or Domenico Distilo born first?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Domenico Distilo`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Shahanuddin Choudhury', 'Domenico Distilo', 'Shah Muhammad Sagir', 'Dugès', 'Shahadat Chowdhury']`
- Impl RRF top5: `['Shahanuddin Choudhury', 'Domenico Distilo', 'Shah Muhammad Sagir', 'Dugès', 'Shahadat Chowdhury']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Domenico Distilo', 'Shahanuddin Choudhury', 'Shah Muhammad Sagir', 'Dugès', 'Shahadat Chowdhury']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Domenico Distilo` sources `['proprag', 'dense']` graph count `2`

### d73fa07c0bdd11eba7f7acde48001122

- Question: What is the date of death of Alfonso Pérez De Guzmán, 5Th Duke Of Medina Sidonia's father?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `Juan Alfonso Pérez de Guzmán, 6th Duke of Medina Sidonia`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Juan Alfonso Pérez de Guzmán, 6th Duke of Medina Sidonia', 'Alfonso Pérez de Guzmán, 5th Duke of Medina Sidonia', 'Enrique Pérez de Guzmán, 4th Duke of Medina Sidonia', 'Juan Alfonso Pérez de Guzmán, 3rd Duke of Medina Sidonia', 'Gaspar Juan Pérez de Guzmán, 10th Duke of Medina Sidonia']`
- Impl RRF top5: `['Juan Alfonso Pérez de Guzmán, 6th Duke of Medina Sidonia', 'Alfonso Pérez de Guzmán, 5th Duke of Medina Sidonia', 'Enrique Pérez de Guzmán, 4th Duke of Medina Sidonia', 'Juan Alfonso Pérez de Guzmán, 3rd Duke of Medina Sidonia', 'Gaspar Juan Pérez de Guzmán, 10th Duke of Medina Sidonia']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Juan Alfonso Pérez de Guzmán, 6th Duke of Medina Sidonia', 'Alfonso Pérez de Guzmán, 5th Duke of Medina Sidonia', 'Enrique Pérez de Guzmán, 4th Duke of Medina Sidonia', 'Juan Alfonso Pérez de Guzmán, 3rd Duke of Medina Sidonia', 'Gaspar Juan Pérez de Guzmán, 10th Duke of Medina Sidonia']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Juan Alfonso Pérez de Guzmán, 6th Duke of Medina Sidonia` sources `['proprag', 'dense']` graph count `2`

### 9bc54aba0bdd11eba7f7acde48001122

- Question: What is the place of birth of the performer of song Merry-Go-Round (Ayumi Hamasaki Song)?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Ayumi Hamasaki`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Ayumi Hamasaki', 'Merry-Go-Round (Ayumi Hamasaki song)', 'Evolution (Ayumi Hamasaki song)', 'Place of birth', 'Merry Go Round (Royce da 5\'9" song)']`
- Impl RRF top5: `['Ayumi Hamasaki', 'Merry-Go-Round (Ayumi Hamasaki song)', 'Evolution (Ayumi Hamasaki song)', 'Place of birth', 'Merry Go Round (Royce da 5\'9" song)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Ayumi Hamasaki', 'Merry-Go-Round (Ayumi Hamasaki song)', 'Evolution (Ayumi Hamasaki song)', 'Place of birth', 'Merry Go Round (Royce da 5\'9" song)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Ayumi Hamasaki` sources `['proprag', 'dense']` graph count `2`

### 36b8644508af11ebbd83ac1f6bf848b6

- Question: Which film has the director who is older than the other, The Lone Prairie or Buckaroo From Powder River? 
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `The Lone Prairie`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Lone Prairie', 'Buckaroo from Powder River', 'Henry Hathaway', 'Bury Me Not on the Lone Prairie (film)', 'The Hawk of Powder River']`
- Impl RRF top5: `['The Lone Prairie', 'Buckaroo from Powder River', 'Henry Hathaway', 'Bury Me Not on the Lone Prairie (film)', 'The Hawk of Powder River']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Lone Prairie', 'Buckaroo from Powder River', 'The Hawk of Powder River', 'Ray Nazarro', 'Bury Me Not on the Lone Prairie (film)']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Lone Prairie` sources `['proprag', 'dense']` graph count `2`

### 81e0d683087c11ebbd6aac1f6bf848b6

- Question: Which film has the director who was born earlier, César And Rosalie or Beatrice (1987 Film)?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `César and Rosalie`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Beatrice (1987 film)', 'César and Rosalie', 'Bertrand Tavernier', 'Claude Sautet', 'Attention bandits!']`
- Impl RRF top5: `['Beatrice (1987 film)', 'César and Rosalie', 'Bertrand Tavernier', 'Claude Sautet', 'Attention bandits!']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Beatrice (1987 film)', 'César and Rosalie', 'Bertrand Tavernier', 'Claude Sautet', 'Attention bandits!']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Beatrice (1987 film)` sources `['proprag', 'dense']` graph count `2`

### 689d225009b411ebbdb0ac1f6bf848b6

- Question: Are Wiqu (Lima) and Qillqa (Melgar) both located in the same country?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Wiqu (Lima)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Wiqu (Lima)', 'Qillqa (Melgar)', 'Inka Pirqa', 'Puka Puka (Lima)', 'Misapata (Cabana-Lucanas)']`
- Impl RRF top5: `['Wiqu (Lima)', 'Qillqa (Melgar)', 'Inka Pirqa', 'Puka Puka (Lima)', 'Misapata (Cabana-Lucanas)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Wiqu (Lima)', 'Qillqa (Melgar)', 'Inka Pirqa', 'Machu Kunturi', 'Puka Puka (Lima)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Wiqu (Lima)` sources `['proprag', 'dense']` graph count `2`

### 5cbb015f087511ebbd67ac1f6bf848b6

- Question: Which film has the director died later, The Glass Wall or Where Does It Hurt??
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Rod Amateau`
- PropRAG top1 manual RRF rank: `6`
- Manual RRF top5: `['The Glass Wall', 'Where Does It Hurt?', 'Where Was I', 'Philippe Arthuys', 'Did a Good Man Die?']`
- Impl RRF top5: `['The Glass Wall', 'Where Does It Hurt?', 'Where Was I', 'Philippe Arthuys', 'Did a Good Man Die?']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Glass Wall', 'Rod Amateau', 'Where Does It Hurt?', 'Maxwell Shane', 'Where Was I']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Glass Wall` sources `['proprag', 'dense']` graph count `2`

### 0016373a0bda11eba7f7acde48001122

- Question: What is the date of death of Hermann Ii, Landgrave Of Hesse's father?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Hermann II, Landgrave of Hesse`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Hermann II, Landgrave of Hesse', 'Louis I, Landgrave of Hesse', 'Moritz, Landgrave of Hesse', 'William VII, Landgrave of Hesse-Kassel', 'William II, Landgrave of Hesse-Wanfried-Rheinfels']`
- Impl RRF top5: `['Hermann II, Landgrave of Hesse', 'Louis I, Landgrave of Hesse', 'Moritz, Landgrave of Hesse', 'William VII, Landgrave of Hesse-Kassel', 'William II, Landgrave of Hesse-Wanfried-Rheinfels']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Hermann II, Landgrave of Hesse', 'Louis I, Landgrave of Hesse', 'William VII, Landgrave of Hesse-Kassel', 'Louis, Count of Verdun', 'Moritz, Landgrave of Hesse']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Hermann II, Landgrave of Hesse` sources `['proprag', 'dense']` graph count `2`

### ccd086f80bdb11eba7f7acde48001122

- Question: Where was the director of film Do You Believe? (Film) born?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Do You Believe? (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Do You Believe? (film)', 'Do You Believe?', 'Mark Pellington', 'Jon Gunn', 'Rod Hardy']`
- Impl RRF top5: `['Do You Believe? (film)', 'Do You Believe?', 'Mark Pellington', 'Jon Gunn', 'Rod Hardy']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Do You Believe?', 'Do You Believe? (film)', 'Jon Gunn', 'Mark Pellington', 'Single Video Theory']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Do You Believe?` sources `['proprag', 'dense']` graph count `2`

### 965e667e0bdd11eba7f7acde48001122

- Question: When is Napier Sturt, 3Rd Baron Alington's father's birthday?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Napier Sturt, 3rd Baron Alington`
- PropRAG top1 manual RRF rank: `3`
- Manual RRF top5: `['Humphrey Sturt, 2nd Baron Alington', 'Henry Sturt, 1st Baron Alington', 'Napier Sturt, 3rd Baron Alington', 'Takayama Tomoteru', 'Francis Baring, 3rd Baron Ashburton']`
- Impl RRF top5: `['Humphrey Sturt, 2nd Baron Alington', 'Henry Sturt, 1st Baron Alington', 'Napier Sturt, 3rd Baron Alington', 'Takayama Tomoteru', 'Francis Baring, 3rd Baron Ashburton']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Humphrey Sturt, 2nd Baron Alington', 'Napier Sturt, 3rd Baron Alington', 'Henry Sturt, 1st Baron Alington', 'William Alington, 3rd Baron Alington', 'Francis Baring, 3rd Baron Ashburton']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Humphrey Sturt, 2nd Baron Alington` sources `['proprag', 'dense']` graph count `2`

### 49ccfdfa08ce11ebbd94ac1f6bf848b6

- Question: Which film has the director who is older, Amira & Sam or Fat Man And Little Boy?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Fat Man and Little Boy`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Fat Man and Little Boy', 'Roland Joffé', 'Amira & Sam', 'Robert F. Boyle', 'Sam Wood']`
- Impl RRF top5: `['Fat Man and Little Boy', 'Roland Joffé', 'Amira & Sam', 'Robert F. Boyle', 'Sam Wood']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Fat Man and Little Boy', 'Amira & Sam', 'Roland Joffé', 'Robert F. Boyle', 'Bruce Robinson']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Fat Man and Little Boy` sources `['proprag', 'dense']` graph count `2`

### 43eb7b380bde11eba7f7acde48001122

- Question: What is the place of birth of the director of film Peter'S Friends?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Peter Greenaway`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Peter Greenaway', "Peter's Friends", 'Peter Glenville', 'Peter Levin', 'Lasse Hallström']`
- Impl RRF top5: `['Peter Greenaway', "Peter's Friends", 'Peter Glenville', 'Peter Levin', 'Lasse Hallström']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Peter Greenaway', "Peter's Friends", 'Peter Glenville', 'Thor: The Dark World', 'Peter Levin']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Peter Greenaway` sources `['proprag', 'dense']` graph count `2`

### f48667b80bda11eba7f7acde48001122

- Question: When was the company that published The Miscellany News founded?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `The Miscellany News`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Miscellany News', 'Las Cruces Sun-News', 'Alamogordo Daily News', 'Chicago Daily Journal', 'Thomson Reuters']`
- Impl RRF top5: `['The Miscellany News', 'Las Cruces Sun-News', 'Alamogordo Daily News', 'Chicago Daily Journal', 'Thomson Reuters']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Miscellany News', 'Las Cruces Sun-News', 'Vassar College', 'Alamogordo Daily News', 'Holm Jølsen']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Miscellany News` sources `['proprag', 'dense']` graph count `2`

### d6e9771908dd11ebbd9eac1f6bf848b6

- Question: Which film has the director who died first, Haiducii (Film) or My Wife'S Best Friend?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `My Wife's Best Friend`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Haiducii (film)', "My Wife's Best Friend", "My Wife's Enemy", 'Les amies de ma femme', 'My Wife Is a Gangster']`
- Impl RRF top5: `['Haiducii (film)', "My Wife's Best Friend", "My Wife's Enemy", 'Les amies de ma femme', 'My Wife Is a Gangster']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Haiducii (film)', "My Wife's Best Friend", 'Dinu Cocea', "My Wife's Enemy", 'Les amies de ma femme']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Haiducii (film)` sources `['proprag', 'dense']` graph count `2`

### bd9609880bdb11eba7f7acde48001122

- Question: Who is the father of the director of film Battle In Seattle?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `Battle in Seattle`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Battle in Seattle', 'Daughter of Destiny', 'Dugès', 'Lars Eliasson', 'Stuart Townsend']`
- Impl RRF top5: `['Battle in Seattle', 'Daughter of Destiny', 'Dugès', 'Lars Eliasson', 'Stuart Townsend']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Battle in Seattle', 'Stuart Townsend', 'Leopoldo Torre Nilsson', 'Daughter of Destiny', 'Dugès']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Battle in Seattle` sources `['proprag', 'dense']` graph count `2`

### 748446060bdb11eba7f7acde48001122

- Question: What nationality is the composer of song Make The World Move?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Make the World Move`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Make the World Move', 'Where Was I', 'Here to Stay', 'Shape of My Heart', 'Take Me Out']`
- Impl RRF top5: `['Make the World Move', 'Where Was I', 'Here to Stay', 'Shape of My Heart', 'Take Me Out']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Make the World Move', 'Where Was I', 'Here to Stay', 'Shape of My Heart', 'Take Me Out']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Make the World Move` sources `['proprag', 'dense']` graph count `2`

### e2de58500bd911eba7f7acde48001122

- Question: Where was the place of death of the director of film The Notorious Miss Lisle?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `The Notorious Miss Lisle`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Notorious Miss Lisle', 'Robert Thornby', 'Louis Mercanton', 'J. Lee Thompson', 'James Vincent']`
- Impl RRF top5: `['The Notorious Miss Lisle', 'Robert Thornby', 'Louis Mercanton', 'J. Lee Thompson', 'James Vincent']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Notorious Miss Lisle', 'James Vincent', 'Robert Thornby', 'John Francis Dillon (director)', 'James Young (director)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Notorious Miss Lisle` sources `['proprag', 'dense']` graph count `2`

### f04cc9b408d611ebbd99ac1f6bf848b6

- Question: Which film whose director was born first, Last Tango In Paris or Beasts Of Prey?
- doc_id overlap@20: `13`
- title overlap@20: `13`
- union size: `27`
- cross-pool docs: `13`
- PropRAG top1: `Beasts of Prey`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Beasts of Prey', 'Last Tango in Paris', 'Bernardo Bertolucci', 'The Last Tango', 'Bertrand Tavernier']`
- Impl RRF top5: `['Beasts of Prey', 'Last Tango in Paris', 'Bernardo Bertolucci', 'The Last Tango', 'Bertrand Tavernier']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Beasts of Prey', 'Last Tango in Paris', 'The Last Tango', 'Bernardo Bertolucci', 'Bertrand Tavernier']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Beasts of Prey` sources `['proprag', 'dense']` graph count `2`

### 5c7842780baf11ebab90acde48001122

- Question: Who is the paternal grandfather of Dui Finn?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Sétna Innarraid`
- PropRAG top1 manual RRF rank: `6`
- Manual RRF top5: `['Dui Finn', 'Obata Toramori', 'Dáire Drechlethan', 'Dugès', 'Takayama Tomoteru']`
- Impl RRF top5: `['Dui Finn', 'Obata Toramori', 'Dáire Drechlethan', 'Dugès', 'Takayama Tomoteru']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dui Finn', 'Sétna Innarraid', 'Obata Toramori', 'Eochu Fíadmuine', 'Conaing Bececlach']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Dui Finn` sources `['proprag', 'dense']` graph count `2`

### 0c24f02e08c711ebbd91ac1f6bf848b6

- Question: Which film was released first, The Tender Years or Range War?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `The Tender Years`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Range War', 'The Tender Years', 'Romance on the Range', "Billy the Kid's Range War", 'Riders of the Range (1949 film)']`
- Impl RRF top5: `['Range War', 'The Tender Years', 'Romance on the Range', "Billy the Kid's Range War", 'Riders of the Range (1949 film)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Range War', 'The Tender Years', "Billy the Kid's Range War", 'Lesley Selander', 'Romance on the Range']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Range War` sources `['proprag', 'dense']` graph count `2`

### 84c99c760bb011ebab90acde48001122

- Question: Who is the maternal grandmother of Louis, Dauphin Of France (Son Of Louis Xv)?
- doc_id overlap@20: `14`
- title overlap@20: `14`
- union size: `26`
- cross-pool docs: `14`
- PropRAG top1: `Maria Teresa Rafaela of Spain`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Louis, Dauphin of France (son of Louis XV)', 'Maria Teresa Rafaela of Spain', 'Marie Leszczyńska', 'Louis, Duke of Burgundy', 'Marie Adélaïde of Savoy']`
- Impl RRF top5: `['Louis, Dauphin of France (son of Louis XV)', 'Maria Teresa Rafaela of Spain', 'Marie Leszczyńska', 'Louis, Duke of Burgundy', 'Marie Adélaïde of Savoy']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Louis, Dauphin of France (son of Louis XV)', 'Maria Teresa Rafaela of Spain', 'Louis, Duke of Burgundy', 'Marie Leszczyńska', 'Marie Adélaïde of Savoy']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Louis, Dauphin of France (son of Louis XV)` sources `['proprag', 'dense']` graph count `2`

### f253a3040bdd11eba7f7acde48001122

- Question: What is the place of birth of Princess Maria Of Greece And Denmark's mother?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Princess Maria of Greece and Denmark`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Princess Maria of Greece and Denmark', 'Princess Maria-Olympia of Greece and Denmark', 'Princess Alexia of Greece and Denmark', 'Prince Achileas-Andreas of Greece and Denmark', 'Princess Xenia Georgievna of Russia']`
- Impl RRF top5: `['Princess Maria of Greece and Denmark', 'Princess Maria-Olympia of Greece and Denmark', 'Princess Alexia of Greece and Denmark', 'Prince Achileas-Andreas of Greece and Denmark', 'Princess Xenia Georgievna of Russia']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Princess Maria of Greece and Denmark', 'Princess Maria-Olympia of Greece and Denmark', 'Prince Achileas-Andreas of Greece and Denmark', 'Princess Alexia of Greece and Denmark', 'Olga Constantinovna of Russia']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Princess Maria of Greece and Denmark` sources `['proprag', 'dense']` graph count `2`

### 8dcaaf24089811ebbd77ac1f6bf848b6

- Question: Which film has the director died first, I Like Only You or Brother Rat?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `I Like Only You`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Brother Rat', 'I Like Only You', 'Brother Rat and a Baby', 'Ralph Nelson', 'Bryan Forbes']`
- Impl RRF top5: `['Brother Rat', 'I Like Only You', 'Brother Rat and a Baby', 'Ralph Nelson', 'Bryan Forbes']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Brother Rat', 'I Like Only You', 'Brother Rat and a Baby', 'William Keighley', 'Ralph Nelson']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Brother Rat` sources `['proprag', 'dense']` graph count `2`

### 43d8b52c0baf11ebab90acde48001122

- Question: Who is Princess Mafalda Of Savoy's maternal grandmother?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Princess Mafalda of Savoy`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Princess Mafalda of Savoy', 'Mafalda von Hessen', 'Princess Yolanda of Savoy', 'Princess Maria Beatrice of Savoy', 'Archduchess Maria Isabella of Austria']`
- Impl RRF top5: `['Princess Mafalda of Savoy', 'Mafalda von Hessen', 'Princess Yolanda of Savoy', 'Princess Maria Beatrice of Savoy', 'Archduchess Maria Isabella of Austria']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Princess Mafalda of Savoy', 'Mafalda von Hessen', 'Princess Yolanda of Savoy', 'Elena of Montenegro', 'Princess Maria Beatrice of Savoy']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Princess Mafalda of Savoy` sources `['proprag', 'dense']` graph count `2`

### 63ccce2308cb11ebbd92ac1f6bf848b6

- Question: Who was born earlier, Wang Shusen or Rona Randall?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Wang Shusen`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Wang Shusen', 'Rona Randall', 'James Randall Marsh', 'Dugès', 'Wang Chen (badminton)']`
- Impl RRF top5: `['Wang Shusen', 'Rona Randall', 'James Randall Marsh', 'Dugès', 'Wang Chen (badminton)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Wang Shusen', 'Rona Randall', 'James Randall Marsh', 'Xue Song (badminton)', 'Wang Chen (badminton)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Wang Shusen` sources `['proprag', 'dense']` graph count `2`

### 15d1892108ef11ebbda8ac1f6bf848b6

- Question: Who was born later, Vincenzo Lapuma or Sy Kattelson?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Sy Kattelson`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sy Kattelson', 'Vincenzo Lapuma', 'Vittorio Siri', 'Robert of Capua', 'Vincenzo Gamba']`
- Impl RRF top5: `['Sy Kattelson', 'Vincenzo Lapuma', 'Vittorio Siri', 'Robert of Capua', 'Vincenzo Gamba']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sy Kattelson', 'Vincenzo Lapuma', 'Vatroslav Mimica', 'Robert of Capua', 'Vittorio Siri']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Sy Kattelson` sources `['proprag', 'dense']` graph count `2`

### 755fdd7c08a411ebbd7cac1f6bf848b6

- Question: Which film whose director was born first, Broadway Or Bust or Ballad Of Tara?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Ballad of Tara`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Ballad of Tara', 'Broadway or Bust', 'Elliot Silverstein', 'Norman Taurog', 'Charles Vidor']`
- Impl RRF top5: `['Ballad of Tara', 'Broadway or Bust', 'Elliot Silverstein', 'Norman Taurog', 'Charles Vidor']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Ballad of Tara', 'Broadway or Bust', 'Edward Sedgwick', 'Elliot Silverstein', "The Ridin' Kid from Powder River"]`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Ballad of Tara` sources `['proprag', 'dense']` graph count `2`

### 94198d580bb011ebab90acde48001122

- Question: Who is William Ii, Count Of Flanders's paternal grandmother?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Florine of Burgundy`
- PropRAG top1 manual RRF rank: `12`
- Manual RRF top5: `['Beatrice of Brabant', 'William II, Count of Flanders', 'William II, Count of Perche', 'Baldwin II, Count of Boulogne', 'William II of Dampierre']`
- Impl RRF top5: `['Beatrice of Brabant', 'William II, Count of Flanders', 'William II, Count of Perche', 'Baldwin II, Count of Boulogne', 'William II of Dampierre']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['William II, Count of Flanders', 'Florine of Burgundy', 'Baldwin II, Count of Boulogne', 'William II, Count of Perche', 'Beatrice of Brabant']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `William II, Count of Flanders` sources `['proprag', 'dense']` graph count `2`

### 0ecb27fa0bde11eba7f7acde48001122

- Question: Where was the performer of song Whispers (Corina Song) born?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `Whispers (Corina song)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Whispers (Corina song)', 'Corina (singer)', 'Where Was I', 'Place of birth', 'Andrea Corr']`
- Impl RRF top5: `['Whispers (Corina song)', 'Corina (singer)', 'Where Was I', 'Place of birth', 'Andrea Corr']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Whispers (Corina song)', 'Corina (singer)', "The Caliph's Tea Party", 'Where Was I', "Heart's on Fire (Passenger song)"]`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Whispers (Corina song)` sources `['proprag', 'dense']` graph count `2`

### e9e18ee608d811ebbd9cac1f6bf848b6

- Question: Which film was released first, Kalaya Tasmai Namaha or Undercover With The Kkk?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Undercover with the KKK`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Kalaya Tasmai Namaha', 'Undercover with the KKK', 'Thiruvannamalai (film)', 'Killed the Family and Went to the Movies', 'Veendum Lisa']`
- Impl RRF top5: `['Kalaya Tasmai Namaha', 'Undercover with the KKK', 'Thiruvannamalai (film)', 'Killed the Family and Went to the Movies', 'Veendum Lisa']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Kalaya Tasmai Namaha', 'Undercover with the KKK', 'Thiruvannamalai (film)', 'Hungama in Dubai', 'Killed the Family and Went to the Movies']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Kalaya Tasmai Namaha` sources `['proprag', 'dense']` graph count `2`

### 77ccf6e40bdd11eba7f7acde48001122

- Question: When did Princess Rodam Of Kartli's husband die?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Princess Rodam of Kartli`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Princess Rodam of Kartli', 'Princess Elene of Georgia', 'George VII of Imereti', 'Solomon II of Imereti', 'Prince Adarnase of Kartli']`
- Impl RRF top5: `['Princess Rodam of Kartli', 'Princess Elene of Georgia', 'George VII of Imereti', 'Solomon II of Imereti', 'Prince Adarnase of Kartli']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Princess Rodam of Kartli', 'George VII of Imereti', 'Princess Elene of Georgia', 'Solomon II of Imereti', 'Prince Adarnase of Kartli']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Princess Rodam of Kartli` sources `['proprag', 'dense']` graph count `2`

### 287c0ebf08b411ebbd87ac1f6bf848b6

- Question: Who is younger, Isaac P. Walker or Nakkhatra Mangala?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Nakkhatra Mangala`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Nakkhatra Mangala', 'Isaac P. Walker', 'Busba Kitiyakara', 'Ranginui Walker', 'Wee Willie Walker']`
- Impl RRF top5: `['Nakkhatra Mangala', 'Isaac P. Walker', 'Busba Kitiyakara', 'Ranginui Walker', 'Wee Willie Walker']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Nakkhatra Mangala', 'Busba Kitiyakara', 'Isaac P. Walker', 'Ranginui Walker', 'Dugès']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Nakkhatra Mangala` sources `['proprag', 'dense']` graph count `2`

### cebd0a2a089a11ebbd77ac1f6bf848b6

- Question: Do both Possession (1922 Film) and La Boum films have the directors from the same country?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Possession (1922 film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Possession (1922 film)', 'La Boum', 'La Boum 2', 'Laurent Tirard', 'Claude Sautet']`
- Impl RRF top5: `['Possession (1922 film)', 'La Boum', 'La Boum 2', 'Laurent Tirard', 'Claude Sautet']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Possession (1922 film)', 'La Boum', 'Louis Mercanton', 'Claude Pinoteau', 'La Boum 2']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Possession (1922 film)` sources `['proprag', 'dense']` graph count `2`

### 6b6a5a7f092e11ebbdaeac1f6bf848b6

- Question: Which film came out earlier, La Marca De Satanás or Sofia'S Last Ambulance?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Sofia's Last Ambulance`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['La marca de Satanás', "Sofia's Last Ambulance", 'El Festín de Satanás', 'Proezas de Satanás na Vila de Leva e Tráz', 'Cinco gallinas y el cielo']`
- Impl RRF top5: `['La marca de Satanás', "Sofia's Last Ambulance", 'El Festín de Satanás', 'Proezas de Satanás na Vila de Leva e Tráz', 'Cinco gallinas y el cielo']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['La marca de Satanás', "Sofia's Last Ambulance", 'Cinco gallinas y el cielo', 'El Festín de Satanás', 'Proezas de Satanás na Vila de Leva e Tráz']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `La marca de Satanás` sources `['proprag', 'dense']` graph count `2`

### 843ef89008fa11ebbdacac1f6bf848b6

- Question: Who is younger, Denise Kandel or Bruce Robinson?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Denise Kandel`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Denise Kandel', 'Bruce Robinson', 'Bruce Appleyard', 'Dugès', 'Geoffrey Robinson (bishop)']`
- Impl RRF top5: `['Denise Kandel', 'Bruce Robinson', 'Bruce Appleyard', 'Dugès', 'Geoffrey Robinson (bishop)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Denise Kandel', 'Bruce Robinson', 'Jim Robinson (racing driver)', 'Bruce Appleyard', 'Dugès']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Denise Kandel` sources `['proprag', 'dense']` graph count `2`

### 7efb04e708a011ebbd78ac1f6bf848b6

- Question: Which film has the director who is older than the other, Goopy Gyne Bagha Byne or Halloween Kills? 
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Halloween Kills`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Halloween Kills', 'Goopy Gyne Bagha Byne', 'David Gordon Green', 'Goopi Gawaiya Bagha Bajaiya', 'Goopy Bagha Phire Elo']`
- Impl RRF top5: `['Halloween Kills', 'Goopy Gyne Bagha Byne', 'David Gordon Green', 'Goopi Gawaiya Bagha Bajaiya', 'Goopy Bagha Phire Elo']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Halloween Kills', 'Goopy Gyne Bagha Byne', 'David Gordon Green', 'Goopy Bagha Phire Elo', 'Goopi Gawaiya Bagha Bajaiya']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Halloween Kills` sources `['proprag', 'dense']` graph count `2`

### 96b7c1a60bdd11eba7f7acde48001122

- Question: Where was the place of death of Sancha Of Castile, Queen Of Navarre's mother?
- doc_id overlap@20: `16`
- title overlap@20: `16`
- union size: `24`
- cross-pool docs: `16`
- PropRAG top1: `Sancha of Castile, Queen of Navarre`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sancha of Castile, Queen of Navarre', 'Eleanor of Castile, Queen of Navarre', 'Blanche of Navarre, Countess of Champagne', 'Beatrice of Navarre, Countess of La Marche', 'Sancha of León']`
- Impl RRF top5: `['Sancha of Castile, Queen of Navarre', 'Eleanor of Castile, Queen of Navarre', 'Blanche of Navarre, Countess of Champagne', 'Beatrice of Navarre, Countess of La Marche', 'Sancha of León']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sancha of Castile, Queen of Navarre', 'Eleanor of Castile, Queen of Navarre', 'Sancha of León', 'Blanche of Navarre, Countess of Champagne', "Margaret of L'Aigle"]`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Sancha of Castile, Queen of Navarre` sources `['proprag', 'dense']` graph count `2`

### 121ddeae0bdc11eba7f7acde48001122

- Question: What nationality is Hypsicratea's husband?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Hypsicratea`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Hypsicratea', 'James Randall Marsh', 'Sybil B. G. Eysenck', 'Takayama Tomoteru', 'Pieter van Vollenhoven']`
- Impl RRF top5: `['Hypsicratea', 'James Randall Marsh', 'Sybil B. G. Eysenck', 'Takayama Tomoteru', 'Pieter van Vollenhoven']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Hypsicratea', 'Carl Haber', 'James Randall Marsh', 'Alexandros Margaritis', 'Sybil B. G. Eysenck']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Hypsicratea` sources `['proprag', 'dense']` graph count `2`

### 3ed2ba7a0bda11eba7f7acde48001122

- Question: Who is the father of the director of film Power Of Women (Film)?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `The Power (1984 film)`
- PropRAG top1 manual RRF rank: `4`
- Manual RRF top5: `['Power of Women (film)', 'Daughter of Destiny', 'Gracia Querejeta', 'The Power (1984 film)', 'Lars Eliasson']`
- Impl RRF top5: `['Power of Women (film)', 'Daughter of Destiny', 'Gracia Querejeta', 'The Power (1984 film)', 'Lars Eliasson']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Power of Women (film)', 'The Power (1984 film)', 'Karin Daughter of Ingmar', 'Jayadevi', 'Sons of Ingmar']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Power of Women (film)` sources `['proprag', 'dense']` graph count `2`

### 77f521780bb011ebab90acde48001122

- Question: Who is Margaret Of Baux's father-in-law?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Margaret of Baux`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Margaret of Baux', 'Margaret of Bavaria, Marchioness of Mantua', 'Margaret of Brieg', 'Cecile of Baux', 'Margaret of Bavaria, Electress Palatine']`
- Impl RRF top5: `['Margaret of Baux', 'Margaret of Bavaria, Marchioness of Mantua', 'Margaret of Brieg', 'Cecile of Baux', 'Margaret of Bavaria, Electress Palatine']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Margaret of Baux', 'Margaret of Bavaria, Marchioness of Mantua', 'Margaret of Brieg', 'Cecile of Baux', 'Peter I, Count of Saint-Pol']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Margaret of Baux` sources `['proprag', 'dense']` graph count `2`

### 0868539a0bdb11eba7f7acde48001122

- Question: Which country Lucius Antonius (Grandson Of Mark Antony)'s father is from?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Lucius Antonius (grandson of Mark Antony)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Lucius Antonius (grandson of Mark Antony)', 'Iulla Antonia', 'Iullus Antonius', 'Lucius Antonius Albus (proconsul of Asia)', 'Takayama Tomoteru']`
- Impl RRF top5: `['Lucius Antonius (grandson of Mark Antony)', 'Iulla Antonia', 'Iullus Antonius', 'Lucius Antonius Albus (proconsul of Asia)', 'Takayama Tomoteru']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Lucius Antonius (grandson of Mark Antony)', 'Iulla Antonia', 'Iullus Antonius', 'Lucius Antonius Albus (proconsul of Asia)', 'Quintus Salvidienus Rufus']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Lucius Antonius (grandson of Mark Antony)` sources `['proprag', 'dense']` graph count `2`

### b088d3ca08e211ebbda4ac1f6bf848b6

- Question: Which film whose director is younger, Forbidden Ground or Free Guy?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Forbidden Ground (2013 film)`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Free Guy', 'Forbidden Ground (2013 film)', 'Forbidden Ground', 'Shawn Levy', 'The Nice Guys']`
- Impl RRF top5: `['Free Guy', 'Forbidden Ground (2013 film)', 'Forbidden Ground', 'Shawn Levy', 'The Nice Guys']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Forbidden Ground (2013 film)', 'Free Guy', 'Forbidden Ground', 'Shawn Levy', 'Martin Copping']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Forbidden Ground (2013 film)` sources `['proprag', 'dense']` graph count `2`

### 44e9813a0bdc11eba7f7acde48001122

- Question: Where was the place of burial of Osthryth's father?
- doc_id overlap@20: `16`
- title overlap@20: `16`
- union size: `24`
- cross-pool docs: `16`
- PropRAG top1: `Osthryth`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Osthryth', 'Oswiu', 'Ælfwine of Deira', 'Edwin of Northumbria', 'Sarre Anglo-Saxon cemetery']`
- Impl RRF top5: `['Osthryth', 'Oswiu', 'Ælfwine of Deira', 'Edwin of Northumbria', 'Sarre Anglo-Saxon cemetery']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Osthryth', 'Oswiu', 'Ælfwine of Deira', 'Edwin of Northumbria', 'Oswine of Deira']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Osthryth` sources `['proprag', 'dense']` graph count `2`

### 2ccc68a108fe11ebbdadac1f6bf848b6

- Question: Which magazine came out first, Entertain Magazine or Woman'S Century?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Entertain Magazine`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Entertain Magazine', "Woman's Century", 'Dugès', 'When I Was Young', 'Where Was I']`
- Impl RRF top5: `['Entertain Magazine', "Woman's Century", 'Dugès', 'When I Was Young', 'Where Was I']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Entertain Magazine', "Woman's Century", 'Dugès', 'American Monthly', 'Sky Magazine']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Entertain Magazine` sources `['proprag', 'dense']` graph count `2`

### 29bf69d20bb011ebab90acde48001122

- Question: Who is the stepmother of Sara Ali Khan?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Sara Ali Khan`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sara Ali Khan', 'Saif Ali Khan', 'Héctor Barrantes', 'Ananya Panday', 'Rani Mukerji']`
- Impl RRF top5: `['Sara Ali Khan', 'Saif Ali Khan', 'Héctor Barrantes', 'Ananya Panday', 'Rani Mukerji']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sara Ali Khan', 'Saif Ali Khan', 'Yeh Gulistan Hamara', 'Daughter of Destiny', 'Héctor Barrantes']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Sara Ali Khan` sources `['proprag', 'dense']` graph count `2`

### acbf27ee089211ebbd73ac1f6bf848b6

- Question: Which film whose director is younger, Before The Bat'S Flight Is Done or Agar Tum Na Hote?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Péter Tímár`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `["Before the Bat's Flight Is Done", 'Péter Tímár', 'Agar Tum Na Hote', 'Lekh Tandon', 'Before Sunset']`
- Impl RRF top5: `["Before the Bat's Flight Is Done", 'Péter Tímár', 'Agar Tum Na Hote', 'Lekh Tandon', 'Before Sunset']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `["Before the Bat's Flight Is Done", 'Péter Tímár', 'Agar Tum Na Hote', 'Lekh Tandon', 'Before Sunset']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Before the Bat's Flight Is Done` sources `['proprag', 'dense']` graph count `2`

### 3040e4fb085611ebbd59ac1f6bf848b6

- Question: Which film was released earlier, Powerful: Energy For Everyone or Scaregivers?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Powerful: Energy for Everyone`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Powerful: Energy for Everyone', 'Scaregivers', 'The Power (1984 film)', 'The Power of Love', 'The Warriors']`
- Impl RRF top5: `['Powerful: Energy for Everyone', 'Scaregivers', 'The Power (1984 film)', 'The Power of Love', 'The Warriors']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Powerful: Energy for Everyone', 'Scaregivers', 'The Power (1984 film)', 'The Power of Love', 'Niko von Glasow']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Powerful: Energy for Everyone` sources `['proprag', 'dense']` graph count `2`

### 696ce7cd08be11ebbd8aac1f6bf848b6

- Question: Does Leslie Pietrzyk have the same nationality as Marianne Wiggins?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Marianne Wiggins`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Marianne Wiggins', 'Leslie Pietrzyk', 'Lara Porzak', 'Dugès', 'Herbert Gold']`
- Impl RRF top5: `['Marianne Wiggins', 'Leslie Pietrzyk', 'Lara Porzak', 'Dugès', 'Herbert Gold']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Marianne Wiggins', 'Leslie Pietrzyk', 'Lara Porzak', 'Francis Alexander Durivage', 'Dugès']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Marianne Wiggins` sources `['proprag', 'dense']` graph count `2`

### c531e7560baf11ebab90acde48001122

- Question: Who is Marie Zéphyrine Of France's paternal grandmother?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Marie Zéphyrine of France`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Marie Zéphyrine of France', 'Marie Leszczyńska', 'Henriette of France (1727–1752)', 'Louis, Dauphin of France (son of Louis XV)', 'Maria Teresa Rafaela of Spain']`
- Impl RRF top5: `['Marie Zéphyrine of France', 'Marie Leszczyńska', 'Henriette of France (1727–1752)', 'Louis, Dauphin of France (son of Louis XV)', 'Maria Teresa Rafaela of Spain']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Marie Zéphyrine of France', 'Marie Leszczyńska', 'Louis, Dauphin of France (son of Louis XV)', 'Henriette of France (1727–1752)', 'Margaret of Valois']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Marie Zéphyrine of France` sources `['proprag', 'dense']` graph count `2`

### 0cafe4700bb011ebab90acde48001122

- Question: Who is the paternal grandfather of Anna Dorothea, Abbess Of Quedlinburg?
- doc_id overlap@20: `13`
- title overlap@20: `13`
- union size: `27`
- cross-pool docs: `13`
- PropRAG top1: `Anna Amalia, Abbess of Quedlinburg`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Anna Amalia, Abbess of Quedlinburg', 'Anna Dorothea, Abbess of Quedlinburg', 'Anna Sophia II, Abbess of Quedlinburg', 'Matilda, Abbess of Quedlinburg', 'Hedwig, Abbess of Quedlinburg']`
- Impl RRF top5: `['Anna Amalia, Abbess of Quedlinburg', 'Anna Dorothea, Abbess of Quedlinburg', 'Anna Sophia II, Abbess of Quedlinburg', 'Matilda, Abbess of Quedlinburg', 'Hedwig, Abbess of Quedlinburg']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Anna Amalia, Abbess of Quedlinburg', 'Anna Dorothea, Abbess of Quedlinburg', "Éléonore Desmier d'Olbreuse", 'Anna Sophia II, Abbess of Quedlinburg', 'Matilda, Abbess of Quedlinburg']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Anna Amalia, Abbess of Quedlinburg` sources `['proprag', 'dense']` graph count `2`

### 60402bf80bde11eba7f7acde48001122

- Question: When did Conall Mac Comgaill's father die?
- doc_id overlap@20: `19`
- title overlap@20: `19`
- union size: `21`
- cross-pool docs: `19`
- PropRAG top1: `Conall mac Comgaill`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Conall mac Comgaill', 'Comgall mac Domangairt', 'Conall mac Fidhghal', 'Domnall mac Caustantín', 'Conall mac Áedáin']`
- Impl RRF top5: `['Conall mac Comgaill', 'Comgall mac Domangairt', 'Conall mac Fidhghal', 'Domnall mac Caustantín', 'Conall mac Áedáin']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Conall mac Comgaill', 'Comgall mac Domangairt', 'Conall mac Fidhghal', 'Domnall mac Caustantín', 'Conall mac Áedáin']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Conall mac Comgaill` sources `['proprag', 'dense']` graph count `2`

### fe2dd4b20bdb11eba7f7acde48001122

- Question: Who is the spouse of the director of film The Glass Boat?
- doc_id overlap@20: `3`
- title overlap@20: `3`
- union size: `37`
- cross-pool docs: `3`
- PropRAG top1: `The Glass Boat`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Glass Boat', 'Jazz Boat', 'Philip May', 'Geoff Murphy', 'James Randall Marsh']`
- Impl RRF top5: `['The Glass Boat', 'Jazz Boat', 'Philip May', 'Geoff Murphy', 'James Randall Marsh']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Glass Boat', 'Geoff Murphy', 'Constantin J. David', 'Dugès', 'King of My Heart']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Glass Boat` sources `['proprag', 'dense']` graph count `2`

### d5cee073085911ebbd5bac1f6bf848b6

- Question: Who is younger, Jan Britstra or Leonas Milčius?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Leonas Milčius`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Leonas Milčius', 'Jan Britstra', 'Miloš Zličić', 'Peter Gravesen', 'Leon de Winter']`
- Impl RRF top5: `['Leonas Milčius', 'Jan Britstra', 'Miloš Zličić', 'Peter Gravesen', 'Leon de Winter']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Leonas Milčius', 'Jan Britstra', 'Miloš Zličić', 'Peter Gravesen', 'When I Was Young']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Leonas Milčius` sources `['proprag', 'dense']` graph count `2`

### 6ee6201a0bdc11eba7f7acde48001122

- Question: Where was the place of death of Humphrey Stafford, Earl Of Stafford's father?
- doc_id overlap@20: `13`
- title overlap@20: `13`
- union size: `27`
- cross-pool docs: `13`
- PropRAG top1: `Humphrey Stafford, Earl of Stafford`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Humphrey Stafford, Earl of Stafford', 'Humphrey Stafford, 1st Duke of Buckingham', 'Thomas Stafford, 3rd Earl of Stafford', 'Humphrey Stafford, 1st Earl of Devon', 'John Stafford, 1st Earl of Wiltshire']`
- Impl RRF top5: `['Humphrey Stafford, Earl of Stafford', 'Humphrey Stafford, 1st Duke of Buckingham', 'Thomas Stafford, 3rd Earl of Stafford', 'Humphrey Stafford, 1st Earl of Devon', 'John Stafford, 1st Earl of Wiltshire']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Humphrey Stafford, Earl of Stafford', 'Humphrey Stafford, 1st Duke of Buckingham', 'Humphrey Stafford, 1st Earl of Devon', 'Thomas Stafford, 3rd Earl of Stafford', 'James Stewart, 1st Duke of Richmond']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Humphrey Stafford, Earl of Stafford` sources `['proprag', 'dense']` graph count `2`

### 1b36c01f08df11ebbd9fac1f6bf848b6

- Question: Do Walfredo Reyes Jr. and Zac Dysert have the same nationality?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Zac Dysert`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Zac Dysert', 'Walfredo Reyes Jr.', 'Walfredo de los Reyes', 'Efren Reyes Jr.', 'Efren Reyes Sr.']`
- Impl RRF top5: `['Zac Dysert', 'Walfredo Reyes Jr.', 'Walfredo de los Reyes', 'Efren Reyes Jr.', 'Efren Reyes Sr.']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Zac Dysert', 'Walfredo Reyes Jr.', 'Walfredo de los Reyes', 'Jim Ballard', 'Efren Reyes Jr.']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Zac Dysert` sources `['proprag', 'dense']` graph count `2`

### 095c87880bdd11eba7f7acde48001122

- Question: Which country Al-Mu'Tasim's father is from?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `Al-Mu'tasim`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `["Al-Mu'tasim", "Al-Mu'ayyad", 'Al-Fadl ibn Marwan', "Al-Abbas ibn al-Ma'mun", 'Itakh']`
- Impl RRF top5: `["Al-Mu'tasim", "Al-Mu'ayyad", 'Al-Fadl ibn Marwan', "Al-Abbas ibn al-Ma'mun", 'Itakh']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `["Al-Mu'tasim", "Al-Mu'ayyad", "Al-Abbas ibn al-Ma'mun", 'Itakh', 'Al-Fadl ibn Marwan']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Al-Mu'tasim` sources `['proprag', 'dense']` graph count `2`

### 81c9a46c0bb011ebab90acde48001122

- Question: Who did Joan Ramon Ii, Count Of Cardona marry?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Joan Ramon II, Count of Cardona`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Joan Ramon II, Count of Cardona', 'John Ramon III, Count of Cardona', 'Joan II, Countess of Burgundy', 'Joan III, Countess of Burgundy', 'Antonio de Cardona']`
- Impl RRF top5: `['Joan Ramon II, Count of Cardona', 'John Ramon III, Count of Cardona', 'Joan II, Countess of Burgundy', 'Joan III, Countess of Burgundy', 'Antonio de Cardona']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Joan Ramon II, Count of Cardona', 'John Ramon III, Count of Cardona', 'Joan II, Countess of Burgundy', 'Antonio de Cardona', 'Joan III, Countess of Burgundy']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Joan Ramon II, Count of Cardona` sources `['proprag', 'dense']` graph count `2`

### 1cf6b11808a111ebbd78ac1f6bf848b6

- Question: Which film has the director born earlier, The Planet Of Junior Brown or A Yank In The R.A.F.?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `A Yank in the R.A.F.`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['A Yank in the R.A.F.', 'The Planet of Junior Brown', 'Peter Glenville', 'William A. Wellman', 'Ken Annakin']`
- Impl RRF top5: `['A Yank in the R.A.F.', 'The Planet of Junior Brown', 'Peter Glenville', 'William A. Wellman', 'Ken Annakin']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['A Yank in the R.A.F.', 'The Planet of Junior Brown', 'Peter Glenville', 'Henry King (director)', 'William A. Wellman']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `A Yank in the R.A.F.` sources `['proprag', 'dense']` graph count `2`

### f20ae8a80bdd11eba7f7acde48001122

- Question: Where does Louisa Charlotte Tyndall's husband work at?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Louisa Charlotte Tyndall`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Louisa Charlotte Tyndall', 'John Tyndall (poet)', 'John Tyndall', 'Countess Charlotte of Hanau-Lichtenberg', 'Dugès']`
- Impl RRF top5: `['Louisa Charlotte Tyndall', 'John Tyndall (poet)', 'John Tyndall', 'Countess Charlotte of Hanau-Lichtenberg', 'Dugès']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Louisa Charlotte Tyndall', 'John Tyndall (poet)', 'John Tyndall', 'Countess Charlotte of Hanau-Lichtenberg', 'Frances Lasker Brody']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Louisa Charlotte Tyndall` sources `['proprag', 'dense']` graph count `2`

### 86ea0e1e0bb011ebab90acde48001122

- Question: Who is the father-in-law of Ermengarde Of Hesbaye?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Sigramnus, Count of Hesbaye`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sigramnus, Count of Hesbaye', 'Ermengarde of Hesbaye', 'Ingerman, Count of Hesbaye', 'Thuringbert, Count of Hesbaye', 'Ermengarde of Tours']`
- Impl RRF top5: `['Sigramnus, Count of Hesbaye', 'Ermengarde of Hesbaye', 'Ingerman, Count of Hesbaye', 'Thuringbert, Count of Hesbaye', 'Ermengarde of Tours']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sigramnus, Count of Hesbaye', 'Ermengarde of Hesbaye', 'Ingerman, Count of Hesbaye', 'Louis the Pious', 'Thuringbert, Count of Hesbaye']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Sigramnus, Count of Hesbaye` sources `['proprag', 'dense']` graph count `2`

### e42d315308f711ebbdaaac1f6bf848b6

- Question: Which film was released earlier, Dreamville Presents: Revenge or One More Time With Feeling?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `One More Time with Feeling`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Dreamville Presents: Revenge', 'One More Time with Feeling', 'Once More with Feeling', 'One More Time', 'Rakka (film)']`
- Impl RRF top5: `['Dreamville Presents: Revenge', 'One More Time with Feeling', 'Once More with Feeling', 'One More Time', 'Rakka (film)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dreamville Presents: Revenge', 'One More Time with Feeling', 'Once More with Feeling', 'One More Time', 'Rakka (film)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Dreamville Presents: Revenge` sources `['proprag', 'dense']` graph count `2`

### bbfe9c84087511ebbd67ac1f6bf848b6

- Question: Which film has the director born first, Mutiny (1952 Film) or The Eagle'S Feather?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `The Eagle's Feather`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `["The Eagle's Feather", 'Mutiny (1952 film)', 'Edward Dmytryk', 'Eagle Feather', 'Lewis Milestone']`
- Impl RRF top5: `['Mutiny (1952 film)', "The Eagle's Feather", 'Edward Dmytryk', 'Eagle Feather', 'Lewis Milestone']`
- RRF top5 match manual: `False`
- PropRAG rank top5 from union: `['Mutiny (1952 film)', "The Eagle's Feather", 'Edward Sloman', 'Edward Dmytryk', 'Eagle Feather']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Mutiny (1952 film)` sources `['proprag', 'dense']` graph count `2`

### 1fa46e26084911ebbd55ac1f6bf848b6

- Question: Do both films The Umbrella Coup and The Cow And I have the directors that share the same nationality?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `The Cow and I`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['The Umbrella Coup', 'The Cow and I', 'Kyōen Kobanzame', 'Gérard Oury', 'Roland Joffé']`
- Impl RRF top5: `['The Umbrella Coup', 'The Cow and I', 'Kyōen Kobanzame', 'Gérard Oury', 'Roland Joffé']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Cow and I', 'The Umbrella Coup', 'Kyōen Kobanzame', 'Henri Verneuil', 'Gérard Oury']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `The Cow and I` sources `['proprag', 'dense']` graph count `2`

### 4f8e349e08e111ebbda2ac1f6bf848b6

- Question: Which film has the director who died later, Hotel Paradis or Devil'S Doorway?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `Anthony Mann`
- PropRAG top1 manual RRF rank: `7`
- Manual RRF top5: `['Hotel Paradis', 'Georges Méliès', "Devil's Doorway", 'Michael Powell', 'John Llewellyn Moxey']`
- Impl RRF top5: `['Hotel Paradis', 'Georges Méliès', "Devil's Doorway", 'Michael Powell', 'John Llewellyn Moxey']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Hotel Paradis', 'Anthony Mann', 'Georges Méliès', "Devil's Doorway", 'George Schnéevoigt']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Hotel Paradis` sources `['proprag', 'dense']` graph count `2`

### f0faeb9a0bdb11eba7f7acde48001122

- Question: Who is the father of the director of film Duniya Meri Jeb Mein?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Duniya Meri Jeb Mein`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Duniya Meri Jeb Mein', 'Karan Johar', 'Kasthuri Raja', 'Vaibhavi Merchant', 'Subhash Ghai']`
- Impl RRF top5: `['Duniya Meri Jeb Mein', 'Karan Johar', 'Kasthuri Raja', 'Vaibhavi Merchant', 'Subhash Ghai']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Duniya Meri Jeb Mein', 'Tinnu Anand', 'Dahleez', 'Subhash Ghai', 'Daughter of Destiny']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Duniya Meri Jeb Mein` sources `['proprag', 'dense']` graph count `2`

### 182ecc9a0baf11ebab90acde48001122

- Question: Who is the paternal grandmother of Sigrid Of Sweden (1566–1633)?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Sigrid of Sweden (1566–1633)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Sigrid of Sweden (1566–1633)', 'Sigrid Sture', 'Euphrosina Heldina von Dieffenau', 'Ulf Tostesson', 'Sophia Magdalena of Denmark']`
- Impl RRF top5: `['Sigrid of Sweden (1566–1633)', 'Sigrid Sture', 'Euphrosina Heldina von Dieffenau', 'Ulf Tostesson', 'Sophia Magdalena of Denmark']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Sigrid of Sweden (1566–1633)', 'Sigrid Sture', 'Eric XIV of Sweden', 'Ulf Tostesson', 'Euphrosina Heldina von Dieffenau']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Sigrid of Sweden (1566–1633)` sources `['proprag', 'dense']` graph count `2`

### bcc4e26c08c111ebbd8bac1f6bf848b6

- Question: Do both films A Trial In Prague and Three Strangers have the directors that share the same nationality?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `A Trial in Prague`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['A Trial in Prague', 'Three Strangers', 'Three for Happiness', 'Three Waltzes', 'André Cayatte']`
- Impl RRF top5: `['A Trial in Prague', 'Three Strangers', 'Three for Happiness', 'Three Waltzes', 'André Cayatte']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['A Trial in Prague', 'Three Strangers', 'Three for Happiness', 'Jean Negulesco', 'Three Waltzes']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `A Trial in Prague` sources `['proprag', 'dense']` graph count `2`

### 31e34c220bde11eba7f7acde48001122

- Question: Where was the wife of Adam Włodek born?
- doc_id overlap@20: `10`
- title overlap@20: `10`
- union size: `30`
- cross-pool docs: `10`
- PropRAG top1: `Adam Włodek`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Adam Włodek', 'Wisława Szymborska', 'Ivona Jezierska', 'Anna Rajecka', 'Wojciech Has']`
- Impl RRF top5: `['Adam Włodek', 'Wisława Szymborska', 'Ivona Jezierska', 'Anna Rajecka', 'Wojciech Has']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Adam Włodek', 'Ivona Jezierska', 'Wisława Szymborska', 'Zuzana Justman', 'Wojciech Has']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Adam Włodek` sources `['proprag', 'dense']` graph count `2`

### 38e0bdd2094e11ebbdaeac1f6bf848b6

- Question: Which film was released first, Ryan'S Daughter or A Day At The Museum?
- doc_id overlap@20: `5`
- title overlap@20: `5`
- union size: `35`
- cross-pool docs: `5`
- PropRAG top1: `A Day at the Museum`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['A Day at the Museum', "Ryan's Daughter", 'A Day', 'One Day', 'H Day']`
- Impl RRF top5: `['A Day at the Museum', "Ryan's Daughter", 'A Day', 'One Day', 'H Day']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['A Day at the Museum', "Ryan's Daughter", 'A Day', 'Roberta Tovey', 'One Day']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `A Day at the Museum` sources `['proprag', 'dense']` graph count `2`

### 37f3bd380baf11ebab90acde48001122

- Question: Who is Karl Von Habsburg's paternal grandmother?
- doc_id overlap@20: `16`
- title overlap@20: `16`
- union size: `24`
- cross-pool docs: `16`
- PropRAG top1: `Archduke Franz Karl of Austria`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Karl von Habsburg', 'Archduke Franz Karl of Austria', 'Monika von Habsburg', 'Francesca von Habsburg', 'Andrea von Habsburg']`
- Impl RRF top5: `['Karl von Habsburg', 'Archduke Franz Karl of Austria', 'Monika von Habsburg', 'Francesca von Habsburg', 'Andrea von Habsburg']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Archduke Franz Karl of Austria', 'Karl von Habsburg', 'Archduchess Elisabeth Amalie of Austria', 'Francesca von Habsburg', 'Monika von Habsburg']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Archduke Franz Karl of Austria` sources `['proprag', 'dense']` graph count `2`

### b8ef144a0bdd11eba7f7acde48001122

- Question: Which country Thomas Lyon-Bowes, Lord Glamis's father is from?
- doc_id overlap@20: `17`
- title overlap@20: `17`
- union size: `23`
- cross-pool docs: `17`
- PropRAG top1: `Claude Bowes-Lyon, 13th Earl of Strathmore and Kinghorne`
- PropRAG top1 manual RRF rank: `5`
- Manual RRF top5: `['Thomas Lyon-Bowes, Lord Glamis', 'Thomas Lyon-Bowes, Master of Glamis (born 1821)', 'Thomas Lyon-Bowes, 12th Earl of Strathmore and Kinghorne', 'Charlotte Lyon-Bowes, Lady Glamis', 'Claude Bowes-Lyon, 13th Earl of Strathmore and Kinghorne']`
- Impl RRF top5: `['Thomas Lyon-Bowes, Lord Glamis', 'Thomas Lyon-Bowes, Master of Glamis (born 1821)', 'Thomas Lyon-Bowes, 12th Earl of Strathmore and Kinghorne', 'Charlotte Lyon-Bowes, Lady Glamis', 'Claude Bowes-Lyon, 13th Earl of Strathmore and Kinghorne']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Claude Bowes-Lyon, 13th Earl of Strathmore and Kinghorne', 'Thomas Lyon-Bowes, Lord Glamis', 'Thomas Lyon-Bowes, 12th Earl of Strathmore and Kinghorne', 'Thomas Lyon-Bowes, Master of Glamis (born 1821)', 'Charlotte Lyon-Bowes, Lady Glamis']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Claude Bowes-Lyon, 13th Earl of Strathmore and Kinghorne` sources `['proprag', 'dense']` graph count `2`

### 5caef2f8096411ebbdafac1f6bf848b6

- Question: Which film was released more recently, An Empress And The Warriors or Die Screaming, Marianne?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `Die Screaming, Marianne`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Die Screaming, Marianne', 'An Empress and the Warriors', 'The Warriors', 'Count Five and Die', 'The Complaint of an Empress']`
- Impl RRF top5: `['Die Screaming, Marianne', 'An Empress and the Warriors', 'The Warriors', 'Count Five and Die', 'The Complaint of an Empress']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Die Screaming, Marianne', 'An Empress and the Warriors', 'The Complaint of an Empress', 'The Warriors', 'Count Five and Die']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Die Screaming, Marianne` sources `['proprag', 'dense']` graph count `2`

### 0856e5d4085b11ebbd5cac1f6bf848b6

- Question: Which film has the director born later, Diary Of A Maniac or Return Of The Hero?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Return of the Hero`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Return of the Hero', 'Diary of a Maniac', 'Édouard Niermans (director)', 'Laurent Tirard', 'The Return of Swamp Thing']`
- Impl RRF top5: `['Return of the Hero', 'Diary of a Maniac', 'Édouard Niermans (director)', 'Laurent Tirard', 'The Return of Swamp Thing']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Return of the Hero', 'Diary of a Maniac', 'Marco Ferreri', 'Édouard Niermans (director)', 'Laurent Tirard']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Return of the Hero` sources `['proprag', 'dense']` graph count `2`

### 498dc8a2089111ebbd72ac1f6bf848b6

- Question: Which film has the director died later, Dahleez or The Lights Of Buenos Aires?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `The Lights of Buenos Aires`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Lights of Buenos Aires', 'Dahleez', 'Manuel Romero', 'Dasari Narayana Rao', 'Leopoldo Torres Ríos']`
- Impl RRF top5: `['The Lights of Buenos Aires', 'Dahleez', 'Manuel Romero', 'Dasari Narayana Rao', 'Leopoldo Torres Ríos']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Lights of Buenos Aires', 'Dahleez', 'Adelqui Migliar', 'Manuel Romero', 'Dasari Narayana Rao']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Lights of Buenos Aires` sources `['proprag', 'dense']` graph count `2`

### 5d3230dc0bde11eba7f7acde48001122

- Question: Where was the husband of Ting-Xing Ye born?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Ting-Xing Ye`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Ting-Xing Ye', 'Zhang Ye (singer)', 'Xia Ye', 'Zhang Ye (footballer, born 1989)', 'Lin Ye (chess player)']`
- Impl RRF top5: `['Ting-Xing Ye', 'Zhang Ye (singer)', 'Xia Ye', 'Zhang Ye (footballer, born 1989)', 'Lin Ye (chess player)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Ting-Xing Ye', 'Zhang Ye (singer)', 'Zhao Yanshou', 'Xia Ye', 'Zheng Junli']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Ting-Xing Ye` sources `['proprag', 'dense']` graph count `2`

### 350ec8b80bde11eba7f7acde48001122

- Question: Where was the composer of song By The Beautiful Sea (Song) born?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `By the Beautiful Sea (song)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['By the Beautiful Sea (song)', 'Harry Carroll', 'Where Was I', 'Mayor of Elizabeth, New Jersey', 'When I Was Young']`
- Impl RRF top5: `['By the Beautiful Sea (song)', 'Harry Carroll', 'Where Was I', 'Mayor of Elizabeth, New Jersey', 'When I Was Young']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['By the Beautiful Sea (song)', 'Harry Carroll', 'Leo Robin', 'Where Was I', 'The Trail of the Lonesome Pine (song)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `By the Beautiful Sea (song)` sources `['proprag', 'dense']` graph count `2`

### ee212a820bdc11eba7f7acde48001122

- Question: What is the place of birth of the director of film The Trail Of The Lonesome Pine (1923 Film)?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `The Trail of the Lonesome Pine (1916 film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Trail of the Lonesome Pine (1916 film)', 'The Trail of the Lonesome Pine (1923 film)', 'The Trail of the Lonesome Pine (1936 film)', 'The Trail of the Lonesome Pine', 'The Trail of the Lonesome Pine (song)']`
- Impl RRF top5: `['The Trail of the Lonesome Pine (1916 film)', 'The Trail of the Lonesome Pine (1923 film)', 'The Trail of the Lonesome Pine (1936 film)', 'The Trail of the Lonesome Pine', 'The Trail of the Lonesome Pine (song)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Trail of the Lonesome Pine (1916 film)', 'The Trail of the Lonesome Pine (1923 film)', 'Robert F. Boyle', 'The Trail of the Lonesome Pine (1936 film)', 'The Trail of the Lonesome Pine']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Trail of the Lonesome Pine (1916 film)` sources `['proprag', 'dense']` graph count `2`

### 337c710c08b511ebbd88ac1f6bf848b6

- Question: Who died first, Thomas Valentine Cooper or John Ellis Roosevelt?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `John Ellis Roosevelt`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['John Ellis Roosevelt', 'Thomas Valentine Cooper', 'T. V. Eddy', 'Thomas Cooper (bishop)', 'John Gilbert Cooper']`
- Impl RRF top5: `['John Ellis Roosevelt', 'Thomas Valentine Cooper', 'T. V. Eddy', 'Thomas Cooper (bishop)', 'John Gilbert Cooper']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['John Ellis Roosevelt', 'Thomas Valentine Cooper', 'Thomas Cooper (bishop)', 'Albert Alonzo Robinson', 'T. V. Eddy']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `John Ellis Roosevelt` sources `['proprag', 'dense']` graph count `2`

### c46f575c0bd911eba7f7acde48001122

- Question: Where was the place of death of Edsel Ford's father?
- doc_id overlap@20: `12`
- title overlap@20: `12`
- union size: `28`
- cross-pool docs: `12`
- PropRAG top1: `Edsel Ford`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Edsel Ford', 'Edsel Ford (poet)', 'Henry Ford', 'William Clay Ford Sr.', 'Edsel Ford (disambiguation)']`
- Impl RRF top5: `['Edsel Ford', 'Edsel Ford (poet)', 'Henry Ford', 'William Clay Ford Sr.', 'Edsel Ford (disambiguation)']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Edsel Ford', 'Edsel Ford (poet)', 'Edsel Ford (disambiguation)', 'Henry Ford', 'William Clay Ford Sr.']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Edsel Ford` sources `['proprag', 'dense']` graph count `2`

### 75668b40094111ebbdaeac1f6bf848b6

- Question: Which film came out earlier, A Jester'S Tale or Ang Kwento Ni Mabuti?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `A Jester's Tale`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Ang Kwento Ni Mabuti', "A Jester's Tale", 'Panday', 'A Tale of Springtime', 'Sidhi (film)']`
- Impl RRF top5: `["A Jester's Tale", 'Ang Kwento Ni Mabuti', 'Panday', 'A Tale of Springtime', 'Sidhi (film)']`
- RRF top5 match manual: `False`
- PropRAG rank top5 from union: `["A Jester's Tale", 'Ang Kwento Ni Mabuti', 'Where Was I', 'Panday', 'The Warriors']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `A Jester's Tale` sources `['proprag', 'dense']` graph count `2`

### 82de9a9f088911ebbd6eac1f6bf848b6

- Question: Which film has the director who was born later, The Sleeping City or The Lawful Cheater?
- doc_id overlap@20: `6`
- title overlap@20: `6`
- union size: `34`
- cross-pool docs: `6`
- PropRAG top1: `The Lawful Cheater`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['The Lawful Cheater', 'The Sleeping City', 'Mabel Cheung', 'André Cayatte', 'George Sherman']`
- Impl RRF top5: `['The Lawful Cheater', 'The Sleeping City', 'Mabel Cheung', 'André Cayatte', 'George Sherman']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['The Lawful Cheater', 'The Sleeping City', 'Mabel Cheung', "Frank O'Connor (actor)", 'André Cayatte']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `The Lawful Cheater` sources `['proprag', 'dense']` graph count `2`

### 5f2bb1060bd911eba7f7acde48001122

- Question: Which country the director of film La Yuma is from?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `La Yuma`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['La Yuma', 'Laurent Tirard', 'Luis Bayón Herrera', 'La Carapate', 'Felices 140']`
- Impl RRF top5: `['La Yuma', 'Laurent Tirard', 'Luis Bayón Herrera', 'La Carapate', 'Felices 140']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['La Yuma', 'Laurent Tirard', 'Felices 140', 'José Luis Garci', 'Luis Bayón Herrera']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `La Yuma` sources `['proprag', 'dense']` graph count `2`

### 92ae94d309bb11ebbdb0ac1f6bf848b6

- Question: Are both mountains, Allqamari and Juch'Uy Llallawa, located in the same country?
- doc_id overlap@20: `16`
- title overlap@20: `16`
- union size: `24`
- cross-pool docs: `16`
- PropRAG top1: `Allqamari`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Allqamari', "Juch'uy Llallawa", 'Jatun Urqu (Bolivia)', "Inka P'iqi", 'Puka Qallpa']`
- Impl RRF top5: `['Allqamari', "Juch'uy Llallawa", 'Jatun Urqu (Bolivia)', "Inka P'iqi", 'Puka Qallpa']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Allqamari', "Juch'uy Llallawa", "Inka P'iqi", 'Muru Qullu (Oruro)', 'Jatun Urqu (Bolivia)']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Allqamari` sources `['proprag', 'dense']` graph count `2`

### f6582e180bae11ebab90acde48001122

- Question: Who is Archduchess Maria Antonia Of Austria (1899–1977)'s maternal grandfather?
- doc_id overlap@20: `18`
- title overlap@20: `18`
- union size: `22`
- cross-pool docs: `18`
- PropRAG top1: `Archduchess Maria Antonia of Austria (1899–1977)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Archduchess Maria Antonia of Austria (1899–1977)', 'Archduke Anton of Austria', 'Archduke Franz Karl of Austria', 'Georg, Duke of Hohenberg', 'Archduchess Elisabeth Amalie of Austria']`
- Impl RRF top5: `['Archduchess Maria Antonia of Austria (1899–1977)', 'Archduke Anton of Austria', 'Archduke Franz Karl of Austria', 'Georg, Duke of Hohenberg', 'Archduchess Elisabeth Amalie of Austria']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Archduchess Maria Antonia of Austria (1899–1977)', 'Archduke Anton of Austria', 'Archduke Franz Karl of Austria', 'Georg, Duke of Hohenberg', 'Archduchess Elisabeth Amalie of Austria']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Archduchess Maria Antonia of Austria (1899–1977)` sources `['proprag', 'dense']` graph count `2`

### ea294116094111ebbdaeac1f6bf848b6

- Question: Which album was released first, Don'T Freak Me Out or Love & War (Barlowgirl Album)?
- doc_id overlap@20: `11`
- title overlap@20: `11`
- union size: `29`
- cross-pool docs: `11`
- PropRAG top1: `Love & War (BarlowGirl album)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Love & War (BarlowGirl album)', "Don't Freak Me Out", 'Let Me Out', 'Love & War (Daniel Merriweather album)', 'Take Me Out']`
- Impl RRF top5: `['Love & War (BarlowGirl album)', "Don't Freak Me Out", 'Let Me Out', 'Love & War (Daniel Merriweather album)', 'Take Me Out']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Love & War (BarlowGirl album)', "Don't Freak Me Out", 'Let Me Out', 'Love & War (Daniel Merriweather album)', 'Take Me Out']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Love & War (BarlowGirl album)` sources `['proprag', 'dense']` graph count `2`

### c4bdfbea08c911ebbd92ac1f6bf848b6

- Question: Was Ausaf Ali or Steven Perry born first?
- doc_id overlap@20: `9`
- title overlap@20: `9`
- union size: `31`
- cross-pool docs: `9`
- PropRAG top1: `Ausaf Ali`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `['Steven Perry', 'Ausaf Ali', 'Steve Miller', 'Steve Wilson', 'Stephen Hill']`
- Impl RRF top5: `['Steven Perry', 'Ausaf Ali', 'Steve Miller', 'Steve Wilson', 'Stephen Hill']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Ausaf Ali', 'Steven Perry', 'Steve Miller', 'Steve Wilson', 'Stephen Hill']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Ausaf Ali` sources `['proprag', 'dense']` graph count `2`

### 9a7f003a0bdb11eba7f7acde48001122

- Question: Who is the father of the director of film Summer Skin (Film)?
- doc_id overlap@20: `4`
- title overlap@20: `4`
- union size: `36`
- cross-pool docs: `4`
- PropRAG top1: `Summer Skin (film)`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Summer Skin (film)', 'Takayama Tomoteru', 'Gia Coppola', "Who's Your Daddy? (film)", 'Lars Eliasson']`
- Impl RRF top5: `['Summer Skin (film)', 'Takayama Tomoteru', 'Gia Coppola', "Who's Your Daddy? (film)", 'Lars Eliasson']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Summer Skin (film)', 'Leopoldo Torre Nilsson', 'Leopoldo Torres Ríos', 'Takayama Tomoteru', 'Killed the Family and Went to the Movies']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Summer Skin (film)` sources `['proprag', 'dense']` graph count `2`

### 6a4840d608dc11ebbd9cac1f6bf848b6

- Question: Which film has the director who is older, A Better Tomorrow Iii: Love & Death In Saigon or Har Dil Jo Pyar Karega?
- doc_id overlap@20: `15`
- title overlap@20: `15`
- union size: `25`
- cross-pool docs: `15`
- PropRAG top1: `A Better Tomorrow III: Love & Death in Saigon`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['A Better Tomorrow III: Love & Death in Saigon', 'Har Dil Jo Pyar Karega', 'A Better Tomorrow II', 'A Better Tomorrow', 'Tsui Hark']`
- Impl RRF top5: `['A Better Tomorrow III: Love & Death in Saigon', 'Har Dil Jo Pyar Karega', 'A Better Tomorrow II', 'A Better Tomorrow', 'Tsui Hark']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['A Better Tomorrow III: Love & Death in Saigon', 'A Better Tomorrow II', 'Har Dil Jo Pyar Karega', 'A Better Tomorrow', 'The Last Airbender']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `A Better Tomorrow III: Love & Death in Saigon` sources `['proprag', 'dense']` graph count `2`

### c213141e08e411ebbda4ac1f6bf848b6

- Question: Which film was released earlier, Flash Gordon Conquers The Universe or Siti Noerbaja?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Siti Noerbaja`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Siti Noerbaja', 'Flash Gordon Conquers the Universe', 'Purple Death from Outer Space', 'Flesh Gordon', "Flash Gordon's Trip to Mars"]`
- Impl RRF top5: `['Siti Noerbaja', 'Flash Gordon Conquers the Universe', 'Purple Death from Outer Space', 'Flesh Gordon', "Flash Gordon's Trip to Mars"]`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Siti Noerbaja', 'Flash Gordon Conquers the Universe', 'Purple Death from Outer Space', 'Oei Tiong Ham', 'Flesh Gordon']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Siti Noerbaja` sources `['proprag', 'dense']` graph count `2`

### 7f95fed108f511ebbdaaac1f6bf848b6

- Question: Who was born first out of C. Maurice Patterson and Maj-Briht Bergström-Walan?
- doc_id overlap@20: `7`
- title overlap@20: `7`
- union size: `33`
- cross-pool docs: `7`
- PropRAG top1: `Maj-Briht Bergström-Walan`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Maj-Briht Bergström-Walan', 'C. Maurice Patterson', 'Steve Cooreman', 'Michał Bergson', 'Dugès']`
- Impl RRF top5: `['Maj-Briht Bergström-Walan', 'C. Maurice Patterson', 'Steve Cooreman', 'Michał Bergson', 'Dugès']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Maj-Briht Bergström-Walan', 'C. Maurice Patterson', 'Steve Cooreman', 'David Elliot Loye', 'Michał Bergson']`
- support_complete manual/reported: `1.0` / `1.0`
- Shared doc: `Maj-Briht Bergström-Walan` sources `['proprag', 'dense']` graph count `2`

### a488d6e2089211ebbd73ac1f6bf848b6

- Question: Which film has the director who died later, Majhli Didi or Dream Of The Rhine?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Dream of the Rhine`
- PropRAG top1 manual RRF rank: `1`
- Manual RRF top5: `['Dream of the Rhine', 'Majhli Didi', 'Mej Didi (1950 film)', 'Zheng Junli', 'Hrishikesh Mukherjee']`
- Impl RRF top5: `['Dream of the Rhine', 'Majhli Didi', 'Mej Didi (1950 film)', 'Zheng Junli', 'Hrishikesh Mukherjee']`
- RRF top5 match manual: `True`
- PropRAG rank top5 from union: `['Dream of the Rhine', 'Majhli Didi', 'Hrishikesh Mukherjee', 'Mej Didi (1950 film)', 'Zheng Junli']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Dream of the Rhine` sources `['proprag', 'dense']` graph count `2`

### 8540ae66088011ebbd6cac1f6bf848b6

- Question: Which film has the director who was born later, Tarzan'S Hidden Jungle or The Brink'S Job?
- doc_id overlap@20: `8`
- title overlap@20: `8`
- union size: `32`
- cross-pool docs: `8`
- PropRAG top1: `Tarzan's Hidden Jungle`
- PropRAG top1 manual RRF rank: `2`
- Manual RRF top5: `["The Brink's Job", "Tarzan's Hidden Jungle", 'Jack Arnold (director)', 'Lewis Milestone', 'Gordon Scott']`
- Impl RRF top5: `["Tarzan's Hidden Jungle", "The Brink's Job", 'Jack Arnold (director)', 'Lewis Milestone', 'Gordon Scott']`
- RRF top5 match manual: `False`
- PropRAG rank top5 from union: `["Tarzan's Hidden Jungle", "The Brink's Job", 'The Son of Tarzan (film)', 'Jack Arnold (director)', 'The Revenge of Tarzan']`
- support_complete manual/reported: `0.0` / `0.0`
- Shared doc: `Tarzan's Hidden Jungle` sources `['proprag', 'dense']` graph count `2`

