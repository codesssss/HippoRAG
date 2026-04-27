# CPAG Implementation Audit

## Summary

- Sample queries: `5`
- Avg doc_id overlap@20: `9.4`
- Avg title overlap@20: `9.4`
- PropRAG top1 manual RRF ranks: `[1, 1, 1, 2, 1]`
- All checks pass: `True`
- Failure count: `0`

## Failures

- None

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

