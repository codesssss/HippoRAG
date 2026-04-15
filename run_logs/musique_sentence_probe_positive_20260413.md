# Sentence Attribution Probe (musique)

- repair report: `outputs_step0_general_musique/eval_reports/width_match_bridge_append_plus_ce_local_repair_qatopk5_20260413smoke.json`
- case mode: `applied_positive`
- cases analyzed: `4`
- CE model: `/mnt/nvme/bge-reranker-v2-m3` on `cuda:7`

## Aggregate

- candidate top2 softmax mass mean / median: `0.7954` / `0.7989`
- candidate top2 token share mean: `0.3004`
- candidate localized rate (top2 mass >= 0.6): `100.0`
- candidate any goldish sentence rate: `75.0`
- candidate top2 goldish sentence rate: `75.0`
- candidate goldish only outside top2 rate: `0.0`

## Cases

### Modern history -> replace Korean War

- question: `Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?`
- delta EM / F1: `1.0` / `1.0`
- CE gap (candidate - replaced): `-5.6719`
- connector / anchor gain: `2` / `0`
- candidate top2 mass / token share: `0.9672` / `0.3186`
- candidate goldish any/top2/outside-top2-only: `True` / `True` / `False`
- replaced top2 mass / token share: `0.922` / `0.197`

- candidate top sentences:
  [score=-4.6758, mass=0.8825, q=3, qe=3, gt=0, ga=1] There was a shift in power from Western Europe and the British Empire to the two new superpowers, the United States and the Soviet Union.
  [score=-7.0195, mass=0.0847, q=3, qe=2, gt=1, ga=1] These two rivals would later face off in the Cold War.
  [score=-9.2344, mass=0.0092, q=1, qe=1, gt=0, ga=1] The former colonies of the European powers began their road to independence.
- replaced top sentences:
  [score=-4.3828, mass=0.6859, q=5, qe=3, gt=1, ga=1] Mao was concerned that the Americans would intervene but agreed to support the North Korean invasion.
  [score=-5.4492, mass=0.2361, q=1, qe=2, gt=1, ga=0] Once Mao's commitment was secured, preparations for war accelerated.
  [score=-7.2812, mass=0.0378, q=3, qe=3, gt=1, ga=1] However, Mao sent more ethnic Korean PLA veterans to Korea and promised to move an army closer to the Korean border.

### Delhi -> replace Indian Rebellion of 1857

- question: `How did did the people fare during the reign of the abolisher of sati partha in India?`
- delta EM / F1: `0.0` / `0.1667`
- CE gap (candidate - replaced): `-1.039`
- connector / anchor gain: `1` / `0`
- candidate top2 mass / token share: `0.9624` / `0.3118`
- candidate goldish any/top2/outside-top2-only: `True` / `True` / `False`
- replaced top2 mass / token share: `0.691` / `0.2791`

- candidate top sentences:
  [score=-5.2812, mass=0.9421, q=4, qe=0, gt=0, ga=2] During the partition of India, thousands of Hindu and Sikh refugees, mainly from West Punjab fled to Delhi, while many Muslim residents of the city migrated to Pakistan.
  [score=-9.1172, mass=0.0203, q=4, qe=1, gt=0, ga=3] During the Indian Rebellion of 1857, Delhi fell to the forces of East India Company after a bloody fight known as the Siege of Delhi.
  [score=-9.2344, mass=0.0181, q=3, qe=0, gt=0, ga=2] New Delhi, also known as Lutyens' Delhi, was officially declared as the capital of the Union of India after the country gained independence on 15 August 1947.
- replaced top sentences:
  [score=-6.9844, mass=0.4763, q=3, qe=0, gt=0, ga=2] Other regions of Company - controlled India -- the Bengal Presidency, the Bombay Presidency and the Madras Presidency -- remained largely calm.
  [score=-7.7812, mass=0.2147, q=3, qe=0, gt=0, ga=2] After the outbreak of the mutiny in Meerut, the rebels very quickly reached Delhi and declared its 81 - year - old Mughal ruler, Bahadur Shah Zafar, as Emperor of Hindustan.
  [score=-8.0781, mass=0.1595, q=4, qe=0, gt=0, ga=4] The large princely states (Hyderabad, Mysore, Travancore, and Kashmir), as well as the smaller ones of Rajputana, did not join the rebellion, serving the British, in the words of Governor - General Lord Canning, as ``breakwaters in a storm.

### Houston Astros -> replace 2017 World Series

- question: `Who had the lowest batting average in the league where the team with the most games in the series after which the MLB MVP is awarded played?`
- delta EM / F1: `1.0` / `1.0`
- CE gap (candidate - replaced): `-3.7217`
- connector / anchor gain: `6` / `0`
- candidate top2 mass / token share: `0.6353` / `0.2446`
- candidate goldish any/top2/outside-top2-only: `True` / `True` / `False`
- replaced top2 mass / token share: `0.9587` / `0.5256`

- candidate top sentences:
  [score=-5.2617, mass=0.3418, q=4, qe=2, gt=3, ga=0] The Astros defeated the Red Sox three games to one, and advanced to the American League Championship Series against the New York Yankees.
  [score=-5.4141, mass=0.2935, q=3, qe=5, gt=2, ga=0] The Astros won the ALCS four games to three, and advanced to the World Series to play against the Los Angeles Dodgers.
  [score=-6.332, mass=0.1172, q=2, qe=2, gt=1, ga=0] The Astros clinched their first division title as a member of the American League West division, and first division title overall since 2001.
- replaced top sentences:
  [score=-3.1777, mass=0.5158, q=4, qe=6, gt=3, ga=0] The 2017 World Series was the championship series of Major League Baseball's (MLB) 2017 season.
  [score=-3.3301, mass=0.4429, q=3, qe=6, gt=1, ga=0] The series was a best - of - seven playoff between the National League (NL) champion Los Angeles Dodgers and the American League (AL) champion Houston Astros.
  [score=-6.1602, mass=0.0261, q=3, qe=3, gt=2, ga=0] The 113th edition of the World Series, it was played between October 24 and November 1.

### Allies of World War II -> replace Pacific War

- question: `What is an example of a railroad line in the country first to invade Manchuria?`
- delta EM / F1: `0.0` / `0.0385`
- CE gap (candidate - replaced): `0.2891`
- connector / anchor gain: `10` / `0`
- candidate top2 mass / token share: `0.6165` / `0.3268`
- candidate goldish any/top2/outside-top2-only: `False` / `False` / `False`
- replaced top2 mass / token share: `0.9114` / `0.5169`

- candidate top sentences:
  [score=-8.7578, mass=0.3572, q=2, qe=2, gt=0, ga=0] At the start of the war on 1 September 1939, the Allies consisted of France, Poland and the United Kingdom, as well as their dependent states, such as British India.
  [score=-9.0781, mass=0.2593, q=2, qe=2, gt=0, ga=0] Within days they were joined by the independent Dominions of the British Commonwealth: Australia, Canada, New Zealand and South Africa.
  [score=-9.7578, mass=0.1314, q=2, qe=2, gt=0, ga=0] After the start of the German invasion of North Europe until the Balkan Campaign, the Netherlands, Belgium, Greece, and Yugoslavia joined the Allies.
- replaced top sentences:
  [score=-7.5703, mass=0.8387, q=4, qe=3, gt=0, ga=1] In September 1940, Japan decided to cut China's only land line to the outside world by seizing Indochina, which was controlled at the time by Vichy France.
  [score=-10.0156, mass=0.0727, q=2, qe=2, gt=0, ga=0] In practice, there was little coordination between Japan and Germany until 1944, by which time the U.S.
  [score=-10.7812, mass=0.0338, q=3, qe=2, gt=0, ga=0] On 27 September Japan signed a military alliance with Germany and Italy, becoming one of the three Axis Powers.
