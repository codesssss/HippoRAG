# Sentence Attribution Probe (2wikimultihopqa)

- repair report: `outputs_step0_general_2wikimultihopqa/eval_reports/width_match_bridge_append_plus_ce_local_repair_qatopk5_20260413smoke.json`
- case mode: `applied_nonpositive`
- cases analyzed: `4`
- CE model: `/mnt/nvme/bge-reranker-v2-m3` on `cuda:7`

## Aggregate

- candidate top2 softmax mass mean / median: `0.8704` / `0.9241`
- candidate top2 token share mean: `0.2273`
- candidate localized rate (top2 mass >= 0.6): `100.0`
- candidate any goldish sentence rate: `100.0`
- candidate top2 goldish sentence rate: `75.0`
- candidate goldish only outside top2 rate: `25.0`

## Cases

### Alicia Keys -> replace Alex da Kid

- question: `What is the place of birth of the performer of song Changed It?`
- delta EM / F1: `0.0` / `0.0`
- CE gap (candidate - replaced): `0.4033`
- connector / anchor gain: `4` / `0`
- candidate top2 mass / token share: `0.9882` / `0.1158`
- candidate goldish any/top2/outside-top2-only: `True` / `True` / `False`
- replaced top2 mass / token share: `0.9275` / `0.1875`

- candidate top sentences:
  [score=-1.2793, mass=0.9842, q=1, qe=0, gt=0, ga=0] Alicia Augello Cook Dean (born January 25, 1981), known professionally as Alicia Keys, is an American musician, singer, and songwriter.
  [score=-6.793, mass=0.004, q=2, qe=0, gt=0, ga=1] Her second album, "The Diary of Alicia Keys" (2003), was also a critical and commercial success, spawning successful singles "You Don't Know My NameIf I Ain't Got You", and "Diary", and selling eight million copies worldwide.
  [score=-7.6719, mass=0.0016, q=2, qe=0, gt=0, ga=1] Keys has received numerous accolades in her career, including 15 competitive Grammy Awards, 17 NAACP Image Awards, 12 ASCAP Awards, and an award from the Songwriters Hall of Fame and National Music Publishers Association.
- replaced top sentences:
  [score=-2.9648, mass=0.8765, q=1, qe=0, gt=0, ga=0] Alexander Junior Grant (born 7 August 1982), professionally known as Alex da Kid, is a British record producer, songwriter, and record executive from Wood Green, London.
  [score=-5.8086, mass=0.051, q=2, qe=0, gt=0, ga=1] Even though he now lives in Los Angeles, the "Evening Standard" named him one of "London's Most Influential People in 2011.
  [score=-6.2852, mass=0.0317, q=1, qe=0, gt=0, ga=0] The track was written and produced by Alex da Kid for KIDinaKORNER, Sam Harris, Casey Harris, Adam Levin, Elle King, and Wiz Khalifa.

### Alicia Keys -> replace Astrid North

- question: `What nationality is the performer of song When The Stars Go Blue?`
- delta EM / F1: `0.0` / `0.0`
- CE gap (candidate - replaced): `1.8833`
- connector / anchor gain: `3` / `0`
- candidate top2 mass / token share: `0.9932` / `0.1158`
- candidate goldish any/top2/outside-top2-only: `True` / `True` / `False`
- replaced top2 mass / token share: `0.9997` / `0.7778`

- candidate top sentences:
  [score=0.0184, mass=0.9905, q=1, qe=0, gt=0, ga=0] Alicia Augello Cook Dean (born January 25, 1981), known professionally as Alicia Keys, is an American musician, singer, and songwriter.
  [score=-5.8945, mass=0.0027, q=2, qe=1, gt=1, ga=0] Her second album, "The Diary of Alicia Keys" (2003), was also a critical and commercial success, spawning successful singles "You Don't Know My NameIf I Ain't Got You", and "Diary", and selling eight million copies worldwide.
  [score=-6.2031, mass=0.002, q=0, qe=0, gt=0, ga=0] Her sixth studio album, "Here" (2016), became her seventh US R&B/ Hip-Hop chart topping album.
- replaced top sentences:
  [score=-2.3301, mass=0.9769, q=0, qe=0, gt=0, ga=0] Astrid North( Astrid Karina North Radmann; 24 August 1973, Berlin – 25 June 2019, Berlin) was a German soul singer and songwriter.
  [score=-6.0898, mass=0.0228, q=2, qe=1, gt=1, ga=0] She was the singer of the German band, with whom she released five Albums.
  [score=-10.4141, mass=0.0003, q=2, qe=1, gt=1, ga=0] As guest singer of the band she published three albums.

### Roman Polanski -> replace Aleksander Ford

- question: `Which country Aleksander Koniecpolski (1620–1659)'s father is from?`
- delta EM / F1: `0.0` / `0.0`
- CE gap (candidate - replaced): `0.4453`
- connector / anchor gain: `1` / `0`
- candidate top2 mass / token share: `0.6403` / `0.1102`
- candidate goldish any/top2/outside-top2-only: `True` / `False` / `True`
- replaced top2 mass / token share: `0.9806` / `0.5338`

- candidate top sentences:
  [score=-5.0586, mass=0.4556, q=0, qe=0, gt=0, ga=0] Two years later, Poland was invaded by Nazi Germany starting World War II and the Polanskis found themselves trapped in the Kraków Ghetto.
  [score=-5.9609, mass=0.1848, q=1, qe=0, gt=0, ga=0] Polanski's first feature- length film," Knife in the Water"( 1962), was made in Poland and was nominated for a United States Academy Award for Best Foreign Language Film.
  [score=-6.0234, mass=0.1736, q=1, qe=0, gt=0, ga=1] Roman Polański( born 18 August 1933 in Paris; original name Raymond Thierry Liebling) is a French- Polish film director, producer, writer, and actor.
- replaced top sentences:
  [score=-5.4141, mass=0.8807, q=1, qe=0, gt=0, ga=0] Following the anti-Semitic purge in the communist party in Poland, in 1968 Ford emigrated to Israel and from there through Germany and Denmark, to the United States.
  [score=-7.5898, mass=0.1, q=2, qe=1, gt=1, ga=1] Aleksander Ford (born Mosze Lifszyc; 24 November 1908 in Kiev, Russian Empire – 4 April 1980 in Naples, Florida, United States) was a Polish film director; and head of the Polish People's Army Film Crew in the Soviet Union during World War II.
  [score=-10.7109, mass=0.0044, q=0, qe=0, gt=0, ga=0] He committed suicide in 1980 in Naples, Florida.

### Alain Corneau -> replace A Night at the Moulin Rouge

- question: `Are both movies, Naked Tango and Algiers (Film), from the same country?`
- delta EM / F1: `0.0` / `0.0`
- CE gap (candidate - replaced): `1.8789`
- connector / anchor gain: `1` / `0`
- candidate top2 mass / token share: `0.86` / `0.5673`
- candidate goldish any/top2/outside-top2-only: `True` / `True` / `False`
- replaced top2 mass / token share: `1.0` / `1.0`

- candidate top sentences:
  [score=-7.9492, mass=0.7476, q=2, qe=1, gt=1, ga=0] Alain Corneau (7 August 1943 – 30 August 2010) was a French film director and writer.
  [score=-9.8438, mass=0.1124, q=2, qe=0, gt=0, ga=0] Originally a musician, he worked with Costa-Gavras as an assistant, which was also his first opportunity to work with the actor Yves Montand, with whom he would collaborate three times later in his career, including "Police Python 357 " (1976) and "La Menace" (1977).
  [score=-10.5, mass=0.0583, q=1, qe=0, gt=0, ga=0] He directed Gérard Depardieu in the screen adaptation of "Tous les matins du monde" in 1991.
- replaced top sentences:
  [score=-5.3867, mass=0.9743, q=2, qe=1, gt=1, ga=0] Much of the film is portrayed as taking place in the Moulin Rouge cabaret nightclub in Paris.
  [score=-9.0234, mass=0.0257, q=3, qe=1, gt=1, ga=0] A Night at the Moulin Rouge( French: Une nuit au Moulin- Rouge) is a 1957 French comedy film directed by Jean- Claude Roy and starring Tilda Thamar, Noël Roquevert and Jean Tissier.
