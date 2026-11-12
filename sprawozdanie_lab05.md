# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** Laboratorium 5 — Wpływ korpusu na tokenizację BPE i embeddingi Word2Vec

**Autor / grupa:** Bartosz Paluda

**Data:** 12.11.2026

## 1. Cel ćwiczenia

Celem ćwiczenia jest empiryczne zbadanie wpływu struktury i rozmiaru słownika na algorytm tokenizacji podsłownej BPE (Byte Pair Encoding) oraz wpływu rozkładu kontekstów wieloznacznego słowa (homonimu) na reprezentacje wektorowe uczone metodą Word2Vec (Skip-gram). Analizie poddano również separowalność grup tematycznych w przestrzeni osadzeń na podstawie rzutu PCA 2D oraz metryk podobieństwa cosinusowego.

## 2. Metoda i implementacja

### Specyfikacja korpusu i preprocessing

* **Źródło danych:** Autorski korpus w języku polskim liczący **60 unikalnych zdań**, podzielony na trzy równoliczne pule po 20 zdań:
  1. **Pula 1 (Znaczenie 1 – budowla):** 20 zdań opisujących średniowieczną warownię (np. *„Stary zamek stoi na wysokim wzgórzu nad rzeką”*).
  2. **Pula 2 (Znaczenie 2 – mechanizm):** 20 zdań opisujących mechanizm zamykający (np. *„Ślusarz naprawił zepsuty zamek w ciężkich drzwiach”*).
  3. **Pula 3 (Zdania neutralne):** 20 zdań niezawierających słowa *zamek*, zawierających słownictwo kontekstowe z obu domen oraz tematy ogólne.

* **Badany wyraz (homonim):** `zamek` — występuje dokładnie 1 raz w każdym zdaniu z Puli 1 i Puli 2 oraz 0 razy w Puli 3.

* **Normalizacja tekstu:** Sprowadzenie wszystkich znaków do małych liter (`lower()`) oraz usunięcie znaków interpunkcyjnych przed treningiem i ewaluacją.

* **Próbka testowa BPE (6 par form fleksyjnych):** Zestawiono formy występujące w korpusie z nieobecnymi formami fleksyjnymi tego samego leksemu (weryfikacja na znormalizowanym tekście): `zamek`–`zamkom`, `drzwiach`–`drzwiami`, `mury`–`murami`, `klucz`–`kluczami`, `turyści`–`turystami`, `ślusarz`–`ślusarzom`.

### Konfiguracja eksperymentów

* **Tokenizator BPE (`tokenizers`):** Pretokenizer `Whitespace`, `min_frequency=1`, token nieznany `[UNK]`, badane limity `vocab_size` $\in \{120, 250, 500\}$.

* **Model Word2Vec (`gensim`):** Architektura Skip-gram (`sg=1`), wymiar osadzeń `vector_size=50`, okno kontekstu `window=4`, `min_count=5`, `epochs=100`, `seed=0`.
  * **Wariant 1 (dominacja budowli):** 20 zdań neutralnych + 16 zdań z Puli 1 + 4 zdania z Puli 2 (łącznie 40 unikalnych zdań, 20 wystąpień słowa *zamek*, powielone 10-krotnie dla stabilizacji wag).
  * **Wariant 2 (dominacja mechanizmu):** 20 zdań neutralnych + 4 zdania z Puli 1 + 16 zdań z Puli 2 (identyczna liczba zdań i identyczna liczba wystąpień słowa *zamek*).

## 3. Wyniki

### 1. Eksperyment BPE — wpływ rozmiaru słownika na tokenizację

* **Hipoteza badawcza:** Zwiększenie limitu `vocab_size` spowoduje spadek średniej liczby tokenów na słowo dla obu grup. Formy widziane w treningu osiągną docelowo długość 1 tokenu, natomiast formy niewidziane nadal będą dzielone na rdzeń i końcówkę fleksyjną, bez generowania tokenów `[UNK]` dzięki pełnemu pokryciu alfabetu w bazie słownika.

| Limit `vocab_size` | Faktyczny rozmiar słownika $|V|$ | Średnia tokenów (widziane) | Liczba `[UNK]` (widziane) | Średnia tokenów (niewidziane) | Liczba `[UNK]` (niewidziane) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **120** | 120 | 2.83 | 0 | 4.00 | 0 |
| **250** | 250 | 1.17 | 0 | 3.00 | 0 |
| **500** | 500 | **1.00** | 0 | **2.83** | 0 |

#### Szczegółowy podział dwóch niewidzianych form dla wszystkich limitów:

* **`zamkom`:**
  * `limit=120` ($|V|=120$): `['zam', 'ko', 'm']` (3 tokeny)
  * `limit=250` ($|V|=250$): `['zam', 'ko', 'm']` (3 tokeny)
  * `limit=500` ($|V|=500$): `['zam', 'kom']` (2 tokeny — scalenie końcówki `ko` + `m` w `kom`)

* **`ślusarzom`:**
  * `limit=120` ($|V|=120$): `['ś', 'lu', 'sa', 'rz', 'o', 'm']` (6 tokenów — rozbicie niemal na pojedyncze znaki)
  * `limit=250` ($|V|=250$): `['ślusarz', 'o', 'm']` (3 tokeny — scalenie całego rdzenia `ślusarz`)
  * `limit=500` ($|V|=500$): `['ślusarz', 'o', 'm']` (3 tokeny)

### 2. Eksperyment Word2Vec — porównanie sąsiadów słowa `'zamek'`

| Pozycja | Wariant 1 (16 budowla / 4 mechanizm) | Cosinus (W1) | Obecne w W2? | Wariant 2 (4 budowla / 16 mechanizm) | Cosinus (W2) | Obecne w W1? |
|:---:|---|:---:|:---:|---|:---:|:---:|
| **1** | `ten` | 0.4193 | TAK | `stary` | 0.4119 | TAK |
| **2** | `stary` | 0.3828 | TAK | `nowy` | 0.3759 | TAK |
| **3** | `wokół` | 0.3659 | NIE | `wymaga` | 0.3672 | NIE |
| **4** | `woda` | 0.3633 | NIE | `drzwiach` | 0.3586 | TAK |
| **5** | `głęboka` | 0.3607 | NIE | `solidny` | 0.3564 | TAK |

### 3. Eksperyment PCA — analiza grup tematycznych (Model 1)

* **Wybrane grupy słów (przed obejrzeniem wyników):**
  * **Grupa 1 (Budowla):** `['zabytkowy', 'mury', 'wieża', 'król', 'turyści']` — rzeczowniki i przymiotniki opisujące architekturę obronną oraz ruch turystyczny.
  * **Grupa 2 (Mechanizm):** `['ślusarz', 'drzwiach', 'klucz', 'stary', 'klamkę']` — osoby, elementy stolarki drzwiowej i cechy związane z zamkami drzwiowymi.

* **Średnie podobieństwo cosinusowe (na pełnych 50-wymiarowych wektorach):**
  * Wewnątrz Grupy 1 (Budowla): **0.3823**
  * Wewnątrz Grupy 2 (Mechanizm): **0.4475**
  * Między Grupą 1 a Grupą 2: **0.3414**

<img width="887" height="590" alt="zamek" src="https://github.com/user-attachments/assets/b8076536-969e-4e18-9801-65efb47b12b0" />


## 4. Analiza i interpretacja

### 1. Tokenizacja BPE

* Postawiona hipoteza została w pełni potwierdzona. Przy zwiększaniu słownika ze 120 do 500 tokenów średnia długość form widzianych spadła z **2.83 do 1.00** (wszystkie 6 form treningowych trafiło do słownika w całości), natomiast dla form niewidzianych spadła z **4.00 do 2.83**.
* W obu badanych wyrazach niewidzianych (`zamkom`, `ślusarzom`) ani razu nie wystąpił token `[UNK]`, ponieważ wszystkie znaki języka polskiego występowały w zbiorze treningowym.
* W słowie `ślusarzom` zwiększenie limitu ze 120 do 250 pozwoliło scalić rozdrobnione sylaby (`['ś', 'lu', 'sa', 'rz']`) w jeden rdzeń `'ślusarz'`, który pochodzi bezpośrednio ze zdań treningowych (np. *„Ślusarz naprawił zepsuty zamek...”*). Końcówka celownika `-om` nie wystąpiła wystarczająco często, aby scalić się w jeden token `om`. W przypadku `zamkom` podział `['zam', 'kom']` wynika z obecności rdzenia `zam` (z form *zamek*, *zamontował*, *zamknięciu*) oraz zbitki `kom` (obecnej np. w słowach *komory*, *komnaty*, *komputerowa*).

### 2. Zmiana rankingu sąsiadów w Word2Vec

* Zmiana proporcji zdań między Wariantem 1 a Wariantem 2 bezpośrednio przesunęła wektor słowa `zamek` w przestrzeni semantycznej:
  * **W Wariancie 1 (dominacja budowli)** w czołowej piątce znalazły się słowa `wokół` (0.3659), `woda` (0.3633) i `głęboka` (0.3607), pochodzące bezpośrednio ze zdania *„Wokół fosy ten zamek otacza głęboka woda”*. Po przejściu do Wariantu 2 słowa te wypadły z rankingu i **nie posiadają nawet wektorów w Modelu 2** (`NIE`), ponieważ zdanie to znajdowało się na pozycji 5. w Puli 1 i nie weszło do pierwszych 4 zdań wybranych do Wariantu 2 (nie przekroczyły progu `min_count=5`).
  * **W Wariancie 2 (dominacja mechanizmu)** na szczyt rankingu awansowały przymiotniki `nowy` (0.3759) i `solidny` (0.3564) oraz rzeczownik `drzwiach` (0.3586), co wynika z licznych zdań z Puli 2 (*„Nowy zamek antywłamaniowy...”*, *„Wymieniłem stary zamek na solidny model...”*, *„Ślusarz naprawił zepsuty zamek w ciężkich drzwiach”*).
  * Przymiotnik `stary` utrzymał wysoką pozycję w obu wariantach (2. miejsce w W1 z $\cos=0.3828$, 1. miejsce w W2 z $\cos=0.4119$), ponieważ pojawia się jako naturalny epitet w obu domenach (*stary zamek na wzgórzu* oraz *stary zamek w bramie*).

### 3. Separacja grup tematycznych na rzucie PCA 2D

* Zarówno wykres PCA 2D, jak i średnie wartości podobieństwa cosinusowego potwierdzają wyraźne rozdzielenie grup tematycznych:
  * Średnie podobieństwo wewnątrz Grupy 1 (**0.3823**) oraz wewnątrz Grupy 2 (**0.4475**) jest wyraźnie wyższe niż podobieństwo między grupami (**0.3414**).
  * Na wykresie PCA większość słów z Grupy 2 (`ślusarz`, `klamkę`, `drzwiach`) skupia się w dolnej części wykresu ($PC2 \in [-2.2, -1.7]$), podczas gdy pojęcia architektoniczne (`wieża`, `mury`, `zabytkowy`) zajmują górną część przestrzeni ($PC2 > 0.8$).
  * **Uzasadnienie na konkretnej parze słów:** Słowa **`klamkę`** oraz **`ślusarz`** (Grupa 2) nakładają się niemal w tym samym punkcie w lewym dolnym rogu wykresu ($PC1 \approx -0.45, PC2 \approx -2.15$). Wynika to z faktu, że w korpusie występują w bardzo bliskim sąsiedztwie kontekstowym wokół drzwi i napraw (np. *„Klamka i zamek w drzwiach...”* oraz *„Fachowiec naprawił klamkę... Ślusarz naprawił zepsuty zamek...”*), przez co algorytm Skip-gram przypisał im niemal identyczne wektory.
  * Z kolei badany wyraz **`ZAMEK`** (złota gwiazdka w prawym górnym rogu, $PC1 \approx 1.5, PC2 \approx 1.1$) leży znacznie bliżej słów `zabytkowy` i `turyści` niż klastru `ślusarz`–`klamkę`, co wizualnie potwierdza, że w Modelu 1 jego reprezentacja została zdominowana przez znaczenie budowli. Zjawiskiem wartym odnotowania jest położenie słowa `stary` (czerwony punkt u góry) blisko klastra budowli — wynika to z jego dwuznacznego użycia w obu pulach zdań.

## 5. Wnioski

1. **Tokenizacja BPE w języku fleksyjnym:** Zwiększenie limitu słownika BPE skutecznie kompresuje znane wyrazy do pojedynczych tokenów, a nieznane odmiany fleksyjne rozbija na wyuczone w treningu rdzenie i sufiksy (np. `ślusarz` + `o` + `m`), całkowicie eliminując problem tokenów `[UNK]` dla znanego alfabetu.
2. **Wrażliwość Word2Vec na rozkład kontekstów:** Statyczny model Word2Vec tworzy pojedynczy wektor dla homonimu, stanowiący wypadkową wszystkich jego kontekstów w korpusie. Zmiana proporcji zdań (16:4 vs 4:16) całkowicie przebudowała listę najbliższych sąsiadów słowa *zamek*, przesuwając jego reprezentację z kontekstu historyczno-turystycznego w stronę ślusarskiego.
3. **Spójność geometryczna przestrzeni osadzeń:** Pomimo niewielkiego rozmiaru korpusu i redukcji wymiarowości z 50D do 2D, rzut PCA zachował strukturę semantyczną danych — wyrazy współwystępujące w podobnych kontekstach tworzą zwarte klastry o wyższym wewnętrznym podobieństwie cosinusowym niż pary międzygrupowe.
