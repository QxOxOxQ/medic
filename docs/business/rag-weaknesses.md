# Slabe Strony Implementacji RAG

## Cel Dokumentu

Ten dokument opisuje znane slabe strony implementacji RAG w projekcie Medic RAG. Kazdy punkt jest zapisany jako osobny temat do pozniejszej pracy: problem, skutek, miejsca w kodzie, sugerowany kierunek naprawy i kryteria ukonczenia.

Zakres dotyczy przeplywu: przygotowanie PDF, chunking, embedding, indeksowanie w Qdrant, retrieval, filtrowanie wynikow po uzytkowniku oraz odpowiedz agenta z cytowaniami.

## 1. Produkcyjny Searcher Nie Uzywa Hybrydowego RRF

**Status (2026-06-28): ROZWIAZANE.** `Searcher.search()` deleguje do `Qdrant.hybrid_search_with_rrf()`, ktory robi dense prefetch + sparse (BM25) prefetch + `Fusion.RRF`. Potwierdzone testem `test_qdrant_hybrid_search_uses_dense_sparse_prefetch_and_rrf`. Opis ponizej zachowany jako kontekst historyczny.

Problem: aplikacja tworzy kolekcje Qdrant z wektorem dense i sparse, a indeksowanie zapisuje oba typy danych, ale glowna sciezka wyszukiwania uzywana przez backend odpytuje tylko dense vector. Metoda hybrydowa z RRF istnieje, lecz nie jest podlaczona do `Searcher`.

Skutek: deklarowana hybrydowosc jest w praktyce niepelna. Zapytania zawierajace rzadkie terminy medyczne, nazwy badan, kody, wartosci laboratoryjne albo konkretne slowa z dokumentu moga miec gorszy recall niz oczekiwany po indeksie dense+sparse.

Miejsca w kodzie:

- `rag/searcher.py` - `Searcher.search()` buduje pojedyncze dense query.
- `rag/qdrant.py` - `hybrid_search_with_rrf()` implementuje dense+sparse prefetch i RRF, ale nie jest glowna sciezka.
- `rag/indexer.py` - `_index_chunks()` zapisuje sparse vector przez `models.Document`, jesli `sparse_vector_name` jest ustawione.

Kierunek naprawy:

- Zmienic `Searcher.search()` tak, aby uzywal hybrydowego searcha dense+sparse z RRF.
- Upewnic sie, ze wynik ma ten sam kontrakt, ktorego oczekuje `RetrievalService`.
- Zachowac mozliwosc testowania Searchera przez wstrzykniecie klienta lub providera.

Kryteria ukonczenia:

- Test jednostkowy potwierdza, ze produkcyjny `Searcher` wysyla do Qdranta dense prefetch, sparse prefetch i `Fusion.RRF`.
- Test API/agenta nadal przechodzi bez zmian w kontrakcie zrodel.
- W dokumentacji nie trzeba juz traktowac hybrydowego RAG jako niepodlaczonej funkcji.

## 2. Filtrowanie Uzytkownika Dzieje Sie Dopiero Po Wyszukaniu W Qdrant

**Status (2026-06-28): ROZWIAZANE.** Indeksowanie zapisuje `owner_user_id` do payloadu punktu (z indeksem KEYWORD), a `hybrid_search_with_rrf()` naklada `Filter` po `owner_user_id` na kazdy `Prefetch` — filtr jest liczony przed top-k, a nie po nim. Filtrowanie po stronie PostgreSQL zachowano jako druga linie obrony. Izolacje potwierdzono: zapytanie jednego wlasciciela nie zwraca punktow innego, nawet gdy sa najlepszym dopasowaniem semantycznym. Istniejace punkty bez pola sa pomijane (fail-closed) i wymagaja reindeksacji.

Problem: Qdrant zwraca globalne top-k z calej kolekcji, a dopiero potem `RetrievalService` filtruje wyniki przez PostgreSQL i sprawdza, czy punkty naleza do aktualnego uzytkownika.

Skutek: przy wielu uzytkownikach cudze dokumenty moga zajac pierwsze miejsca w top-k. Po odfiltrowaniu backend moze zwrocic malo wynikow albo brak wynikow, mimo ze dokumenty aktualnego uzytkownika zawieraja odpowiedz. To jest problem jakosci retrieval, a nie bezposredni wyciek danych, bo odpowiedzi sa filtrowane przed pokazaniem agentowi.

Miejsca w kodzie:

- `rag/retrieval.py` - `RetrievalService.search()` pobiera wyniki z providera bez filtra ownera.
- `rag/retrieval.py` - `search_results_from_response()` filtruje wyniki po stronie aplikacji.
- `rag/database/repositories.py` - `ownership_for_search()` sprawdza w PostgreSQL, ktore punkty sa dozwolone dla uzytkownika.

Kierunek naprawy:

- Dodac do payloadu punktow Qdranta `owner_user_id` lub stabilny `document_id`.
- Przekazywac filtr Qdrant w czasie wyszukiwania, zanim liczony jest top-k.
- Zachowac obecne filtrowanie po stronie DB jako druga linie obrony.

Kryteria ukonczenia:

- Nowe punkty w Qdrant zawieraja pole pozwalajace filtrowac po wlascicielu.
- Searcher przyjmuje kontekst uzytkownika albo osobny obiekt zapytania zawierajacy filtr.
- Test pokazuje, ze przy cudzych wysoko ocenionych punktach user nadal dostaje swoje wyniki.

## 3. Brak Oversamplingu Przed Filtrowaniem Wynikow

**Status (2026-06-28): ROZWIAZANE docelowym kierunkiem.** Wprowadzono prefiltracje po `owner_user_id` w Qdrant (patrz #2), wiec kandydaci z prefetch sa juz ograniczeni do wlasciciela i post-filtr DB praktycznie nic nie ucina. Oversampling przestaje byc potrzebny dla problemu izolacji wielu najemcow. Opis ponizej zachowany jako kontekst historyczny.

Problem: `RetrievalService` prosi providera o dokladnie `limit` wynikow, a potem usuwa wyniki nienalezace do uzytkownika. Nie ma zapasu kandydatow przed filtrowaniem.

Skutek: nawet jesli filtrowanie po stronie DB pozostaje tymczasowo potrzebne, maly limit moze powodowac puste odpowiedzi. Przy limicie 5 i globalnej kolekcji wystarczy, ze pierwsze 5 punktow nalezy do innych dokumentow.

Miejsca w kodzie:

- `rag/retrieval.py` - `response = self._search_provider.search(query, k=limit)`.
- `rag/retrieval.py` - `_database_search_results()` ucina dopiero po zebraniu dozwolonych wynikow.

Kierunek naprawy:

- Do czasu wprowadzenia prefiltra w Qdrant pobierac wiecej kandydatow niz finalny limit, np. `limit * 5` z gornym ograniczeniem.
- Finalny limit stosowac dopiero po filtrowaniu ownership.
- Docelowo zastapic to prefiltracja po `owner_user_id` w Qdrant.

Kryteria ukonczenia:

- Test pokrywa przypadek: pierwsze wyniki z Qdranta sa cudze, ale dalsze punkty naleza do usera.
- API zwraca finalnie maksymalnie zadany limit.
- Oversampling ma limit bezpieczenstwa, zeby nie powodowac kosztownych zapytan.

## 4. Brak Rerankingu I Progu Jakosci Wynikow

Problem: system przekazuje agentowi wyniki bez dodatkowego rerankingu i bez minimalnego progu trafnosci. Agent dostaje top-k z Qdranta po score, ale nie ma osobnej walidacji, czy fragment faktycznie odpowiada na pytanie.

Skutek: odpowiedzi moga byc oparte na fragmentach tematycznie podobnych, ale nieodpowiadajacych na konkretne pytanie. W domenie medycznej to podnosi ryzyko nieprecyzyjnych interpretacji.

Miejsca w kodzie:

- `rag/searcher.py` - brak rerankingu po Qdrant query.
- `rag/retrieval.py` - `SearchResult` przenosi score, ale nie stosuje thresholdow.
- `tools/rag_search.py` - narzedzie agenta zapisuje wszystkie wyniki zwrocone przez retriever.

Kierunek naprawy:

- Dodac etap rerankingu kandydatow albo przynajmniej progi score dla wynikow.
- Rozwazyc osobny `RetrievalRanker` jako port aplikacyjny, aby nie mieszac rerankingu z Qdrant adapterem.
- Dla niskiej pewnosci zwracac agentowi mniej zrodel albo jawny stan insufficient retrieval.

Kryteria ukonczenia:

- Jest test dla wyniku ponizej progu, ktory nie trafia do zrodel agenta.
- Jest test dla wielu kandydatow, ktory potwierdza stabilna kolejnosc po rerankingu.
- Trace agenta pokazuje liczbe kandydatow przed i po rerankingu.

## 5. Brak Zestawu Ewaluacyjnego Retrieval

Problem: testy sprawdzaja kontrakty techniczne, ale nie mierza jakosci wyszukiwania. Nie ma zestawu pytan z oczekiwanymi dokumentami, chunkami lub minimalnym recall@k.

Skutek: zmiany w chunkingu, embeddingach, Qdrant, filtrach albo promptach moga pogorszyc retrieval bez widocznego bledu w testach. System moze nadal przechodzic testy, ale odpowiadac gorzej.

Miejsca w kodzie:

- `tests/test_searcher.py` - sprawdza konstrukcje query, nie jakosc wynikow.
- `tests/test_indexer.py` - sprawdza indeksowanie i payload, nie skutecznosc wyszukiwania.
- `rag/measurement/performance.py` - mierzy wydajnosc i overlap, ale nie jest produkcyjnym regression setem dla demo medycznego.

Kierunek naprawy:

- Dodac maly zestaw ewaluacyjny oparty o syntetyczne dokumenty demo.
- Dla kazdego pytania zapisac oczekiwany dokument, ewentualnie oczekiwany fragment lub `content_hash`.
- Uruchamiac ewaluacje lokalnie bez live OpenRouter, jesli to mozliwe, przez deterministyczny embedding testowy albo lokalny fast embedding.

Kryteria ukonczenia:

- An evaluation file exists with expected sources for each question.
- Test lub komenda raportuje recall@k dla kluczowych scenariuszy demo.
- Regression set covers ACL, psoriasis/phototherapy, and GLP-1 remote monitoring questions.

## 6. Chunking Jest Zbyt Prosty Dla Dokumentow Medycznych

Problem: chunking uzywa stalego rozmiaru markdown chunkow i overlapa. Nie ma semantycznego modelu sekcji, dat, tabel laboratoryjnych, naglowkow, jednostek ani relacji miedzy wynikiem i zakresem referencyjnym.

Skutek: wazne informacje moga zostac rozdzielone albo wymieszane. Dla badan laboratoryjnych ryzykiem jest zgubienie kontekstu: nazwa badania, wynik, jednostka, zakres i data powinny trafic do tego samego fragmentu.

Miejsca w kodzie:

- `rag/chunking/process_text.py` - `MARKDOWN_CHUNK_SIZE = 800`, `MARKDOWN_CHUNK_OVERLAP = 120`.
- `rag/indexer.py` - `chunks_from_text()` opiera sie na `ProcessText.markdown_chunking()`.
- `tests/test_process_text.py` - testuje tylko proste zachowanie tabel i dlugiego tekstu.

Kierunek naprawy:

- Dodac chunker, ktory zachowuje cale tabele i sekcje kliniczne, gdy sa ponizej limitu.
- Przenosic naglowki sekcji do metadanych chunkow.
- Dodac specjalne testy dla dlugich tabel, wynikow z jednostkami, datami i zakresami referencyjnymi.

Kryteria ukonczenia:

- Test pokazuje, ze rekord laboratoryjny nie traci jednostki, zakresu ani daty.
- Chunk metadata includes at least source, character range, and optional section headers.
- Dashboard nadal pokazuje poprawne preview chunkow.

## 7. `rag/indexer.py` Has Too Many Responsibilities

Problem: jeden modul odpowiada za chunking, embedding, walidacje embeddingow, tworzenie kolekcji Qdrant, budowanie punktow, payload preview, idempotencje po `content_hash`, synchronizacje chunkow do PostgreSQL i progress events.

Skutek: kazda zmiana w indeksowaniu ma duzy koszt poznawczy i ryzyko regresji. Trudniej testowac osobno logike aplikacyjna, adapter Qdrant i synchronizacje bazy.

Miejsca w kodzie:

- `rag/indexer.py` - centralny plik indeksowania.
- `rag/full_process.py` - wywoluje `index_text()` jako glowny indexer.
- `rag/markdown_indexing.py` - opakowuje indeksowanie markdownow, ale sama logika nadal siedzi w `rag/indexer.py`.

Kierunek naprawy:

- Rozdzielic modul na mniejsze role:
  - `ChunkDocumentUseCase` albo `MarkdownChunker`.
  - `EmbeddingService` lub port dla embeddingow.
  - `VectorIndexWriter` jako adapter Qdrant.
  - `IndexedChunkSynchronizer` dla PostgreSQL.
  - `IndexDocumentUseCase` jako cienki orkiestrator.

Kryteria ukonczenia:

- Kazda nowa klasa ma jedna odpowiedzialnosc i prosty kontrakt.
- Testy istnieja na poziomie chunkera, writera Qdrant i use case'u.
- Publiczny kontrakt `FullProcess` pozostaje kompatybilny z dashboardem.

## 8. Warstwy Architektoniczne Sa Wymieszane

Problem: kod RAG miesza logike aplikacyjna z infrastruktura. Moduly z obszaru `rag` importuja Qdrant, SQLAlchemy, repozytoria, LangChain splittery i Pydantic. To nie spelnia standardu z `AGENTS.md`, ktory wymaga oddzielenia domeny, aplikacji, infrastruktury i prezentacji.

Skutek: trudniej wymienic Qdrant, embedding provider, baze danych albo sposob chunkingu bez dotykania logiki procesu. Trudniej tez utrzymac testowalnosc i scisle typowanie.

Miejsca w kodzie:

- `rag/indexer.py` - importuje Qdrant, SQLAlchemy sessionmaker, repozytoria, modele Qdrant i chunking.
- `rag/chunking/process_text.py` - model chunkingu dziedziczy po Pydantic `BaseModel`.
- `rag/retrieval.py` - use case retrieval zna SQLAlchemy `sessionmaker` i konkretny `DocumentRepository`.

Kierunek naprawy:

- Zdefiniowac porty aplikacyjne dla wyszukiwania, zapisu indeksu i repozytorium dokumentow.
- Przeniesc zaleznosci Qdrant/SQLAlchemy do adapterow infrastruktury.
- Zostawic w warstwie aplikacyjnej typy i protokoly niezalezne od frameworkow.

Kryteria ukonczenia:

- Core use case'y nie importuja Qdrant ani SQLAlchemy.
- Chunker nie wymaga Pydantic.
- Testy use case'ow uzywaja fake portow bez monkeypatchowania globalnych klas.

## 9. Brak Sensownego Trybu Retrieval Bez Kontekstu Bazy

Problem: `search_results_from_response()` zwraca pusta liste, jesli nie ma `owner_user_id` i `database_session_factory`. To jest bezpieczne dla dashboardu, ale czyni retriever malo uzytecznym poza zalogowanym przeplywem.

Skutek: CLI, narzedzia diagnostyczne albo przyszle testy RAG nie moga latwo uzyc tego samego retrievera bez pelnego kontekstu uzytkownika i bazy. Powstaje pokusa omijania `RetrievalService` i uzywania Qdranta bezposrednio.

Miejsca w kodzie:

- `rag/retrieval.py` - `search_results_from_response()` konczy `return []` bez ownership context.
- `dashboard/services/search_service.py` - dziedziczy po `RetrievalService` i zaklada kontekst dashboardu.

Kierunek naprawy:

- Nazwac obecny use case jednoznacznie jako `UserScopedRetrievalService`.
- Dla diagnostyki dodac osobny, jawny tryb administracyjny lub dev-only, ktory nie udaje sciezki produkcyjnej.
- Nie rozluzniac zabezpieczen dashboardu.

Kryteria ukonczenia:

- Nazwy klas i testow jasno rozrozniaja retrieval user-scoped od diagnostycznego.
- Brak kontekstu usera powoduje jawny blad w sciezce produkcyjnej, a nie ciche puste wyniki.
- Ewentualny tryb diagnostyczny jest oddzielony od endpointow uzytkownika.

## 10. Obsluga Bledow Infrastruktury Jest Zbyt Ogolna

Problem: wiele miejsc lapie szerokie `Exception` i zwraca `str(error)` do warstwy wyzszej. To pomaga w demo, ale nie daje stabilnej klasyfikacji bledow operacyjnych.

Skutek: UI i backend maja ograniczona mozliwosc rozroznienia problemow typu brak konfiguracji, timeout Qdrant, blad embedding providera, blad parsowania PDF albo blad bazy. Trudniej budowac retry, alerty i bezpieczne komunikaty dla uzytkownika.

Miejsca w kodzie:

- `dashboard/services/qdrant_index.py` - status, delete i preview lapia szerokie bledy.
- `dashboard/routes/search.py` - endpoint `/api/search` lapie `Exception`.
- `backend/use_cases.py` i `backend/chat_use_cases.py` - opakowuja wiele bledow jako `AgentExecutionError`.

Kierunek naprawy:

- Wprowadzic jawne typy bledow aplikacyjnych: `VectorStoreUnavailable`, `EmbeddingProviderUnavailable`, `DocumentParsingFailed`, `RetrievalFailed`.
- Logowac techniczny szczegol po stronie serwera, ale do UI zwracac kontrolowany komunikat.
- Zachowac szczegoly diagnostyczne w trace albo logach, nie w przypadkowych odpowiedziach JSON.

Kryteria ukonczenia:

- Endpointy zwracaja stabilne kody i kategorie bledow.
- Testy pokrywaja timeout Qdrant, brak konfiguracji i blad embeddingow.
- Uzytkownik widzi zrozumialy komunikat, a log zawiera szczegoly techniczne.

## 11. Konfiguracja Modelu Embeddingow I Kolekcji Moze Latwo Sie Rozjechac

Problem: kolekcja Qdrant jest walidowana pod katem wymiaru i typu wektora, ale zmiana modelu embeddingow nadal wymaga swiadomej migracji albo nowej kolekcji. W projekcie nie ma jasnego procesu reindeksacji po zmianie modelu.

Skutek: po zmianie modelu mozna dostac blad konfiguracji albo niejednoznaczny stan, w ktorym stare punkty i nowe oczekiwania nie pasuja do siebie. Dla demo to jest akceptowalne, ale dla dalszego rozwoju wymaga procesu operacyjnego.

Miejsca w kodzie:

- `rag/qdrant.py` - `_validate_collection_vectors()` sprawdza konfiguracje istniejacej kolekcji.
- `rag/indexer.py` - `_validate_collection_vectors()` powtarza podobna walidacje.
- `rag/settings.json` - zawiera wybrany model embeddingow i nazwy wektorow.

Kierunek naprawy:

- Dodac dokumentowany proces: zmiana modelu oznacza nowa kolekcje albo migracje i pelny reindex.
- Rozwazyc zapis `embedding_model` i `embedding_provider` w payloadzie lub metadanych dokumentu.
- Ujednolicic walidacje kolekcji w jednym adapterze.

Kryteria ukonczenia:

- README albo docs opisuje procedure zmiany modelu embeddingow.
- Test potwierdza czytelny blad przy niekompatybilnej kolekcji.
- Walidacja konfiguracji kolekcji nie jest zduplikowana w kilku miejscach.

## 12. Missing Answer Quality Observability

Problem: system zapisuje trace zdarzen agenta i sources, ale nie zapisuje metryk jakosci retrieval, liczby kandydatow przed filtrowaniem, liczby odrzuconych wynikow, uzytego trybu searcha ani powodow insufficient context.

Skutek: gdy agent odpowie slabo albo zwroci brak kontekstu, trudno szybko ustalic, czy zawinil parsing, chunking, indeks, filtr ownera, Searcher, prompt czy model.

Miejsca w kodzie:

- `tools/rag_search.py` - trace zapisuje query, limit i source count.
- `agents/graph.py` - trace zapisuje model calls, tool calls i synteze.
- `rag/retrieval.py` - brak jawnych metryk kandydatow przed i po filtrowaniu.

Kierunek naprawy:

- Dodac do trace retrieval: liczbe kandydatow z Qdranta, liczbe po filtrze ownera, tryb searcha, finalny limit.
- Zapisywac powod insufficient context, np. brak kandydatow, kandydaci odfiltrowani, niskie score, blad providera.
- Pokazac podstawowe informacje w panelu trace dashboardu.

Kryteria ukonczenia:

- Trace dla zapytania RAG pokazuje pelny lejek retrieval.
- Test agenta sprawdza, ze event RAG zawiera candidate count and source count.
- Debugowanie pustej odpowiedzi nie wymaga recznego odpytywania Qdranta.

## 13. Index Deletion Relies Mainly On `content_hash`

Problem: przy usuwaniu dokumentu system usuwa punkty Qdrant po `content_hash`. To dziala dla obecnego modelu, ale `content_hash` nie jest identyfikatorem wlasciciela ani dokumentu.

Skutek: jesli dwoch uzytkownikow ma identyczna tresc dokumentu, usuniecie po samym `content_hash` moze usunac punkty wspoldzielone logicznie przez wiecej niz jeden rekord dokumentu. Aktualna deduplikacja po `content_hash` moze tez utrudnic precyzyjne zarzadzanie dokumentami.

Miejsca w kodzie:

- `dashboard/services/document_storage.py` - delete uruchamia cleanup po `content_hash`.
- `dashboard/services/qdrant_index.py` - `delete_content_hash()` usuwa punkty filtrem po `content_hash`.
- `rag/indexer.py` - `_content_hash_exists()` pomija embedding, jesli content hash juz istnieje.

Kierunek naprawy:

- Dodac `document_id` i `owner_user_id` do payloadu Qdrant.
- Rozwazyc punkt per dokument+chunk, nawet dla identycznej tresci, albo jawna tabele referencji dla deduplikacji.
- Usuwac po `document_id` w sciezce dokumentu uzytkownika.

Kryteria ukonczenia:

- Test pokazuje dwa dokumenty o tym samym `content_hash` u roznych userow.
- Deleting one user's document does not remove another user's index entries.
- Cleanup Qdrant ma filtr po `document_id` albo po parze `owner_user_id` + `document_id`.

## 14. Typowanie I Kontrakty Sa Miejscami Zbyt Luzne

Problem: wiele granic uzywa `Any`, slownikow i obiektow Qdrant bez wlasnych typow aplikacyjnych. To utrudnia utrzymanie strict typing i zwieksza ryzyko regresji przy zmianach klienta Qdrant lub modeli odpowiedzi.

Skutek: bledy kontraktu moga wyjsc dopiero runtime. Trudniej tez zrozumiec, jakie pola sa wymagane w payloadzie punktu, wyniku searcha albo trace eventu.

Miejsca w kodzie:

- `rag/retrieval.py` - response i point sa typu `Any`.
- `rag/indexer.py` - wiele helperow wektorowych operuje na `Any`.
- `dashboard/services/qdrant_preview.py` i `dashboard/services/qdrant_index.py` - adaptery Qdrant zwracaja slowniki.

Kierunek naprawy:

- Wprowadzic dataclasses dla `VectorSearchCandidate`, `IndexedChunkPayload`, `RetrievalTrace`.
- Ograniczyc `Any` do adapterow infrastruktury.
- Dodac konwersje z typow Qdrant do typow aplikacyjnych na brzegu adaptera.

Kryteria ukonczenia:

- Use case retrieval nie operuje bezposrednio na obiektach Qdrant.
- Payload punktu ma jawny typ i test serializacji.
- `mypy --strict` ma mniej wyjatkow albo nie wymaga luznych typow w warstwie aplikacji.

## 15. Odpowiadanie Na Pytania Ogolnomedyczne Jest Ograniczone Przez Wymuszenie Zrodel

Problem: prompt dopuszcza ogolne informacje medyczne, ale runtime finalnie zwraca insufficient context, jesli nie ma zarejestrowanych zrodel. To jest dobre dla pytan o dokumenty uzytkownika, lecz ogranicza pytania ogolne.

Skutek: uzytkownik moze pytac o ogolne wyjasnienie medyczne, a system potraktuje brak zrodel z dokumentow jako brak mozliwosci odpowiedzi. To moze byc celowa decyzja produktowa, ale powinna byc jawnie nazwana.

Miejsca w kodzie:

- `agents/prompts/base.md` - rozroznia pytania o dokumenty i pytania ogolne.
- `agents/graph.py` - `_synthesize_answer()` wymaga obecnosci zrodel.
- `docs/agent-prompt-authoring-brief.md` - wskazuje, ze prompt-only moze nie wystarczyc do obslugi pytan ogolnych.

Kierunek naprawy:

- Podjac decyzje produktowa: system tylko document-grounded czy takze general medical assistant.
- Jesli tylko document-grounded, doprecyzowac prompt i UI.
- Jesli takze ogolnomedyczny, dodac osobna sciezke odpowiedzi bez cytowan z dokumentow, wyraznie oznaczona jako informacja ogolna.

Kryteria ukonczenia:

- Test rozroznia pytanie o dokumenty od pytania ogolnego.
- UI pokazuje, czy odpowiedz jest oparta na dokumentach, czy jest informacja ogolna.
- Pytania o dokumenty nadal wymagaja zrodel i cytowan.

## Proponowana Kolejnosc Prac

1. ~~Naprawic sciezke search: podlaczyc hybrid RRF i dodac testy.~~ ZROBIONE (#1).
2. ~~Dodac filtrowanie po uzytkowniku w Qdrant~~ ZROBIONE — prefiltracja po `owner_user_id` w Qdrant (#2, #3).
3. Przygotowac regression set retrieval dla dokumentow demo.
4. Rozbic `rag/indexer.py` na mniejsze komponenty.
5. Uporzadkowac architekture portow i adapterow.
6. Dodac obserwowalnosc retrieval i kategorie bledow.
7. Ulepszyc chunking medyczny i metadane chunkow.
8. Doprecyzowac polityke pytan ogolnomedycznych.
