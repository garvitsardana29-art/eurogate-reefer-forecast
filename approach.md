# Dokumentation / Vorgehensweise

## 1. Kurzfassung

In dieser Challenge soll die **kombinierte elektrische Last aller angeschlossenen Reefer-Container eines Terminals** stundenweise vorhergesagt werden. Pro Zielstunde müssen zwei Werte geliefert werden:

- `pred_power_kw`: meine Punktschätzung für die Last
- `pred_p90_kw`: eine vorsichtige obere Schätzung

Für mich war von Anfang an wichtig, die Aufgabe **sauber als 24-Stunden-Vorhersageproblem** zu behandeln. Das bedeutet: Für eine Zielstunde `t` darf ich nur Informationen verwenden, die zum Vorhersagezeitpunkt realistisch verfügbar wären. Genau darauf ist mein finales Modell ausgelegt. Es nutzt nur Merkmale, die **24 Stunden oder älter** sind, und wurde ausschließlich mit **Vor-Januar-Daten** ausgewählt und kalibriert. [1][2]

Das finale Ergebnis dieser Abgabe ist die Datei [predictions_strict_day_ahead_blend.csv](predictions_strict_day_ahead_blend.csv). Erzeugt wird sie durch das Skript [strict_day_ahead_blend_submission.py](strict_day_ahead_blend_submission.py).

## 2. Aufgabenverständnis und Zielsetzung

Die Challenge ist **kein allgemeines Terminal-Workload-Projekt**, sondern ein **Lastprognoseproblem für Reefer-Stromverbrauch**. Das Ziel ist also nicht, Schiffsanläufe, Yard-Bewegungen oder allgemeine Hafenauslastung vorherzusagen, sondern die Frage zu beantworten:

> In welchen Stunden des nächsten Tages ist mit hoher elektrischer Last durch angeschlossene Reefer-Container zu rechnen?

Die Bewertungslogik gewichtet drei Dinge: die mittlere Genauigkeit über alle Stunden, die Genauigkeit in Hochlaststunden und die Qualität des oberen Risiko-Schätzwerts `pred_p90_kw`. Der kombinierte Score lautet: [2]

`0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

Daraus ergeben sich für mich drei praktische Anforderungen:

1. Das Modell muss im Mittel stabil sein.
2. Es darf Peak-Stunden nicht verfehlen.
3. `pred_p90_kw` muss nützlich sein und darf nicht nur ein pauschaler Aufschlag sein.

## 3. Randbedingungen und Regelkonformität

Ich habe mich bewusst für eine Lösung entschieden, die sich gut gegenüber den Challenge-Regeln begründen lässt. Die wichtigsten Punkte waren: [1]

- Es werden nur die bereitgestellten Dateien verwendet.
- Die Modelllogik behandelt die Aufgabe als **24-Stunden-Ahead-Forecast**.
- Für eine Zielstunde werden nur Daten aus `t-24h` oder älter verwendet.
- Für die finale Modellauswahl wurden **keine sichtbaren Januar-2026-Zielzeilen** zur Parametertuning-Logik verwendet.

Gerade dieser letzte Punkt war mir wichtig, weil die Organisatoren den Code später mit einer erweiterten, privaten Zielliste erneut ausführen. Eine Lösung, die nur auf den sichtbaren Public-Block optimiert ist, wäre kurzfristig vielleicht attraktiv, aber methodisch nicht sauber genug für eine belastbare Abgabe. [1][2]

## 4. Datengrundlage

Die Projektbasis besteht aus drei Kernquellen:

### 4.1 Reefer-Daten

Die Datei `reefer_release.zip` enthält die operativen Reefer-Daten auf **Container-Ebene**. Aus diesen Rohdaten lassen sich unter anderem ableiten:

- tatsächliche Terminal-Last
- Anzahl aktiver Container
- Temperaturinformationen
- Setpoint-Verteilungen
- Stapel- und Hardware-Mix
- Kundenmischung

### 4.2 Wetterdaten

Die Datei `wetterdaten.zip` liefert Wetterinformationen, unter anderem für die Referenzpunkte:

- `Zentralgate`
- `VC Halle 3`

Diese Daten habe ich auf Stundenebene aggregiert und im Modell getestet. Für die finale regelkonforme Auswahl waren Wettermerkmale nicht der stärkste Treiber, aber sie waren für Analyse, Feature-Tests und das Verständnis des Pakets trotzdem nützlich.

### 4.3 Zielstunden

Die Datei `target_timestamps.csv` enthält genau die Stunden, für die eine Prognose abgegeben werden muss. Diese Liste definiert also den Vorhersagehorizont der Abgabe. [1]

## 5. Datenaufbereitung

### 5.1 Warum überhaupt eine Aggregation nötig war

Die größte strukturelle Herausforderung war für mich, dass die Rohdaten **containerbasiert** vorliegen, das Ziel aber **terminalbasiert** ist. Ich musste also zuerst aus vielen einzelnen Reefer-Zeilen pro Stunde **eine einzige Terminal-Stunde** machen.

Genau das übernimmt das Skript [step1_build_hourly_terminal_dataset.py](step1_build_hourly_terminal_dataset.py). Es erzeugt die Datei [hourly_terminal_dataset.csv](hourly_terminal_dataset.csv).

### 5.2 Inhalt des stündlichen Terminal-Datensatzes

Die aufbereitete Stundentabelle enthält unter anderem:

- gesamte Terminal-Last in kW
- Anzahl aktiver Reefer
- Durchschnittswerte für Ambient-, Setpoint-, Return- und Supply-Temperaturen
- Setpoint-Minimum, -Maximum und -Spannweite
- Containeranzahlen in den Gruppen `frozen`, `cold`, `chilled` und `warm`
- Stack-Tier-Zählungen
- Hardware-Mix
- Kundenkonzentration
- Kennzeichen für beobachtete Stunden (`is_observed_hour`)

Damit habe ich aus den Rohdaten eine Modellbasis gemacht, die sowohl zeitlich konsistent als auch fachlich deutlich aussagekräftiger ist als eine reine Last-Zeitreihe.

### 5.3 Wetter-Aggregation

Zusätzlich habe ich die Wetterdaten mit [step3_build_hourly_weather_dataset.py](step3_build_hourly_weather_dataset.py) zu [hourly_weather_dataset.csv](hourly_weather_dataset.csv) verdichtet. Dabei wurden insbesondere stündliche Temperatur-, Wind- und Richtungsinformationen zusammengeführt.

## 6. Modellidee

### 6.1 Warum ich mich nicht nur für ein einzelnes Modell entschieden habe

Im Verlauf der Experimente hat sich gezeigt:

- Ein reines Baseline-Modell ist stabil, aber nicht stark genug.
- Ein reines nichtlineares Modell kann Peaks gut erfassen, ist aber nicht in jeder Situation die beste Wahl.

Deshalb ist mein finales Modell ein **striktes Day-Ahead-Blend-Modell** aus zwei Komponenten:

- einem **XGBoost-Regressor**
- einem **Ridge-Regressor**

Die Idee dahinter ist einfach: Das lineare Modell stabilisiert die Vorhersage, während XGBoost nichtlineare Zusammenhänge und Interaktionen besser erfassen kann. Diese Kombination war in meiner sauberen Vor-Januar-Validierung die beste regelkonforme Variante.

### 6.2 Finale Implementierung

Das finale Skript ist:

- [strict_day_ahead_blend_submission.py](strict_day_ahead_blend_submission.py)

Die finale Vorhersagedatei ist:

- [predictions_strict_day_ahead_blend.csv](predictions_strict_day_ahead_blend.csv)

## 7. Verwendete Features

Für das finale Modell habe ich nur Merkmale verwendet, die für Stunde `t` bereits **24 Stunden vorher oder noch früher** bekannt sind.

### 7.1 Last-Historie

Die Last-Historie war der wichtigste Informationsblock. Verwendet wurden unter anderem:

- `lag_load_24`
- `lag_load_48`
- `lag_load_72`
- `lag_load_168`
- Mittelwert, Standardabweichung und Maximum über das vorherige Issue-Fenster
- Vergleich zwischen Tages- und Wochenhistorie

### 7.2 Kalendermerkmale

Zusätzlich wurden zyklische Zeitmerkmale eingebaut:

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `is_weekend`

Damit kann das Modell Tagesmuster und Wochenstrukturen abbilden, ohne harte Brüche zwischen einzelnen Stunden oder Wochentagen zu erzeugen.

### 7.3 Operative Terminalmerkmale

Aus der stündlichen Tabelle wurden nur lagged Versionen verwendet, zum Beispiel:

- `active_container_count`
- `avg_temperature_ambient`
- `avg_temperature_setpoint`
- `count_setpoint_frozen`
- `count_setpoint_warm`
- `count_stack_tier_3`
- `top_tier_extreme_pressure`
- `customer_hhi_top5`
- `count_hw_ML3`
- `mixed_setpoint_pressure`

Für diese Merkmale wurden insbesondere folgende Ableitungen verwendet:

- `lag24`
- `lag168`
- `delta_day_week`

### 7.4 Interaktionen

Ich habe außerdem einige fachlich plausible Interaktionen ergänzt, unter anderem:

- Ambient-Temperatur × aktive Containerzahl
- Stack Tier 3 × ML3-Hardware
- Warm-gegen-Frozen-Differenz

Gerade solche Interaktionen helfen dabei, physische Zusammenhänge besser abzubilden, zum Beispiel dann, wenn nicht nur die Anzahl der Container, sondern auch deren Zusammensetzung entscheidend ist.

## 8. Validierung und Modellauswahl

Für die Modellauswahl habe ich **Forward-Chaining / Time-Based Validation** verwendet. Dabei wurden die Gewichte und die `p90`-Kalibrierung ausschließlich auf einem späten Vor-Januar-Fenster bestimmt:

- Validierungsstart: `2025-12-11`
- Zielstart: `2026-01-01`

Das heißt konkret:

- Training für die Auswahl lief nur auf Daten **vor dem 11.12.2025**
- Gewichte und `p90`-Parameter wurden auf der späten Dezember-Periode bestimmt
- der sichtbare Januar-Block blieb für die eigentliche Modellauswahl unberührt

Die im finalen Skript gewählten Gewichte sind:

- XGBoost: `0.40`
- Ridge: `0.60`
- Naive-24h-Komponente: `0.00`

Für `pred_p90_kw` wurde ein Dezember-basierter Uplift verwendet:

- Basis-Uplift: `187.538 kW`
- Skalierung: `1.00`
- Shift: `0.0 kW`

## 9. Ergebnisse

Auf dem lokal sichtbaren Public-Januar-Fenster ergab sich für das finale strikte Modell:

- `MAE all = 61.013 kW`
- `MAE peak = 26.726 kW`
- `Pinball p90 = 23.059`
- `P90 coverage = 1.000`
- kombinierter Score = `43.136`

Diese Werte stammen aus der lokalen Auswertung auf dem sichtbaren Public-Block. Der offizielle Endscore der Organisatoren kann davon abweichen, weil dort zusätzlich ein privater, nicht freigegebener Teil berücksichtigt wird. [2]

## 10. Warum ich dieses Modell als finale Abgabe gewählt habe

Ich habe mich bewusst **nicht** für eine rein leaderboard-orientierte Lösung entschieden, sondern für die Variante, die sich fachlich und methodisch am besten vertreten lässt.

Die Gründe dafür sind:

1. Das Modell ist sauber als 24h-Ahead-Prognose formuliert.
2. Es nutzt keine Januar-Tuning-Tricks.
3. Es ist stabil reproduzierbar.
4. Es ist für einen Organisator-Rerun nachvollziehbar.
5. Es verbindet lineare Stabilität mit nichtlinearer Flexibilität.

Für mich ist das die beste Kombination aus Performance, Regelkonformität und technischer Glaubwürdigkeit.

## 11. Reproduzierbarkeit

Die finale Abgabe lässt sich aus dem Projektordner mit zwei Schritten reproduzieren:

```bash
python3 step1_build_hourly_terminal_dataset.py
.venv_check/bin/python strict_day_ahead_blend_submission.py
```

Optional kann auch die Wettertabelle neu erzeugt werden:

```bash
python3 step3_build_hourly_weather_dataset.py
```

Wenn zusätzlich das Demo inklusive Upload-Funktion gestartet werden soll, kann das lokale Backend so ausgeführt werden:

```bash
.venv_check/bin/python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

## 12. Technische Umsetzung im Repository

Die wichtigsten Dateien im finalen Paket sind:

- [step1_build_hourly_terminal_dataset.py](step1_build_hourly_terminal_dataset.py)
- [step3_build_hourly_weather_dataset.py](step3_build_hourly_weather_dataset.py)
- [strict_day_ahead_blend_submission.py](strict_day_ahead_blend_submission.py)
- [predictions_strict_day_ahead_blend.csv](predictions_strict_day_ahead_blend.csv)
- [backend_api.py](backend_api.py)
- [demo/index.html](demo/index.html)
- [demo/rerun.html](demo/rerun.html)

Die Website dient vor allem der Präsentation. Zusätzlich gibt es mit dem Rerun-Lab eine technische Zusatzfunktion, über die ein neues Paket hochgeladen und die Vorhersage erneut erzeugt werden kann.

## 13. Grenzen der aktuellen Lösung

Trotz des guten finalen Ergebnisses gibt es natürlich Grenzen:

- Exakte physische Nachbarschaften einzelner Container sind im Datensatz nicht vollständig sichtbar.
- Wetter war im finalen rules-safe Modell nicht der dominierende Treiber.
- Das Modell sagt Terminal-Last auf Gesamtterminebene voraus, nicht je Container oder je Subzone.
- Der private Organisatoren-Block bleibt naturgemäß unbekannt.

Gerade deshalb war mir eine robuste und sauber begründete Methodik wichtiger als ein rein auf den sichtbaren Public-Block optimierter Ansatz.

## 14. Einsatz von KI-Werkzeugen

KI-Werkzeuge wurden in diesem Projekt unterstützend verwendet, insbesondere für:

- Ideenfindung bei Features
- Strukturierung der Pipeline
- Code-Entwurf und Überarbeitung
- Analyse von Experimenten
- Erstellung der Dokumentation

Die finale Auswahl des Modells basiert aber nicht auf Textgeneratoren, sondern auf tatsächlich ausgeführten Zeitreihen-Experimenten, Validierungen und reproduzierbaren Skripten.

## 15. Persönliches Fazit

Rückblickend war die wichtigste Entscheidung, die Aufgabe **nicht als beliebiges ML-Problem**, sondern als **sauberes Day-Ahead-Forecasting-Problem** zu behandeln. Genau dadurch wurde die Lösung robuster.

Ich habe am Ende bewusst eine Variante gewählt, die ich fachlich vertreten kann: regelkonform, reproduzierbar, nachvollziehbar und mit einem deutlich besseren Ergebnis als die frühen Baselines.

## 16. Quellenverzeichnis

[1] EUROGATE / Challenge-Unterlagen: [REEFER_PEAK_LOAD_CHALLENGE.md](REEFER_PEAK_LOAD_CHALLENGE.md), Zugriff im Projekt am 10.04.2026.

[2] EUROGATE / Bewertungslogik: [EVALUATION_AND_WINNER_SELECTION.md](EVALUATION_AND_WINNER_SELECTION.md), Zugriff im Projekt am 10.04.2026.

[3] Primäre Datengrundlage: `reefer_release.zip`, `wetterdaten.zip`, `target_timestamps.csv`, bereitgestellt im Teilnehmerpaket, Zugriff im Projekt am 10.04.2026.

[4] Eigene Implementierung: [step1_build_hourly_terminal_dataset.py](step1_build_hourly_terminal_dataset.py), [step3_build_hourly_weather_dataset.py](step3_build_hourly_weather_dataset.py), [strict_day_ahead_blend_submission.py](strict_day_ahead_blend_submission.py), [backend_api.py](backend_api.py), Zugriff im Projekt am 10.04.2026.

[5] scikit-learn, Dokumentation zu Ridge Regression: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), Zugriff am 10.04.2026.

[6] XGBoost, offizielle Python API: [Python API Reference](https://xgboost.readthedocs.io/en/release_1.3.0/python/python_api.html), Zugriff am 10.04.2026.

[7] FastAPI, offizielle Dokumentation zu Datei-Uploads: [Request Files](https://fastapi.tiangolo.com/pt/tutorial/request_files/), Zugriff am 10.04.2026.

[8] FastAPI, offizielle Dokumentation zu statischen Dateien: [StaticFiles](https://fastapi.tiangolo.com/es/reference/staticfiles/), Zugriff am 10.04.2026.
