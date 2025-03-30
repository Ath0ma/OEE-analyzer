# 🛠️ OEE Analyzer – Streamlit App zur Maschinenzustandsanalyse

Diese Streamlit-App analysiert Produktionsvideos von Drehmaschinen, erkennt automatisch Maschinenzustände und berechnet Kennzahlen zur Verfügbarkeit (OEE). Die Objekterkennung erfolgt mit einem YOLOv8-Modell. Anschließend werden erkannte Objekte regelbasiert in Zustände (Produktion, geplante und ungeplante Stillstände) klassifiziert und visuell ausgewertet.

---

## 🚀 Funktionen

- Video-Upload und automatische Analyse
- Objekterkennung mit YOLOv8 (`best.pt`)
- Regelbasierte Zustandsklassifikation
- Interaktive Tabellen und Diagramme (Plotly)
- Statistische Auswertung (Verfügbarkeitsanteile)

---

## 📦 Installation

1. Python 3.10+ installieren
2. Projektdateien klonen oder herunterladen
3. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

4. App starten:

```bash
streamlit run OEE_analyzer.py
```

---

## 📂 Projektstruktur

```
├── OEE_analyzer.py        # Streamlit-App
├── requirements.txt       # Abhängigkeiten
├── README.md              # Diese Anleitung
└── model/
    └── best.pt            # YOLO-Modell
```

---

## ⚙️ Technische Details

- **Modell:** YOLOv8, trainiert zur Erkennung von:
  - Drehmaschine (drehend & still)
  - Reinigungsstange
  - Zange

- **Frame-Sampling:** Es wird nur **jeder 5. Frame analysiert** (`frame_skip = 5`), um die Performance zu verbessern.

- **Zeitberechnung:** Die Zeit jedes erkannten Frames wird mit der Video-FPS synchronisiert. Die Zustandsdauer ergibt sich aus der Differenz zwischen aufeinanderfolgenden Zustandswechseln.

- **Klassifikation:**
  - `"Drehmaschine_drehend"` → Produktion
  - `"Drehmaschine_still"` + `"Reinigungsstange"` → ungeplante Downtime
  - `"Drehmaschine_still"` + `"Zange"` → geplante Downtime

---

## 📊 Ausgabe und Interpretation

Nach der Analyse werden folgende Ergebnisse angezeigt:

- Tabelle aller erkannten Objekte mit Zeitstempel
- Zeitstrahl mit Zustandsblöcken (Produktion, geplante & ungeplante Downtime)
- Balken- & Liniendiagramme zur Visualisierung der Verfügbarkeit
- Prozentuale Anteile der Zustände

---

## 📌 Hinweise

- Das YOLO-Modell wurde spezifisch auf die Maschinenteile einer bestimmten Zerspanungsmaschine trainiert.
- Bei anderen Maschinen oder Umgebungen kann die Erkennungsgenauigkeit abweichen.
- Für andere Anwendungsfälle ist ein **Retraining des Modells mit eigenen Bildern möglich** (z. B. mit `ultralytics`).
- Die Videoanalyse ist auf **Nachbearbeitung** ausgelegt, nicht auf Echtzeit.

---

## 👩‍💻 Autorin

Anja Thoma  
Friedrich-Alexander-Universität Erlangen-Nürnberg  
Seminararbeit WS 2024/25
