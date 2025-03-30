import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

# Laden des YOLO-Modells
MODEL_PATH = "model/best.pt" #Modellpfad
model = YOLO(MODEL_PATH)  # Modell auf GPU verschieben (falls verfuegbar)

# Streamlit App UI
st.title("OEE Analyzer")

frame_skip = 5 #Nummer, der wie vielte Frame geskipped wird

# Video-Upload
uploaded_file = st.file_uploader("Lade ein Video hoch", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    running_time = 0
    object_log = []

    stframe = st.empty()
    progress_bar = st.progress(0)
    table_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        results = model(frame, conf=0.55)
        detections = results[0].boxes.data
        detected_classes = [model.names[int(det[5])] for det in detections]
        current_time = frame_count / frame_rate

        # Laufzeit zählen
        if "Drehmaschine_drehend" in detected_classes:
            running_time += frame_skip / frame_rate

        # Log-Eintrag
        object_log.append((current_time, detected_classes))

        # Bounding Boxes zeichnen
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det.tolist()
            class_name = model.names[int(class_id)]
            color = (0, 255, 0) if class_name == "Drehmaschine_drehend" else (0, 0, 255) #Drehmaschine_drehend hat eine gruene Bounding Box, alle anderen rot
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Frame anzeigen
        stframe.image(frame, channels='BGR', use_container_width=True)

        # Fortschritt anzeigen
        progress = int((frame_count / total_video_frames) * 100)
        progress_bar.progress(progress)

        # Live-Log als Tabelle anzeigen
        df_log = pd.DataFrame(object_log, columns=["Zeit (s)", "Erkannte Objekte"])
        table_placeholder.dataframe(df_log.tail(1)) #die Zahl gibt an wie viele Zeilen man sieht

    cap.release()
    

    # Gesamtzeit des Videos korrekt berechnen
    total_video_time = total_video_frames / frame_rate

    df_timeline = pd.DataFrame(object_log, columns=["Zeit (s)", "Erkannte Objekte"])
    
    df_zustand = []
    letzter_zustand = None
    startzeit = None
    downtime_start = None
    reinigungsstange_gesehen = False
    zange_zeitpunkt = None  # Zeitpunkt, ab dem Zange erkannt wird

    # Durch die Logdaten iterieren, um die Zustaende korrekt zu setzen
    for i, row in df_timeline.iterrows():
        erkannte_objekte = row["Erkannte Objekte"]
        aktuelle_zeit = row["Zeit (s)"]

        # Bestimme den aktuellen Zustand
        if "Drehmaschine_drehend" in erkannte_objekte:
            aktueller_zustand = "Produktion"
        elif "Drehmaschine_still" in erkannte_objekte:
            aktueller_zustand = "Downtime"
        else:
            continue  # Überspringe, wenn kein Zustand erkannt wird

        # Anfangszustand setzen, falls noch nicht gesetzt
        if letzter_zustand is None:
            letzter_zustand = aktueller_zustand
            startzeit = aktuelle_zeit
            if aktueller_zustand == "Downtime":
                downtime_start = aktuelle_zeit
            continue

        # Zustandswechsel prüfen
        if aktueller_zustand != letzter_zustand:
            if letzter_zustand == "Produktion":
                # Produktion endet, Downtime beginnt
                df_zustand.append({
                    "Startzeit (s)": startzeit,
                    "Endzeit (s)": aktuelle_zeit,
                    "Dauer (s)": aktuelle_zeit - startzeit,
                    "Zustand": "Produktion"
                })
                startzeit = aktuelle_zeit
                downtime_start = aktuelle_zeit
                reinigungsstange_gesehen = False
                zange_zeitpunkt = None
                
            elif letzter_zustand == "Downtime":
                # Downtime endet, Produktion beginnt
                if reinigungsstange_gesehen and zange_zeitpunkt is None:
                    # Nur Ungeplante Downtime
                    df_zustand.append({
                        "Startzeit (s)": startzeit,
                        "Endzeit (s)": aktuelle_zeit,
                        "Dauer (s)": aktuelle_zeit - startzeit,
                        "Zustand": "Ungeplante Downtime"
                    })
                elif reinigungsstange_gesehen and zange_zeitpunkt is not None:
                    # Ungeplante Downtime bis Zange, dann Geplante Downtime
                    df_zustand.append({
                        "Startzeit (s)": startzeit,
                        "Endzeit (s)": zange_zeitpunkt,
                        "Dauer (s)": zange_zeitpunkt - startzeit,
                        "Zustand": "Ungeplante Downtime"
                    })
                    df_zustand.append({
                        "Startzeit (s)": zange_zeitpunkt,
                        "Endzeit (s)": aktuelle_zeit,
                        "Dauer (s)": aktuelle_zeit - zange_zeitpunkt,
                        "Zustand": "Geplante Downtime"
                    })
                else:
                    # Nur Geplante Downtime
                    df_zustand.append({
                        "Startzeit (s)": startzeit,
                        "Endzeit (s)": aktuelle_zeit,
                        "Dauer (s)": aktuelle_zeit - startzeit,
                        "Zustand": "Geplante Downtime"
                    })
                startzeit = aktuelle_zeit
                reinigungsstange_gesehen = False
                zange_zeitpunkt = None
                downtime_start = None

            letzter_zustand = aktueller_zustand

        # Während Downtime: Flags setzen
        if aktueller_zustand == "Downtime":
            if "Reinigungsstange" in erkannte_objekte and zange_zeitpunkt is None:
                reinigungsstange_gesehen = True
            if "Zange" in erkannte_objekte and zange_zeitpunkt is None:
                zange_zeitpunkt = aktuelle_zeit

    # Letzte Phase abschließen
    if letzter_zustand == "Produktion" and startzeit is not None:
        df_zustand.append({
            "Startzeit (s)": startzeit,
            "Endzeit (s)": total_video_time,
            "Dauer (s)": total_video_time - startzeit,
            "Zustand": "Produktion"
        })
    elif letzter_zustand == "Downtime" and startzeit is not None:
        if reinigungsstange_gesehen and zange_zeitpunkt is None:
            df_zustand.append({
                "Startzeit (s)": startzeit,
                "Endzeit (s)": total_video_time,
                "Dauer (s)": total_video_time - startzeit,
                "Zustand": "Ungeplante Downtime"
            })
        elif reinigungsstange_gesehen and zange_zeitpunkt is not None:
            df_zustand.append({
                "Startzeit (s)": startzeit,
                "Endzeit (s)": zange_zeitpunkt,
                "Dauer (s)": zange_zeitpunkt - startzeit,
                "Zustand": "Ungeplante Downtime"
            })
            df_zustand.append({
                "Startzeit (s)": zange_zeitpunkt,
                "Endzeit (s)": total_video_time,
                "Dauer (s)": total_video_time - zange_zeitpunkt,
                "Zustand": "Geplante Downtime"
            })
        else:
            df_zustand.append({
                "Startzeit (s)": startzeit,
                "Endzeit (s)": total_video_time,
                "Dauer (s)": total_video_time - startzeit,
                "Zustand": "Geplante Downtime"
            })

    # Falls keine Zustände erkannt wurden (nur Produktion)
    if not df_zustand:
        st.write("Keine Downtime erkannt – nur Produktion vorhanden!")
        df_zustand.append({
            "Startzeit (s)": 0,
            "Endzeit (s)": total_video_time,
            "Dauer (s)": total_video_time,
            "Zustand": "Produktion"
        })

    # In DataFrame umwandeln
    df_zustand = pd.DataFrame(df_zustand)

    tab1, tab2 = st.tabs(["Tabellen", "Ergebnisse und Statistiken"])

    with tab1:
        st.subheader("Tabelle der erkannten Objekte")
        df_log2 = pd.DataFrame(object_log, columns=["Startzeit (s)", "Erkannte Objekte"])
        st.dataframe(df_log2)

        # Erstelle DataFrame
        st.subheader("Zustandstabelle")
        st.dataframe(df_zustand)

        
    with tab2:

        # Berechnung der Dauer für jeden Zustand
        produktion_dauer = df_zustand[df_zustand["Zustand"] == "Produktion"]["Dauer (s)"].sum()
        geplante_downtime_dauer = df_zustand[df_zustand["Zustand"] == "Geplante Downtime"]["Dauer (s)"].sum()
        ungeplante_downtime_dauer = df_zustand[df_zustand["Zustand"] == "Ungeplante Downtime"]["Dauer (s)"].sum()
        total_downtime = geplante_downtime_dauer + ungeplante_downtime_dauer

        total_time = produktion_dauer + geplante_downtime_dauer + ungeplante_downtime_dauer

        production_percentage = (produktion_dauer / total_time) * 100
        planned_downtime_percentage = (geplante_downtime_dauer / total_time) * 100
        unplanned_downtime_percentage = (ungeplante_downtime_dauer / total_time) * 100
        total_downtime_percentage = planned_downtime_percentage + unplanned_downtime_percentage

        # 1. Balkendiagramm aktualisieren
        fig = go.Figure()

        # Gesamtzeit
        fig.add_trace(go.Bar(
            x=["Gesamtzeit"], 
            y=[total_time],
            name="Gesamtzeit",
            marker_color="gray"))

        # Produktion
        fig.add_trace(go.Bar(
            x=["Produktion"],
            y=[produktion_dauer],
            name="Produktion",
            marker_color="green"))

        # Geplante Downtime
        fig.add_trace(go.Bar(
            x=["Geplante Downtime"],
            y=[geplante_downtime_dauer],
            base=[produktion_dauer],
            name="Geplante Downtime",
            marker_color="blue"))

        # Ungeplante Downtime
        fig.add_trace(go.Bar(
            x=["Ungeplante Downtime"],
            y=[ungeplante_downtime_dauer],
            base=[produktion_dauer + geplante_downtime_dauer],
            name="Ungeplante Downtime",
            marker_color="red"))

        # Layout
        fig.update_layout(
            title="Produktionszeit vs. Downtime",
            yaxis_title="Dauer (s)",
            barmode="stack",
            legend_title="Kategorie")

        st.plotly_chart(fig)

        #Angabe Zahlen zur Verfuegbarkeit
        st.write(f"**Produktionszeit:** {running_time:.2f} Sekunden ({production_percentage:.2f}%)")
        st.write(f"**Gesamte Downtime:** {total_downtime:.2f} Sekunden ({total_downtime_percentage:.2f}%)")
        st.write(f"     - Geplante Downtime: {geplante_downtime_dauer:.2f} Sekunden ({planned_downtime_percentage:.2f}%)")
        st.write(f"     - Ungeplante Downtime: {ungeplante_downtime_dauer:.2f} Sekunden ({unplanned_downtime_percentage:.2f}%)")

       
        # 2. Linien-Diagramm (Produktion vs. Downtime)
        df_line = pd.DataFrame(object_log, columns=["Startzeit (s)", "Erkannte Objekte"])
        df_line["Produktion"] = df_line["Erkannte Objekte"].apply(lambda x: 1 if "Drehmaschine_drehend" in x else 0)
        fig_line = px.line(df_line, x="Startzeit (s)", 
                           y="Produktion", 
                           title="Produktion vs. Downtime")
        st.plotly_chart(fig_line)

        # 3. Zeitverlauf-Diagramm (Produktion, Geplante Downtime, Ungeplante Downtime) als gefüllte Blöcke
        fig_timeline = go.Figure()

        # Farben definieren
        zustand_farben = {
            "Produktion": "green",
            "Geplante Downtime": "blue",
            "Ungeplante Downtime": "red"
        }

        # Zustände als Blöcke im Diagramm
        for zustand, farbe in zustand_farben.items():
            df_subset = df_zustand[df_zustand["Zustand"] == zustand]

            for _, row in df_subset.iterrows():
                fig_timeline.add_trace(go.Scatter(
                    x=[row["Startzeit (s)"], row["Endzeit (s)"]],
                    y=[1, 1],  # 1 für aktiv
                    fill="tozeroy",
                    mode="lines",
                    name=zustand,
                    line=dict(color=farbe, width=4)
                ))
                

        fig_timeline.update_layout(
            title="Zeitstrahl der Produktion und Downtime",
            xaxis_title="Startzeit (s)",
            template="plotly_white")
        
        st.plotly_chart(fig_timeline)

        anzahl_produktion = df_zustand[df_zustand["Zustand"] == "Produktion"].shape[0]
        anzahl_geplante_downtime = df_zustand[df_zustand["Zustand"] == "Geplante Downtime"].shape[0]
        anzahl_ungeplante_downtime = df_zustand[df_zustand["Zustand"] == "Ungeplante Downtime"].shape[0]

        st.write("Anzahl der Produktionsphasen:", anzahl_produktion)
        st.write("Anzahl der geplanten Downtimephasen:", anzahl_geplante_downtime)
        st.write("Anzahl der ungeplanten Downtimephasen:", anzahl_ungeplante_downtime)
