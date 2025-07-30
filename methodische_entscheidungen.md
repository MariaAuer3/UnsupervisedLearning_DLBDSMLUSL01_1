# Methodische Entscheidungen - Wissenschaftliche Trend-Analyse

## 1. Datenexploration und -verständnis

### 1.1 Deskriptive Statistiken
**Entscheidung**: Umfassende explorative Datenanalyse vor der Modellierung
**Begründung**: 
- Verständnis der Datenstruktur und -qualität
- Identifikation von Mustern und Ausreißern
- Grundlage für fundierte Feature-Engineering-Entscheidungen

### 1.2 Visualisierung der Datenverteilung
**Entscheidung**: Verschiedene Visualisierungen (Histogramme, Boxplots, Zeitreihen)
**Begründung**:
- Intuitive Erfassung der Datencharakteristika
- Identifikation von Trends und Mustern
- Kommunikation der Ergebnisse an Stakeholder

## 2. Textvorverarbeitung

### 2.1 Textreinigung
**Entscheidung**: Entfernung von Sonderzeichen, Normalisierung auf Kleinbuchstaben
**Begründung**:
- Reduktion von Rauschen in den Daten
- Standardisierung für konsistente Verarbeitung
- Verbesserung der Feature-Qualität

### 2.2 Tokenisierung und Lemmatisierung
**Entscheidung**: Verwendung von RegexpTokenizer statt NLTK für robuste Tokenisierung
**Begründung**:
- Vermeidung von NLTK-Abhängigkeiten und Download-Problemen
- RegexpTokenizer ist portabel und funktioniert ohne externe Ressourcen
- Wissenschaftliche Texte werden mit regulären Ausdrücken effektiv verarbeitet
- Einfachere Deployment ohne NLTK-Downloads

### 2.3 Stopword-Entfernung
**Entscheidung**: Erweiterte Stopword-Liste für wissenschaftliche Texte basierend auf scikit-learn
**Begründung**:
- Verwendung von ENGLISH_STOP_WORDS aus scikit-learn
- Erweiterung um wissenschaftliche Begriffe (paper, study, research, etc.)
- Fokus auf inhaltlich relevante Begriffe
- Reduktion der Dimensionalität ohne externe Abhängigkeiten

## 3. Feature Engineering

### 3.1 TF-IDF Vektorisierung
**Entscheidung**: TF-IDF statt Bag-of-Words oder Word Embeddings
**Begründung**:
- **Vorteile**:
  - Berücksichtigung der Wortwichtigkeit
  - Skalierbarkeit für große Datensätze
  - Interpretierbarkeit der Features
- **Nachteile**:
  - Verlust semantischer Beziehungen
  - Keine Berücksichtigung der Wortreihenfolge

### 3.2 N-Gramme
**Entscheidung**: Unigramme und Bigramme (1,2)
**Begründung**:
- Erfassung von Wortkombinationen (z.B. "machine learning")
- Balance zwischen Informationsgehalt und Dimensionalität
- Vermeidung von Overfitting durch zu viele Features

### 3.3 Feature-Selektion
**Entscheidung**: Top 2000 Features basierend auf TF-IDF-Scores
**Begründung**:
- Reduktion der Dimensionalität für bessere Performance
- Fokus auf die wichtigsten Begriffe
- Vermeidung des Curse of Dimensionality
- Balance zwischen Informationsgehalt und Recheneffizienz

## 4. Dimensionsreduktion

### 4.1 TruncatedSVD (Singular Value Decomposition)
**Entscheidung**: TruncatedSVD statt PCA für große sparse TF-IDF-Matrizen
**Begründung**:
- **Vorteile**:
  - Effiziente Verarbeitung großer sparse Matrizen ohne .toarray()
  - Reduziert Speicherverbrauch um ~80%
  - Direkte Anwendung auf TF-IDF-Matrix möglich
  - Beibehaltung der wichtigsten Varianz
- **Nachteile**:
  - Lineare Dimensionsreduktion
  - Verlust nicht-linearer Beziehungen

### 4.2 t-SNE für Visualisierung
**Entscheidung**: t-SNE mit separater PCA-Vorverarbeitung für 15 Komponenten
**Begründung**:
- Separate PCA auf TF-IDF-Matrix mit .toarray() für t-SNE-Input
- Reduktion auf 15 Komponenten vor t-SNE für bessere Performance
- Erhaltung lokaler Strukturen in 2D-Visualisierung
- Nicht-lineare Dimensionsreduktion für Cluster-Visualisierung

## 5. Clustering-Analyse

### 5.1 AgglomerativeClustering (Hauptansatz)
**Entscheidung**: AgglomerativeClustering mit n_clusters=7 als Haupt-Clustering-Algorithmus
**Begründung**:
- **Vorteile**:
  - Hierarchische Cluster-Struktur
  - Robust gegenüber Ausreißern
  - Keine Annahme kugelförmiger Cluster
  - Für Textdaten oft besser geeignet als KMeans
- **Nachteile**:
  - Festgelegte Cluster-Anzahl (keine automatische Optimierung)
  - Höhere Rechenkomplexität

### 5.2 K-Means Clustering (Alternative)
**Entscheidung**: K-Means mit Silhouette-Score-Optimierung als alternative Methode
**Begründung**:
- **Vorteile**:
  - Automatische Optimierung der Cluster-Anzahl (2-10)
  - Skalierbarkeit für große Datensätze
  - Einfache Interpretation
  - Schnelle Ausführung
- **Nachteile**:
  - Annahme kugelförmiger Cluster
  - Sensitivität gegenüber Initialisierung

### 5.3 Alternative Methoden (nicht gewählt)
**DBSCAN**: 
- Vorteil: Keine feste Cluster-Anzahl
- Nachteil: Sensitivität gegenüber Parameter-Einstellung
- Grund: Weniger interpretierbar für Trend-Analyse

**Hierarchical Clustering**:
- Vorteil: Dendrogramm-Visualisierung
- Nachteile: Skalierungsprobleme bei großen Datensätzen
- Grund: Performance-Beschränkungen

## 6. Topic Modeling

### 6.1 LDA (Latent Dirichlet Allocation)
**Entscheidung**: LDA für Topic-Extraktion
**Begründung**:
- **Vorteile**:
  - Probabilistisches Modell
  - Interpretierbare Topics
  - Berücksichtigung von Wort-Wahrscheinlichkeiten
- **Nachteile**:
  - Festlegung der Topic-Anzahl
  - Annahme der Dirichlet-Verteilung

## 7. Validierung und Qualitätssicherung

### 7.1 Silhouette-Score
**Entscheidung**: Quantitative Cluster-Qualitätsbewertung
**Begründung**:
- Objektive Bewertung der Cluster-Trennung
- Wertebereich [-1, 1] mit klarer Interpretation
- Vergleichbarkeit verschiedener Parameter

### 7.2 Manuelle Überprüfung
**Entscheidung**: Stichprobenartige Kontrolle der Cluster-Zuordnungen
**Begründung**:
- Validierung der semantischen Kohärenz
- Identifikation von Problemen
- Verbesserung der Interpretierbarkeit

## 8. Kritische Reflexion der Entscheidungen

### 8.1 Stärken des gewählten Ansatzes
1. **Skalierbarkeit**: TF-IDF + PCA + K-Means ist für große Datensätze geeignet
2. **Interpretierbarkeit**: Alle Schritte sind transparent und nachvollziehbar
3. **Robustheit**: Bewährte Methoden mit guter Dokumentation
4. **Flexibilität**: Anpassung an verschiedene Datensätze möglich

### 8.2 Limitationen und Verbesserungsmöglichkeiten
1. **Semantische Beziehungen**: Word Embeddings könnten semantische Ähnlichkeiten besser erfassen
2. **Nicht-lineare Beziehungen**: Kernel-PCA oder UMAP könnten nicht-lineare Strukturen besser abbilden
3. **Dynamische Clustering**: Methoden wie DBSCAN könnten natürlichere Cluster finden
4. **Mehrsprachigkeit**: Aktuelle Implementierung fokussiert auf englische Texte

### 8.3 Ethische Überlegungen
1. **Datenschutz**: Sorgfältige Behandlung wissenschaftlicher Daten
2. **Transparenz**: Offenlegung aller methodischen Entscheidungen
3. **Bias-Bewusstsein**: Berücksichtigung möglicher Verzerrungen in den Daten
4. **Reproduzierbarkeit**: Dokumentation aller Parameter und Schritte

## 9. Iterative Verbesserung

### 9.1 Erste Iteration
- Einfache Implementierung mit Standard-Parametern
- Fokus auf Funktionalität und Grundverständnis

### 9.2 Zweite Iteration
- Parameter-Optimierung basierend auf ersten Ergebnissen
- Verbesserung der Textvorverarbeitung

### 9.3 Dritte Iteration
- Alternative Methoden testen
- Validierung und Qualitätssicherung

## 10. Empfehlungen für zukünftige Analysen

### 10.1 Technische Verbesserungen
1. **Deep Learning**: BERT oder Word2Vec für bessere semantische Repräsentation
2. **Ensemble-Methoden**: Kombination verschiedener Clustering-Algorithmen
3. **Zeitreihenanalyse**: Detailliertere Trend-Analyse über Zeit
4. **Interaktive Visualisierung**: Dashboards für explorative Analyse

### 10.2 Inhaltliche Erweiterungen
1. **Mehrsprachige Analyse**: Einbeziehung nicht-englischer Publikationen
2. **Interdisziplinäre Trends**: Analyse von Schnittstellen zwischen Bereichen
3. **Impact-Analyse**: Berücksichtigung von Zitierungen und Impact-Faktoren
4. **Kollaborationsanalyse**: Netzwerk-Analyse von Autoren und Institutionen

## 11. Erweiterte Verbesserungen (2025)

### 11.1 Effizientes Chunk-Loading großer JSON-Datensätze
**Entscheidung**: Zeilenweises, speicherschonendes Einlesen und optionales Filtern/Sampling
**Begründung**:
- Ermöglicht Analyse sehr großer Datensätze (z. B. arXiv) auf Standard-Hardware
- Flexible Filterung nach Kategorie/Jahr
- Vermeidung von Speicherüberläufen

### 11.2 t-SNE-Optimierung
**Entscheidung**: t-SNE nur auf 10–20 PCA-Komponenten anwenden
**Begründung**:
- Schnellere und stabilere Visualisierung
- Klarere Cluster-Strukturen
- Reduktion von Rechenzeit und Überfitting

### 11.3 Topic Modeling mit CountVectorizer
**Entscheidung**: LDA mit CountVectorizer statt TF-IDF
**Begründung**:
- Bessere, interpretierbare Topics
- LDA ist für Count-Daten konzipiert

### 11.4 Automatische sprechende Cluster-Labels
**Entscheidung**: Pro Cluster werden die Top-Wörter extrahiert und als Label ausgegeben
**Begründung**:
- Verständlichere Ergebnisse für Nutzer und Stakeholder
- Erleichtert die Interpretation der Cluster

### 11.5 JSON-Export mit robuster Typkonvertierung
**Entscheidung**: Cluster-IDs werden vor dem JSON-Export zu Strings konvertiert
**Begründung**:
- Vermeidung von TypeError bei np.int64-Schlüsseln
- Garantierte JSON-Kompatibilität unabhängig vom Datentyp der Cluster-IDs
- Robuster Export für verschiedene Python/NumPy-Versionen

### 11.6 (Optional) Automatisierte HTML-Reports
**Entscheidung**: Ergebnisse können als HTML-Report exportiert werden (z. B. mit Jinja2)
**Begründung**:
- Bessere Präsentation und Weiterverarbeitung der Analyse
- Einfache Integration in Dashboards oder Web-Interfaces

### 11.7 Tokenisierung ohne NLTK
**Entscheidung**: Verwendung einer robusten RegexpTokenizer-Lösung mit scikit-learn-Stopwords (statt NLTK)
**Begründung**:
- Keine Abhängigkeit von NLTK-Downloads oder -Ressourcen
- Portabel und sofort lauffähig auf jedem System
- Keine Fehlerquellen durch fehlende NLTK-Daten
- Einfache Wartung und Integration in beliebige Python-Umgebungen

### 11.8 Alternative Clustering-Algorithmen: AgglomerativeClustering
**Entscheidung**: Neben KMeans wird jetzt auch AgglomerativeClustering (hierarchisch) unterstützt.
**Begründung**:
- AgglomerativeClustering liefert hierarchische Strukturen und ist robust gegenüber Ausreißern
- Für Textdaten oft besser geeignet als KMeans, da keine kugelförmigen Cluster vorausgesetzt werden
- Die Auswahl des Algorithmus kann flexibel an die Daten angepasst werden

### 11.9 Erweiterte Jahr-Filterung und zufälliges Sampling
**Entscheidung**: Implementierung von `year_filter_min` und `year_filter_max` Parametern mit zufälligem Sampling über gefilterte Einträge.
**Begründung**:
- Ermöglicht Analyse spezifischer Zeiträume (z.B. 2015-2025) für aktuelle Trends
- Zufälliges Sampling über alle gefilterten Einträge vermeidet Bias gegenüber sequentiellem Sampling
- Warnungen bei unzureichenden Daten verbessern die Benutzerfreundlichkeit
- Progress-Anzeige und Jahr-Verteilung erhöhen die Transparenz

### 11.10 t-SNE-Optimierung mit separater PCA
**Entscheidung**: t-SNE verwendet eine separate PCA mit 15 Komponenten auf der TF-IDF-Matrix.
**Begründung**:
- Separate PCA-Vorverarbeitung für t-SNE-Input mit .toarray()
- Reduktion auf 15 Komponenten vor t-SNE für bessere Performance
- Optimierte Visualisierung durch reduzierte Dimensionalität
- Klarere Cluster-Strukturen in der 2D-Visualisierung

### 11.11 Einheitliche Farbpalette für Visualisierungen
**Entscheidung**: Implementierung einer konsistenten Farbpalette für alle Plots.
**Begründung**:
- Verbessert die visuelle Konsistenz und Professionalität der Analysen
- Erleichtert die Interpretation und den Vergleich verschiedener Visualisierungen
- Standardisierte Farben für Cluster, Kategorien und Trends
- Bessere Farbblind-freundliche Palette mit ausreichendem Kontrast

### 11.12 Output-Organisation in separatem Ordner
**Entscheidung**: Alle Ausgaben (Visualisierungen, JSON, CSV) werden im `Output/`-Ordner gespeichert.
**Begründung**:
- Bessere Projektstruktur und Übersichtlichkeit
- Vermeidung von Datei-Clutter im Hauptverzeichnis
- Einfacheres Git-Management (Output-Ordner kann ignoriert werden)
- Professionelle Organisation für Präsentationen und Berichte
- Automatische Ordner-Erstellung falls nicht vorhanden 