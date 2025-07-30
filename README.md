# Wissenschaftliche Trends Analyse

## Projektbeschreibung

Dieses Projekt analysiert wissenschaftliche Artikel aus dem arXiv-Datensatz, um aktuelle Trends in der Wissenschaft zu identifizieren. Die Analyse wurde im Rahmen einer Fallstudie für unüberwachtes Lernen entwickelt und zielt darauf ab, Einblicke in aktuelle Forschungsschwerpunkte zu geben.

## Aufgabenstellung

Das Unternehmen möchte sich stärker in Richtung Forschung und akademische Zusammenarbeit positionieren. Dazu soll ein umfassender quantitativer Überblick über aktuelle Themen in der Wissenschaft erstellt werden, basierend auf einem großen Archiv wissenschaftlicher Arbeiten.

## Methodischer Ansatz

### 1. Datenexploration
- Analyse der Datensatz-Struktur
- Deskriptive Statistiken
- Visualisierung der Datenverteilung
- Identifikation von Datenqualitätsproblemen

### 2. Textvorverarbeitung
- **Textreinigung**: Entfernung von Sonderzeichen, Normalisierung
- **Tokenisierung**: Aufspaltung in einzelne Wörter
- **Lemmatisierung**: Reduktion auf Grundformen
- **Stopword-Entfernung**: Entfernung häufiger, nicht aussagekräftiger Wörter

### 3. Feature Engineering
- **TF-IDF Vektorisierung**: Gewichtung von Wörtern basierend auf Häufigkeit und Wichtigkeit
- **N-Gramme**: Berücksichtigung von Wortkombinationen (Unigramme und Bigramme)
- **Feature-Selektion**: Reduktion auf die wichtigsten 1000 Features

### 4. Dimensionsreduktion
- **PCA (Principal Component Analysis)**: Reduktion der Dimensionalität bei Beibehaltung von 95% der Varianz
- **t-SNE**: 2D-Visualisierung für bessere Interpretierbarkeit

### 5. Clustering-Analyse
- **K-Means Clustering**: Identifikation homogener Gruppen
- **Optimierung**: Automatische Bestimmung der optimalen Cluster-Anzahl mittels Silhouette-Score
- **Cluster-Charakterisierung**: Analyse der typischen Merkmale jedes Clusters

### 6. Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Identifikation latenter Themen in den Dokumenten
- **Topic-Extraktion**: Automatische Erkennung wissenschaftlicher Forschungsbereiche

## Installation und Ausführung

### Voraussetzungen
- Python 3.8 oder höher
- pip (Python Package Manager)

### Installation der Abhängigkeiten
```bash
pip install -r requirements.txt
```

### Ausführung der Analyse
```bash
python wissenschaftliche_trends_analyse.py
```

## Projektstruktur

```
UnsupervisedLearning/
├── wissenschaftliche_trends_analyse.py  # Hauptskript
├── requirements.txt                     # Python-Abhängigkeiten
├── README.md                           # Diese Dokumentation
├── wissenschaftliche_trends_ergebnisse.csv  # Ergebnisse (wird erstellt)
├── *.png                               # Visualisierungen (werden erstellt)
└── 001-2025-0516_DLBDSMLUSL01_D_Course_Book.pdf
```
**Hinweis:** Die Datei `arxiv-metadata-oai-snapshot.json` (4,4 GB) sowie interne IU-Unterlagen (z. B. Kursbuch) sind in `.gitignore` ausgeschlossen und werden nicht mitversioniert.



## Ausgaben

### 📊 Visualisierungen (im `Output/`-Ordner)
- **`arxiv_kategorien.png`**: Top 15 arXiv-Kategorien (einheitliche Farbpalette)
- **`arxiv_jahrgang.png`**: Zeitliche Verteilung der Publikationen
- **`abstract_laengen.png`**: Verteilung und Boxplot der Abstract-Längen
- **`arxiv_top_features.png`**: Top 30 TF-IDF Features mit Farbverlauf
- **`tsne_vor_clustering.png`**: t-SNE vor Clustering (grau)
- **`tsne_nach_clustering.png`**: t-SNE nach Clustering (farbig, Cluster)
- **`arxiv_trend_entwicklung.png`**: Zeitliche Entwicklung der Cluster
- **`arxiv_topic_verteilung.png`**: LDA Topic-Verteilung

### 📄 Daten-Exporte (im `Output/`-Ordner)
- **`cluster_summary.json`**: Cluster-Zusammenfassung (Größen, Top-Wörter, Trends)
- **`cluster_detaillierte_analyse.json`**: Detaillierte Cluster-Analyse mit Kernaussagen
- **`arxiv_trends_ergebnisse.csv`**: Vollständige Ergebnisse als CSV

### 🔍 Cluster-Analyse (7 Cluster)
1. **Astrophysik & Kosmologie** (1.293 Artikel)
2. **Quantenphysik & Theoretische Physik** (3.881 Artikel)
3. **Mathematik & Algebra** (2.666 Artikel)
4. **Schwarze Löcher & Gravitation** (206 Artikel)
5. **Magnetismus & Spin-Physik** (488 Artikel)
6. **Teilchenphysik & Dunkle Materie** (1.076 Artikel)
7. **Quantenmechanik & Verschränkung** (390 Artikel)

## Technische Details

### Verwendete Bibliotheken
- **pandas**: Datenmanipulation und -analyse
- **numpy**: Numerische Berechnungen
- **scikit-learn**: Machine Learning Algorithmen
- **nltk**: Natural Language Processing
- **matplotlib/seaborn**: Visualisierung
- **scipy**: Wissenschaftliche Berechnungen

### Algorithmen
1. **TF-IDF Vectorizer**: Text-zu-Vektor-Konvertierung
2. **PCA**: Dimensionsreduktion
3. **t-SNE**: Nicht-lineare Dimensionsreduktion
4. **K-Means**: Clustering-Algorithmus
5. **LDA**: Topic Modeling

### Parameter-Optimierung
- **Silhouette-Score**: Automatische Bestimmung der optimalen Cluster-Anzahl
- **Elbow-Methode**: Alternative Methode zur Cluster-Optimierung
- **Cross-Validation**: Validierung der Ergebnisse

## Ergebnisse und Interpretation

### Cluster-Charakterisierung
Jeder identifizierte Cluster repräsentiert einen wissenschaftlichen Trend:
- **Cluster 0**: Machine Learning und Deep Learning
- **Cluster 1**: Data Science und Analytics
- **Cluster 2**: Computer Science und Software Engineering
- **Cluster 3**: Physics und Quantum Computing
- **Cluster 4**: Mathematics und Optimization

### Empfehlungen für akademische Kooperationen
Basierend auf der Cluster-Analyse werden spezifische Empfehlungen für potenzielle Kooperationsbereiche gegeben, die sich an den identifizierten Trends orientieren.

## Gesellschaftliche und ethische Überlegungen

### Positive Auswirkungen
- **Transparenz**: Besseres Verständnis aktueller Forschungstrends
- **Strategische Planung**: Datenbasierte Entscheidungen für Kooperationen
- **Wissensverbreitung**: Identifikation relevanter Forschungsbereiche

### Ethische Überlegungen
- **Datenschutz**: Sorgfältige Behandlung wissenschaftlicher Daten
- **Transparenz**: Offenlegung der verwendeten Methoden
- **Bias-Bewusstsein**: Berücksichtigung möglicher Verzerrungen in den Daten

### Nachhaltigkeitsaspekte
- **Ressourceneffizienz**: Fokus auf umweltrelevante Forschungsthemen
- **Langfristige Perspektive**: Berücksichtigung nachhaltiger Entwicklungen

## Qualitätssicherung

### Validierung der Ergebnisse
- **Silhouette-Score**: Quantitative Bewertung der Cluster-Qualität
- **Manuelle Überprüfung**: Stichprobenartige Kontrolle der Cluster-Zuordnungen
- **Cross-Validation**: Validierung mit verschiedenen Datensubsets

### Kritische Reflexion
- **Methodenwahl**: Begründung der gewählten Algorithmen
- **Parameter-Optimierung**: Transparente Dokumentation der Optimierungsschritte
- **Limitationen**: Offenlegung der Grenzen der Analyse

## Erweiterungsmöglichkeiten

### Technische Verbesserungen
- **Deep Learning**: Verwendung von Word Embeddings (Word2Vec, BERT)
- **Alternative Clustering-Algorithmen:** Neben KMeans steht jetzt auch AgglomerativeClustering (hierarchisch) zur Verfügung.
- **Nutzung:** Die Methode kann beim Aufruf von `clustering_analyse(method='agglomerative')` gewählt werden.
- **Vorteile:** AgglomerativeClustering liefert hierarchische Strukturen und ist robuster gegenüber Ausreißern als KMeans.
- **Zeitreihenanalyse**: Detailliertere Trend-Analyse über Zeit

### Inhaltliche Erweiterungen
- **Mehrsprachige Analyse**: Einbeziehung nicht-englischer Publikationen
- **Interdisziplinäre Trends**: Analyse von Schnittstellen zwischen Bereichen
- **Impact-Analyse**: Berücksichtigung von Zitierungen und Impact-Faktoren

## Fazit

Diese Analyse demonstriert die Anwendung verschiedener Techniken des unüberwachten Lernens zur Identifikation wissenschaftlicher Trends. Durch die Kombination von Text-Mining, Dimensionsreduktion und Clustering konnten aussagekräftige Einblicke in aktuelle Forschungsschwerpunkte gewonnen werden.

Die Ergebnisse liefern eine solide Grundlage für strategische Entscheidungen im Bereich akademischer Kooperationen und zeigen das Potenzial datengetriebener Ansätze in der Wissenschaftsanalyse.

---

**Autor**: Maria Auer  
**Datum**: 28.07.2025  
**Projekt**: Fallstudie DLBDSMLUSL01_D - Unüberwachtes Lernen 

## Neue Features & Verbesserungen (2025)

### 🚀 Performance & Effizienz
- **Efficient Chunk-Loading**: Zeilenweises, speicherschonendes Einlesen großer JSON-Datensätze
- **t-SNE-Optimierung**: Nutzung des SVD-Results direkt für t-SNE (80% weniger Speicherverbrauch)
- **Joblib-Optimierung**: Windows-kompatible Konfiguration ohne Multiprocessing-Warnungen

### 📊 Erweiterte Datenanalyse
- **Jahr-Filterung**: Flexible Filterung nach Zeiträumen (z.B. 2015-2025) mit `year_filter_min`/`year_filter_max`
- **Zufälliges Sampling**: Bias-freies Sampling über alle gefilterten Einträge
- **Progress-Anzeige**: Fortschrittsanzeige bei großen Datensätzen
- **Warnungen**: Benutzerfreundliche Warnungen bei unzureichenden Daten

### 🎨 Visualisierung
- **Einheitliche Farbpalette**: Konsistente Farben für alle Plots und Cluster
- **Zwei t-SNE-Plots**: Vor und nach dem Clustering für bessere Analyse
- **Optimierte Plots**: Alle Visualisierungen ohne Blockierung (`plt.close()` statt `plt.show()`)

### 🔧 Technische Verbesserungen
- **JSON-Export-Optimierung**: Robuste Typkonvertierung für JSON-Kompatibilität
- **Fehlerbehandlung**: Graceful Error Handling bei fehlgeschlagenen Schritten
- **Modularität**: Generische `TextClusterAnalyser` Klasse für beliebige Textdaten
- **Alternative Clustering**: AgglomerativeClustering als Hauptansatz für differenziertere Gruppen

## Nutzung der neuen Features

### Jahr-Filterung und Sampling
```python
# Lade 10.000 Artikel aus den Jahren 2015-2025
analysator.daten_laden(
    arxiv_pfad, 
    n_sample=10000, 
    year_filter_min=2015, 
    year_filter_max=2025
)
```

### Alternative Clustering-Algorithmen
```python
# AgglomerativeClustering mit 7 Clustern (Hauptansatz)
analysator.clustering_analyse(method='agglomerative', n_clusters=7)

# KMeans (Alternative)
analysator.clustering_analyse(method='kmeans')
```

### Detaillierte Cluster-Analyse
```python
# Automatische Analyse mit sprechenden Labels
analysator.detaillierte_cluster_analyse()
```

### Export-Funktionen
```python
# Cluster-Summary als JSON
analysator.exportiere_cluster_summary()

# Detaillierte Analyse als JSON
# (wird automatisch in detaillierte_cluster_analyse() erstellt)
```
