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
- **Textreinigung**: Entfernung von LaTeX-Formeln, Sonderzeichen und Normalisierung
- **Tokenisierung**: Regex-basierte Tokenisierung ohne NLTK-Abhängigkeiten
- **Stopword-Entfernung**: Erweiterte Liste mit wissenschaftlichen Begriffen (paper, study, research, etc.)
- **Feature-Engineering**: TF-IDF mit 2000 Features und N-Grammen (1,3)

### 3. Feature Engineering
- **TF-IDF Vektorisierung**: Konvertierung von Text zu numerischen Features
- **Feature-Anzahl**: 2000 Features für alle Analysen (Hauptanalyse und detaillierte Cluster-Analyse)
- **N-Gramme**: Unigramme, Bigramme und Trigramme (1,3)
- **Feature-Selektion**: Automatische Auswahl der wichtigsten Begriffe

### 4. Dimensionsreduktion
- **TruncatedSVD**: Dimensionsreduktion für große sparse Matrizen (100 Komponenten ≈ 70% Varianz)
- **t-SNE**: Nicht-lineare Dimensionsreduktion für Visualisierung (erste 15 SVD-Komponenten)
- **Speichereffizienz**: Direkte Verarbeitung von TF-IDF-Matrizen ohne .toarray()

### 5. Clustering-Analyse
- **K-Means**: Standard-Clustering-Algorithmus mit automatischer Silhouette-Score-Optimierung
- **AgglomerativeClustering**: Empfohlener Algorithmus für die Fallstudie mit n_clusters=6 (Standard) oder 7 (empfohlen)
- **Cluster-Labels**: Automatische Generierung sprechender Beschriftungen
- **Validierung**: Manuelle Überprüfung und quantitative Bewertung

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
├── arxiv_daten_analyse.py              # Hauptskript für arXiv-Analyse
├── wissenschaftliche_trends_analyse.py # Alternative Analyse-Skript
├── methodische_entscheidungen.md       # Dokumentation der methodischen Entscheidungen
├── requirements.txt                    # Python-Abhängigkeiten
├── README.md                           # Diese Dokumentation
├── Output/                             # Ausgabe-Ordner für alle Ergebnisse
│   ├── *.png                           # Visualisierungen
│   ├── *.json                          # JSON-Exporte
├── .gitignore                          # Git-Ignore-Datei
└── arxiv-metadata-oai-snapshot.json    # arXiv-Datensatz (4,5 GB)
```
**Hinweis:** Die Datei `arxiv-metadata-oai-snapshot.json` (4,4 GB) ist in `.gitignore` ausgeschlossen und wird nicht mitversioniert.



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

### 🔍 Cluster-Analyse
- **6 Cluster** werden standardmäßig automatisch identifiziert
- **7 Cluster** empfohlen für die Fallstudie (explizit angeben)
- **Dynamische Labels** basierend auf Top-Wörtern jedes Clusters
- **Cluster-Größen** variieren je nach Zufalls-Sampling
- **Detaillierte Analyse** mit Kernaussagen für jeden Cluster

## Technische Details

### Verwendete Bibliotheken
- **pandas**: Datenmanipulation und -analyse
- **numpy**: Numerische Berechnungen
- **scikit-learn**: Machine Learning Algorithmen
- **matplotlib/seaborn**: Visualisierung
- **scipy**: Wissenschaftliche Berechnungen
- **regex**: Textverarbeitung (ohne NLTK-Downloads)

### Algorithmen
1. **TF-IDF Vectorizer**: Text-zu-Vektor-Konvertierung mit 2000 Features
2. **TruncatedSVD**: Dimensionsreduktion für große sparse Matrizen
3. **t-SNE**: Nicht-lineare Dimensionsreduktion für Visualisierung
4. **K-Means**: Standard-Clustering-Algorithmus mit Silhouette-Optimierung
5. **AgglomerativeClustering**: Empfohlener Clustering-Algorithmus für Fallstudie
6. **LDA**: Topic Modeling

### Parameter-Optimierung
- **K-Means**: Standard mit automatischer Silhouette-Score-Optimierung (2-10 Cluster)
- **AgglomerativeClustering**: Empfohlen für Fallstudie mit n_clusters=6 (Standard) oder 7 (empfohlen)
- **Manuelle Validierung**: Stichprobenartige Überprüfung der Cluster-Qualität

### Verfügbare Clustering-Methoden
```python
# Standard (K-Means mit automatischer Optimierung):
analysator.clustering_analyse(method='kmeans')

# Empfohlen für Fallstudie (AgglomerativeClustering):
analysator.clustering_analyse(method='agglomerative', n_clusters=7)

# Alternative (AgglomerativeClustering mit 6 Clustern):
analysator.clustering_analyse(method='agglomerative')
```

## Ergebnisse und Interpretation

### Cluster-Charakterisierung
- **Automatische Identifikation** wissenschaftlicher Trends basierend auf TF-IDF-Analyse
- **Dynamische Cluster-Labels** werden aus den Top-Wörtern jedes Clusters generiert
- **Inhaltliche Profile** mit Kernaussagen für jeden identifizierten Cluster
- **Zeitliche Entwicklung** der Cluster über die analysierten Jahre

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
- **Silhouette-Score**: Quantitative Bewertung der Cluster-Qualität (nur für KMeans)
- **Manuelle Überprüfung**: Stichprobenartige Kontrolle der Cluster-Zuordnungen
- **Automatische Cluster-Labels**: Sprechende Beschriftungen basierend auf Top-Wörtern
- **Detaillierte Cluster-Analyse**: Kernaussagen und inhaltliche Profile für jeden Cluster

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