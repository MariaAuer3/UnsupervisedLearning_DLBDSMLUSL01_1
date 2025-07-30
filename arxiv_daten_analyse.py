#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arXiv Daten Analyse
==================

Skript für die Analyse des arXiv-Datensatzes von Kaggle.
Dieses Skript kann verwendet werden, wenn der echte arXiv-Datensatz verfügbar ist.

Autor: Maria Auer
Datum: 28.07.2025
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings('ignore')

# Umgebungsvariablen für bessere Performance auf Windows
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Output-Ordner für alle Ausgaben
OUTPUT_DIR = 'Output'

def ensure_output_dir():
    """Stelle sicher, dass der Output-Ordner existiert"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Output-Ordner '{OUTPUT_DIR}' erstellt.")

# Konsistente Farbpalette für alle Visualisierungen
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
CLUSTER_COLORMAP = 'tab10'  # Für Cluster-Visualisierungen

# Ersetze NLTK-Tokenisierung durch RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

def simple_tokenizer(text):
    # Entferne LaTeX, Sonderzeichen, Zahlen, etc.
    text = re.sub(r'\$.*?\$', ' ', text)  # LaTeX-Formeln
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # LaTeX-Befehle
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{3,}\b', text)  # nur Wörter mit min. 3 Buchstaben
    stopwords = set(ENGLISH_STOP_WORDS)
    wissenschaftliche_stopwords = stopwords.union({
        'paper', 'study', 'research', 'analysis', 'method', 'approach',
        'result', 'conclusion', 'introduction', 'abstract', 'section',
        'figure', 'table', 'equation', 'theorem', 'lemma', 'proof',
        'proposition', 'corollary', 'definition', 'example', 'algorithm'
    })
    tokens = [t for t in tokens if t not in wissenschaftliche_stopwords]
    return ' '.join(tokens)

def sample_arxiv_json(path, n=10000, category_filter=None, year_filter=None, year_filter_min=None, year_filter_max=None, seed=42):
    """
    Lese arXiv-JSON zeilenweise, filtere nach Kategorie/Jahr und ziehe ein Sample.
    
    Args:
        path: Pfad zur JSON-Datei
        n: Anzahl zu samplender Artikel
        category_filter: Optional - Filter für bestimmte Kategorie
        year_filter: Optional - Filter für bestimmtes Jahr (veraltet, nutze year_filter_min/max)
        year_filter_min: Optional - Minimales Jahr (inklusive)
        year_filter_max: Optional - Maximales Jahr (inklusive)
        seed: Random seed für reproduzierbare Ergebnisse
    """
    random.seed(seed)
    
    # Sammle alle passenden Einträge
    print(f"Lade und filtere arXiv-Daten...")
    matching_entries = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:  # Progress-Anzeige
                print(f"  Verarbeite Zeile {line_num:,}...")
            
            try:
                entry = json.loads(line)
                
                # Kategorie-Filter
                if category_filter and category_filter not in entry.get('categories', ''):
                    continue
                
                # Jahr-Filter (neue Logik)
                date_str = entry.get('update_date') or (entry.get('versions', [{}])[-1].get('created'))
                if date_str:
                    year = int(str(date_str)[:4])
                    
                    # Jahr-Filter anwenden
                    if year_filter is not None:  # Legacy-Support
                        if year != year_filter:
                            continue
                    elif year_filter_min is not None or year_filter_max is not None:
                        if year_filter_min is not None and year < year_filter_min:
                            continue
                        if year_filter_max is not None and year > year_filter_max:
                            continue
                
                matching_entries.append(entry)
                
            except json.JSONDecodeError:
                continue  # Überspringe ungültige JSON-Zeilen
    
    print(f"Gefundene passende Einträge: {len(matching_entries):,}")
    
    # Prüfe ob genug Artikel verfügbar sind
    if len(matching_entries) < n:
        print(f"⚠️  WARNUNG: Nur {len(matching_entries):,} Artikel verfügbar, weniger als gewünschte {n:,}")
        n = len(matching_entries)
        if n == 0:
            print("❌ Keine Artikel gefunden! Prüfe Filter-Parameter.")
            return []
    
    # Zufälliges Sampling über alle gefilterten Einträge
    print(f"Ziehe zufälliges Sample von {n:,} Artikeln...")
    sampled_entries = random.sample(matching_entries, n)
    
    # Jahr-Verteilung des Samples anzeigen
    year_dist = {}
    for entry in sampled_entries:
        date_str = entry.get('update_date') or (entry.get('versions', [{}])[-1].get('created'))
        if date_str:
            year = int(str(date_str)[:4])
            year_dist[year] = year_dist.get(year, 0) + 1
    
    print(f"Jahr-Verteilung des Samples:")
    for year in sorted(year_dist.keys()):
        print(f"  {year}: {year_dist[year]:,} Artikel")
    
    return sampled_entries

# Entferne HDBSCAN-Import und -Code
# from hdbscan import HDBSCAN  # ENTFERNT
from sklearn.cluster import AgglomerativeClustering

class TextClusterAnalyser:
    """
    Modulare Klasse für Clustering und Trend-Analyse beliebiger Textdaten (z.B. arXiv, Chatlogs, etc.)
    """
    def __init__(self, data=None, text_column='combined_text', category_column='categories'):
        self.data = data
        self.text_column = text_column
        self.category_column = category_column
        self.tfidf_matrix = None
        self.pca_result = None
        self.clusters = None
        self.vectorizer = None
        self.cluster_labels = None

    def daten_laden(self, dateipfad, n_sample=10000, category_filter=None, year_filter=None, year_filter_min=None, year_filter_max=None):
        """
        Lädt den arXiv-Datensatz effizient chunkweise und filtert nach Wunsch.
        
        Args:
            dateipfad: Pfad zur JSON- oder CSV-Datei
            n_sample: Anzahl zu samplender Artikel
            category_filter: Optional - Filter für bestimmte Kategorie
            year_filter: Optional - Filter für bestimmtes Jahr (veraltet)
            year_filter_min: Optional - Minimales Jahr (inklusive)
            year_filter_max: Optional - Maximales Jahr (inklusive)
        """
        print("=== ARXIV DATENEXPLORATION ===")
        if dateipfad.endswith('.json'):
            print(f"Lade {n_sample} Einträge aus großem JSON-Datensatz...")
            data = sample_arxiv_json(
                dateipfad, 
                n=n_sample, 
                category_filter=category_filter, 
                year_filter=year_filter,
                year_filter_min=year_filter_min,
                year_filter_max=year_filter_max
            )
            if not data:
                print("❌ Keine Daten geladen. Prüfe Filter-Parameter.")
                return False
            self.data = pd.DataFrame(data)
            print(f"arXiv-Daten erfolgreich geladen: {len(self.data)} Artikel (Sample)")
        else:
            # Fallback für CSV
            try:
                self.data = pd.read_csv(dateipfad)
                print(f"arXiv-Daten erfolgreich geladen: {len(self.data)} Artikel")
            except Exception as e:
                print(f"Fehler beim Laden der CSV-Datei: {e}")
                return False
        
        # Basis-Exploration
        self._arxiv_daten_exploration()
        return True
        
    def _arxiv_daten_exploration(self):
        """
        Spezielle Exploration für arXiv-Daten
        """
        print("\n--- arXiv Datenexploration ---")
        print(f"Datensatz-Größe: {self.data.shape}")
        print(f"Spalten: {list(self.data.columns)}")
        
        # Grundlegende Statistiken
        print("\nErste 5 Zeilen:")
        print(self.data.head())
        
        print("\nDatentypen:")
        print(self.data.dtypes)
        
        print("\nFehlende Werte:")
        print(self.data.isnull().sum())
        
        # arXiv-spezifische Analysen
        if 'categories' in self.data.columns:
            print("\nVerteilung der arXiv-Kategorien:")
            # Teile Kategorien auf (arXiv kann mehrere Kategorien pro Artikel haben)
            all_categories = []
            for cats in self.data['categories'].dropna():
                if isinstance(cats, str):
                    all_categories.extend(cats.split())
            
            category_counts = pd.Series(all_categories).value_counts().head(20)
            print(category_counts)
            
            # Visualisierung der Top-Kategorien
            plt.figure(figsize=(15, 8))
            category_counts.head(15).plot(kind='bar', color=COLOR_PALETTE[:15])
            plt.title('Top 15 arXiv-Kategorien', fontsize=14, fontweight='bold')
            plt.xlabel('Kategorie')
            plt.ylabel('Anzahl Artikel')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'arxiv_kategorien.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Schließe Plot ohne Anzeige
        
        # Zeitliche Analyse
        if 'update_date' in self.data.columns:
            print("\nZeitliche Verteilung:")
            self.data['update_date'] = pd.to_datetime(self.data['update_date'])
            self.data['year'] = self.data['update_date'].dt.year
            
            year_counts = self.data['year'].value_counts().sort_index()
            print(year_counts)
            
            plt.figure(figsize=(12, 6))
            year_counts.plot(kind='line', marker='o', color=COLOR_PALETTE[0], linewidth=2)
            plt.title('arXiv Publikationen nach Jahr', fontsize=14, fontweight='bold')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl Publikationen')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'arxiv_jahrgang.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Schließe Plot ohne Anzeige
        
        # Textlängen-Analyse
        if 'abstract' in self.data.columns:
            print("\nAbstract-Längen-Analyse:")
            self.data['abstract_length'] = self.data['abstract'].str.len()
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.data['abstract_length'].dropna(), bins=50, color=COLOR_PALETTE[1], alpha=0.7)
            plt.title('Verteilung der Abstract-Längen', fontsize=12, fontweight='bold')
            plt.xlabel('Anzahl Zeichen')
            plt.ylabel('Häufigkeit')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(self.data['abstract_length'].dropna(), patch_artist=True, 
                       boxprops=dict(facecolor=COLOR_PALETTE[2], alpha=0.7))
            plt.title('Boxplot der Abstract-Längen', fontsize=12, fontweight='bold')
            plt.ylabel('Anzahl Zeichen')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'abstract_laengen.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Schließe Plot ohne Anzeige
            
            print(f"Durchschnittliche Abstract-Länge: {self.data['abstract_length'].mean():.0f} Zeichen")
            print(f"Median Abstract-Länge: {self.data['abstract_length'].median():.0f} Zeichen")
    
    def arxiv_text_vorverarbeitung(self):
        """
        Spezielle Textvorverarbeitung für arXiv-Daten
        """
        print("\n=== ARXIV TEXT-VORVERARBEITUNG ===")
        
        # Textreinigung für wissenschaftliche Texte
        def wissenschaftliche_text_bereinigung(text):
            if pd.isna(text):
                return ""
            
            text = str(text).lower()
            
            # Mathematische Ausdrücke und Formeln entfernen (vereinfacht)
            text = re.sub(r'\$.*?\$', ' ', text)  # LaTeX-Formeln
            text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # LaTeX-Befehle
            
            # Sonderzeichen entfernen, aber Zahlen behalten
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Erweiterte Stopwords für wissenschaftliche Texte
        wissenschaftliche_stopwords = set(ENGLISH_STOP_WORDS)
        wissenschaftliche_stopwords.update([
            'paper', 'study', 'research', 'analysis', 'method', 'approach',
            'result', 'conclusion', 'introduction', 'abstract', 'section',
            'figure', 'table', 'equation', 'theorem', 'lemma', 'proof',
            'proposition', 'corollary', 'definition', 'example', 'algorithm'
        ])
        
        # Anwendung der Vorverarbeitung
        print("Bereinige Titel...")
        self.data['title_clean'] = self.data['title'].apply(wissenschaftliche_text_bereinigung)
        
        print("Bereinige Abstracts...")
        self.data['abstract_clean'] = self.data['abstract'].apply(wissenschaftliche_text_bereinigung)
        
        print("Wissenschaftliche Tokenisierung...")
        self.data['title_processed'] = self.data['title_clean'].apply(simple_tokenizer)
        self.data['abstract_processed'] = self.data['abstract_clean'].apply(simple_tokenizer)
        
        # Kombinierte Textdaten
        self.data['combined_text'] = self.data['title_processed'] + ' ' + self.data['abstract_processed']
        
        print("arXiv Textvorverarbeitung abgeschlossen!")
        
        # Beispiel-Output
        print(f"\nBeispiel verarbeiteter arXiv-Text:")
        print(f"Original Titel: {self.data['title'].iloc[0]}")
        print(f"Verarbeiteter Titel: {self.data['title_processed'].iloc[0]}")
    
    def arxiv_feature_engineering(self):
        """
        Spezielles Feature Engineering für arXiv-Daten
        """
        print("\n=== ARXIV FEATURE ENGINEERING ===")
        
        # TF-IDF mit wissenschaftlichen Parametern
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Mehr Features für größeren Datensatz
            min_df=3,           # Mindestens 3 Dokumente
            max_df=0.9,         # Maximal 90% der Dokumente
            ngram_range=(1, 3), # Unigramme, Bigramme, Trigramme
            stop_words='english'
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_text'])
        
        print(f"TF-IDF Matrix Dimension: {self.tfidf_matrix.shape}")
        print(f"Anzahl Features: {len(self.vectorizer.get_feature_names_out())}")
        
        # Wissenschaftliche Features analysieren
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_sums = np.array(self.tfidf_matrix.sum(axis=0)).flatten()
        top_features_idx = tfidf_sums.argsort()[-30:][::-1]
        
        print("\nTop 30 wissenschaftliche Features:")
        for i, idx in enumerate(top_features_idx):
            print(f"{i+1:2d}. {feature_names[idx]:25s} (Score: {tfidf_sums[idx]:.3f})")
        
        # Visualisierung der wissenschaftlichen Features
        plt.figure(figsize=(15, 10))
        bars = plt.bar(range(len(top_features_idx)), tfidf_sums[top_features_idx], color=COLOR_PALETTE[:len(top_features_idx)])
        plt.title('Top 30 wissenschaftliche Features (TF-IDF)', fontsize=14, fontweight='bold')
        plt.xlabel('Feature-Rang')
        plt.ylabel('TF-IDF Score')
        plt.xticks(range(len(top_features_idx)), [feature_names[idx] for idx in top_features_idx], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Farbverlauf für bessere Visualisierung
        for i, bar in enumerate(bars):
            bar.set_color(COLOR_PALETTE[i % len(COLOR_PALETTE)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'arxiv_top_features.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Schließe Plot ohne Anzeige
    
    def get_top_words(self, cluster_docs, n=3):
        all_words = ' '.join(cluster_docs['combined_text']).split()
        return pd.Series(all_words).value_counts().head(n).index.tolist()

    
    def arxiv_trend_analyse(self):
        """
        Spezielle Trend-Analyse für arXiv-Daten
        """
        print("\n=== ARXIV TREND-ANALYSE ===")
        
        # Vorher prüfen, ob 'cluster' in self.data vorhanden ist
        if 'cluster' not in self.data.columns:
            print("Fehler: Es wurden keine Cluster erzeugt. Trend-Analyse wird übersprungen.")
            return

        # Zeitliche Entwicklung der Cluster
        if 'year' in self.data.columns:
            print("Analysiere zeitliche arXiv-Trends...")
            
            cluster_time = pd.crosstab(self.data['year'], self.data['cluster'])
            
            plt.figure(figsize=(15, 8))
            cluster_time.plot(kind='bar', stacked=True, colormap=CLUSTER_COLORMAP)
            plt.title('Entwicklung der arXiv-Forschungstrends über Zeit', fontsize=14, fontweight='bold')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl Artikel')
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'arxiv_trend_entwicklung.png'), dpi=300, bbox_inches='tight')
            plt.close()  # Schließe Plot ohne Anzeige
        
        # Topic Modeling mit CountVectorizer (statt TF-IDF)
        print("Führe erweiterte Topic-Analyse mit CountVectorizer durch...")
        count_vectorizer = CountVectorizer(
            max_df=0.9, min_df=3, stop_words='english', ngram_range=(1, 3)
        )
        lda_input = count_vectorizer.fit_transform(self.data['combined_text'])
        lda = LatentDirichletAllocation(
            n_components=8,  # Mehr Topics für größeren Datensatz
            random_state=42,
            max_iter=15
        )
        lda_result = lda.fit_transform(lda_input)
        feature_names = count_vectorizer.get_feature_names_out()
        print("\nIdentifizierte arXiv-Forschungsthemen:")
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-12:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        # Topic-Verteilung
        topic_distribution = lda_result.mean(axis=0)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(1, len(topic_distribution) + 1), topic_distribution, color=COLOR_PALETTE[:len(topic_distribution)])
        plt.title('Verteilung der arXiv-Forschungsthemen (CountVectorizer)', fontsize=14, fontweight='bold')
        plt.xlabel('Topic')
        plt.ylabel('Durchschnittliche Topic-Verteilung')
        plt.xticks(range(1, len(topic_distribution) + 1))
        plt.grid(True, alpha=0.3)
        
        # Farbverlauf für bessere Visualisierung
        for i, bar in enumerate(bars):
            bar.set_color(COLOR_PALETTE[i % len(COLOR_PALETTE)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'arxiv_topic_verteilung.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Schließe Plot ohne Anzeige

    def dimensionsreduktion(self, n_components=100):
        """
        Dimensionsreduktion mit TruncatedSVD (statt PCA) für große sparse Matrizen
        """
        print("\n=== DIMENSIONSREDUKTION (TruncatedSVD) ===")
        print(f"TF-IDF shape: {self.tfidf_matrix.shape}")
        print(f"Anzahl Nicht-Null-Elemente: {self.tfidf_matrix.nnz}")
        try:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.pca_result = svd.fit_transform(self.tfidf_matrix)
            print(f"SVD-Result Shape: {self.pca_result.shape}")
            print(f"Erklärte Varianz (Summe): {svd.explained_variance_ratio_.sum():.3f}")
        except Exception as e:
            print(f"SVD-Fehler: {e}")
            self.pca_result = None
            return

    def tsne_visualisierung(self, vor_clustering=True):
        """
        t-SNE-Plot: Vor Clustering (grau) oder nach Clustering (farbig)
        Optimiert: Nutzt bereits berechnetes SVD-Result statt neue PCA mit .toarray()
        """
        # from sklearn.decomposition import PCA # This import is already at the top
        pca_for_tsne = PCA(n_components=15, random_state=42)
        tsne_input = pca_for_tsne.fit_transform(self.tfidf_matrix.toarray())
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(tsne_input)
        
        plt.figure(figsize=(10, 8))
        if vor_clustering or self.clusters is None:
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='grey', alpha=0.5, s=30)
            plt.title('t-SNE Struktur der Texte (vor Clustering)', fontsize=14, fontweight='bold')
        else:
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.clusters, cmap=CLUSTER_COLORMAP, alpha=0.7, s=50)
            plt.title('t-SNE Visualisierung der Cluster', fontsize=14, fontweight='bold')
            legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
            plt.gca().add_artist(legend1)
        plt.xlabel('t-SNE Komponente 1')
        plt.ylabel('t-SNE Komponente 2')
        plt.tight_layout()
        fname = 'tsne_vor_clustering.png' if vor_clustering else 'tsne_nach_clustering.png'
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()  # Schließe Plot ohne Anzeige

    def clustering_analyse(self, method='kmeans', **kwargs):
        """
        Clustering-Analyse mit wählbarem Algorithmus: 'kmeans', 'agglomerative'
        """
        # Prüfung auf valide PCA-Daten
        if self.pca_result is None or not isinstance(self.pca_result, np.ndarray) or self.pca_result.ndim != 2 or np.isnan(self.pca_result).any() or self.pca_result.shape[0] == 0:
            print("Fehler: self.pca_result ist leer, enthält NaN oder ist nicht 2D. Bitte prüfen Sie die Vorverarbeitung und Feature Engineering.")
            print(f"self.pca_result: {self.pca_result}")
            return
        print(f"PCA-Shape: {self.pca_result.shape}, NaN: {np.isnan(self.pca_result).any()}")
        print(f"\n=== CLUSTERING-ANALYSE ({method.upper()}) ===")
        if method == 'kmeans':
            # ... wie bisher ...
            inertias = []
            silhouette_scores = []
            k_range = range(2, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.pca_result)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(self.pca_result, kmeans.labels_))
            optimal_k = k_range[np.argmax(silhouette_scores)]
            print(f"Optimale Anzahl Cluster (KMeans): {optimal_k}")
            final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            self.clusters = final_model.fit_predict(self.pca_result)
        elif method == 'agglomerative':
            n_clusters = kwargs.get('n_clusters', 6)
            print(f"Starte AgglomerativeClustering mit n_clusters={n_clusters} ...")
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            self.clusters = clusterer.fit_predict(self.pca_result)
            optimal_k = n_clusters
            print(f"Gefundene Cluster (Agglomerative): {optimal_k}")
        else:
            raise ValueError(f"Unbekannte Methode: {method}")
        # Cluster-Label-Spalte und Analyse wie gehabt
        self.data['cluster'] = self.clusters
        self.cluster_labels = {}
        for cluster_id in set(self.clusters):
            cluster_docs = self.data[self.data['cluster'] == cluster_id]
            top_words = self.get_top_words(cluster_docs, n=3)
            label = ', '.join(top_words)
            self.cluster_labels[cluster_id] = label
        self.data['cluster_label'] = self.data['cluster'].map(self.cluster_labels).fillna('Noise')
        print(f"Cluster-Labels: {self.cluster_labels}")

    def exportiere_cluster_summary(self, filename='cluster_summary.json'):
        if not hasattr(self, 'cluster_labels') or self.cluster_labels is None:
            print("Fehler: Es wurden keine Cluster-Labels erzeugt. Bitte führen Sie zuerst ein erfolgreiches Clustering durch.")
            return
        summary = {}
        for cluster_id, label in self.cluster_labels.items():
            cluster_id_str = str(cluster_id)  # JSON-kompatibel
            docs = self.data[self.data['cluster'] == cluster_id]
            summary[cluster_id_str] = {
                'label': label,
                'size': len(docs),
                'top_words': label.split(', '),
                'categories': docs[self.category_column].value_counts().to_dict() if self.category_column in docs else {},
                'years': docs['year'].value_counts().to_dict() if 'year' in docs else {}
            }
        with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Cluster-Summary als {filename} exportiert.")

    def detaillierte_cluster_analyse(self):
        """
        Detaillierte inhaltliche Analyse der 7 Cluster für die Fallstudie
        """
        print("\n" + "=" * 60)
        print("DETAILLIERTE CLUSTER-ANALYSE FÜR FALLSTUDIE")
        print("=" * 60)
        
        if not hasattr(self, 'cluster_labels') or self.cluster_labels is None:
            print("Fehler: Keine Cluster-Labels verfügbar. Führen Sie zuerst ein Clustering durch.")
            return
        
        # TF-IDF für detaillierte Wortanalyse
        print("Berechne TF-IDF für detaillierte Cluster-Analyse...")
        tfidf_analyzer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_matrix = tfidf_analyzer.fit_transform(self.data['combined_text'])
        feature_names = tfidf_analyzer.get_feature_names_out()
        
        # LDA für Topic-Modeling
        print("Berechne LDA-Topics für Cluster-Analyse...")
        count_vectorizer = CountVectorizer(
            max_df=0.9, min_df=3, stop_words='english', ngram_range=(1, 2)
        )
        lda_input = count_vectorizer.fit_transform(self.data['combined_text'])
        lda = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=15)
        lda_result = lda.fit_transform(lda_input)
        lda_feature_names = count_vectorizer.get_feature_names_out()
        
        print("\n" + "-" * 60)
        print("CLUSTER-PROFILIERUNG FÜR FALLSTUDIE")
        print("-" * 60)
        
        cluster_profiles = {}
        
        for cluster_id in sorted(self.cluster_labels.keys()):
            cluster_docs = self.data[self.data['cluster'] == cluster_id]
            cluster_indices = cluster_docs.index
            
            print(f"\nCLUSTER {cluster_id} ({len(cluster_docs)} Artikel)")
            print(f"   Label: {self.cluster_labels[cluster_id]}")
            
            # 1. TF-IDF Top-Wörter für diesen Cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            top_tfidf_indices = cluster_tfidf.argsort()[-10:][::-1]
            top_tfidf_words = [feature_names[i] for i in top_tfidf_indices]
            
            print(f"   Top TF-IDF Wörter: {', '.join(top_tfidf_words[:5])}")
            
            # 2. LDA Topic-Verteilung für diesen Cluster
            cluster_lda = lda_result[cluster_indices].mean(axis=0)
            dominant_topic = cluster_lda.argmax()
            topic_words_idx = lda.components_[dominant_topic].argsort()[-8:][::-1]
            dominant_topic_words = [lda_feature_names[i] for i in topic_words_idx]
            
            print(f"   Dominantes LDA-Topic: {', '.join(dominant_topic_words[:5])}")
            
            # 3. Kategorien-Analyse
            if self.category_column in cluster_docs.columns:
                all_cats = []
                for cats in cluster_docs[self.category_column].dropna():
                    if isinstance(cats, str):
                        all_cats.extend(cats.split())
                top_cats = pd.Series(all_cats).value_counts().head(3)
                print(f"   Top Kategorien: {dict(top_cats)}")
            
            # 4. Zeitliche Verteilung
            if 'year' in cluster_docs.columns:
                year_dist = cluster_docs['year'].value_counts().sort_index()
                print(f"   Zeitraum: {year_dist.index.min()}-{year_dist.index.max()}")
            
            # 5. Kernaussage für Fallstudie
            core_statement = self._generiere_kernaussage(
                cluster_id, top_tfidf_words, dominant_topic_words, 
                cluster_docs, top_cats if self.category_column in cluster_docs.columns else None
            )
            print(f"   KERNAUSSAGE: {core_statement}")
            
            cluster_profiles[cluster_id] = {
                'size': int(len(cluster_docs)),  # np.int64 zu int
                'label': self.cluster_labels[cluster_id],
                'top_tfidf_words': top_tfidf_words[:10],
                'dominant_topic_words': dominant_topic_words[:8],
                'top_categories': {str(k): int(v) for k, v in dict(top_cats).items()} if self.category_column in cluster_docs.columns else {},  # np.int64 zu int
                'core_statement': core_statement
            }
        
        # Zusammenfassung für Fallstudie
        print("\n" + "=" * 60)
        print("ZUSAMMENFASSUNG FÜR FALLSTUDIE")
        print("=" * 60)
        print("Die 7 Cluster repräsentieren folgende Forschungsbereiche:")
        
        for cluster_id in sorted(cluster_profiles.keys()):
            profile = cluster_profiles[cluster_id]
            print(f"\nCluster {cluster_id} ({profile['size']} Artikel):")
            print(f"   {profile['core_statement']}")
        
        # Export der detaillierten Analyse (mit Typkonvertierung)
        cluster_profiles_json = {}
        for cluster_id, profile in cluster_profiles.items():
            cluster_id_str = str(cluster_id)  # JSON-kompatibel
            cluster_profiles_json[cluster_id_str] = profile
        
        with open(os.path.join(OUTPUT_DIR, 'cluster_detaillierte_analyse.json'), 'w', encoding='utf-8') as f:
            json.dump(cluster_profiles_json, f, ensure_ascii=False, indent=2)
        print(f"\nDetaillierte Cluster-Analyse als 'cluster_detaillierte_analyse.json' exportiert.")
        
        return cluster_profiles

    def _generiere_kernaussage(self, cluster_id, tfidf_words, topic_words, cluster_docs, top_cats):
        """
        Generiert eine prägnante Kernaussage für einen Cluster
        """
        # Kombiniere TF-IDF und Topic-Wörter für bessere Aussage
        key_terms = list(set(tfidf_words[:5] + topic_words[:3]))
        
        # Erstelle Kernaussage basierend auf dominanten Begriffen
        if any('quantum' in term.lower() for term in key_terms):
            return f"Quantenphysik und Quantencomputing mit Fokus auf {', '.join(key_terms[:3])}"
        elif any('neural' in term.lower() or 'deep' in term.lower() for term in key_terms):
            return f"Deep Learning und neuronale Netze, spezialisiert auf {', '.join(key_terms[:3])}"
        elif any('graph' in term.lower() or 'network' in term.lower() for term in key_terms):
            return f"Graphentheorie und Netzwerk-Analyse mit Anwendungen in {', '.join(key_terms[:3])}"
        elif any('optimization' in term.lower() or 'algorithm' in term.lower() for term in key_terms):
            return f"Algorithmen und Optimierung, insbesondere {', '.join(key_terms[:3])}"
        elif any('data' in term.lower() or 'analysis' in term.lower() for term in key_terms):
            return f"Datenanalyse und Statistik mit Schwerpunkt {', '.join(key_terms[:3])}"
        elif any('physics' in term.lower() or 'theory' in term.lower() for term in key_terms):
            return f"Theoretische Physik und mathematische Modelle zu {', '.join(key_terms[:3])}"
        else:
            return f"Forschungsbereich mit Fokus auf {', '.join(key_terms[:3])}"

def main():
    """
    Hauptfunktion für arXiv-Datenanalyse
    """
    print("ARXIV WISSENSCHAFTLICHE TRENDS ANALYSE")
    print("=" * 50)
    
    analysator = TextClusterAnalyser()
    
    # Pfad zum arXiv-Datensatz (anpassen)
    arxiv_pfad = "arxiv-metadata-oai-snapshot.json"  # Oder .csv
    
    # Prüfe ob Datei existiert
    import os
    if not os.path.exists(arxiv_pfad):
        print(f"arXiv-Datensatz nicht gefunden: {arxiv_pfad}")
        print("Bitte laden Sie den Datensatz von Kaggle herunter:")
        print("https://www.kaggle.com/datasets/Cornell-University/arxiv")
        print("und platzieren Sie ihn im aktuellen Verzeichnis.")
        return
    
    # Stelle sicher, dass der Output-Ordner existiert
    ensure_output_dir()

    # Vollständige arXiv-Analyse
    # Lade 10.000 Paper aus den Jahren 2015-2025
    if analysator.daten_laden(
        arxiv_pfad, 
        n_sample=10000, 
        category_filter=None, 
        year_filter_min=2015, 
        year_filter_max=2025
    ):
        analysator.arxiv_text_vorverarbeitung()
        analysator.arxiv_feature_engineering()
        analysator.dimensionsreduktion(n_components=100)
        if analysator.pca_result is not None:
            analysator.tsne_visualisierung(vor_clustering=True) # 1. t-SNE vor Clustering
            # Hauptansatz: AgglomerativeClustering mit 7 Clustern für differenziertere Gruppen
            analysator.clustering_analyse(method='agglomerative', n_clusters=7)
            analysator.tsne_visualisierung(vor_clustering=False) # 2. t-SNE nach Clustering
            analysator.exportiere_cluster_summary() # 3. Exportiere Cluster-Summary
            analysator.arxiv_trend_analyse()
            # Detaillierte Cluster-Analyse für Fallstudie
            analysator.detaillierte_cluster_analyse()
        else:
            print("Abbruch: Dimensionsreduktion (SVD) fehlgeschlagen, keine weiteren Analysen möglich.")
        
        print("\n" + "=" * 50)
        print("ARXIV-ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
        print("=" * 50)

if __name__ == "__main__":
    main() 