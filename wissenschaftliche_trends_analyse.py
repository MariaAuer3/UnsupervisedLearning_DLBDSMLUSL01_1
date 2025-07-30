#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wissenschaftliche Trends Analyse
================================

Dieses Skript analysiert wissenschaftliche Artikel aus dem arXiv-Datensatz,
um aktuelle Trends in der Wissenschaft zu identifizieren.

Autor: Maria Auer
Datum: 28.07.2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import os

# Output-Ordner für alle Ausgaben
OUTPUT_DIR = 'Output'

def ensure_output_dir():
    """Stelle sicher, dass der Output-Ordner existiert"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Output-Ordner '{OUTPUT_DIR}' erstellt.")

warnings.filterwarnings('ignore')

# NLTK-Downloads (falls nicht vorhanden)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Deutsche Lokalisierung für Plots
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8')

class WissenschaftlicheTrendsAnalyse:
    """
    Hauptklasse für die Analyse wissenschaftlicher Trends
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.tfidf_matrix = None
        self.pca_result = None
        self.clusters = None
        self.vectorizer = None
        
    def daten_laden(self, dateipfad=None):
        """
        Lädt die wissenschaftlichen Daten
        """
        print("=== 1. DATENEXPLORATION ===")
        
        if dateipfad:
            try:
                self.data = pd.read_csv(dateipfad)
                print(f"Daten erfolgreich geladen: {len(self.data)} Artikel")
            except FileNotFoundError:
                print("Datei nicht gefunden. Erstelle Beispieldaten...")
                self._beispieldaten_erstellen()
        else:
            print("Erstelle Beispieldaten für die Demonstration...")
            self._beispieldaten_erstellen()
        
        self._daten_exploration()
        
    def _beispieldaten_erstellen(self):
        """
        Erstellt Beispieldaten für die Demonstration
        """
        np.random.seed(42)
        
        # Verschiedene wissenschaftliche Bereiche
        bereiche = {
            'Machine Learning': [
                'deep learning neural networks', 'reinforcement learning algorithms',
                'computer vision image processing', 'natural language processing',
                'artificial intelligence applications', 'neural network architectures',
                'machine learning optimization', 'deep reinforcement learning',
                'computer vision applications', 'nlp transformer models'
            ],
            'Data Science': [
                'big data analytics', 'data mining techniques', 'statistical analysis',
                'predictive modeling', 'data visualization', 'business intelligence',
                'data preprocessing', 'feature engineering', 'model evaluation',
                'data science applications'
            ],
            'Computer Science': [
                'algorithm optimization', 'software engineering', 'distributed systems',
                'database management', 'cybersecurity', 'cloud computing',
                'programming languages', 'system architecture', 'network protocols',
                'software development methodologies'
            ],
            'Physics': [
                'quantum mechanics', 'particle physics', 'astrophysics',
                'condensed matter physics', 'theoretical physics', 'experimental physics',
                'quantum computing', 'cosmology', 'nuclear physics',
                'quantum field theory'
            ],
            'Mathematics': [
                'mathematical modeling', 'optimization theory', 'statistical analysis',
                'algebraic geometry', 'number theory', 'differential equations',
                'mathematical physics', 'combinatorics', 'topology',
                'mathematical optimization'
            ]
        }
        
        # Erstelle Beispieldaten
        artikel = []
        for bereich, themen in bereiche.items():
            for i in range(20):  # 20 Artikel pro Bereich
                titel = f"{np.random.choice(themen)} research paper {i+1}"
                abstract = f"This paper presents novel research in {np.random.choice(themen)}. "
                abstract += f"The study focuses on {np.random.choice(themen)} and its applications. "
                abstract += f"Results show significant improvements in {np.random.choice(themen)}."
                
                artikel.append({
                    'id': f"{bereich[:2].upper()}{i+1:03d}",
                    'title': titel,
                    'abstract': abstract,
                    'categories': bereich,
                    'authors': f"Author_{i+1}",
                    'year': np.random.randint(2020, 2025)
                })
        
        self.data = pd.DataFrame(artikel)
        print(f"Beispieldaten erstellt: {len(self.data)} Artikel")
        
    def _daten_exploration(self):
        """
        Führt eine explorative Datenanalyse durch
        """
        print("\n--- Datenexploration ---")
        print(f"Datensatz-Größe: {self.data.shape}")
        print(f"Spalten: {list(self.data.columns)}")
        
        # Grundlegende Statistiken
        print("\nErste 5 Zeilen:")
        print(self.data.head())
        
        print("\nDatentypen:")
        print(self.data.dtypes)
        
        print("\nFehlende Werte:")
        print(self.data.isnull().sum())
        
        # Verteilung der Kategorien
        if 'categories' in self.data.columns:
            print("\nVerteilung der wissenschaftlichen Bereiche:")
            kategorie_verteilung = self.data['categories'].value_counts()
            print(kategorie_verteilung)
            
            # Visualisierung der Kategorieverteilung
            plt.figure(figsize=(12, 6))
            kategorie_verteilung.plot(kind='bar', color='skyblue')
            plt.title('Verteilung der wissenschaftlichen Bereiche', fontsize=14, fontweight='bold')
            plt.xlabel('Wissenschaftlicher Bereich')
            plt.ylabel('Anzahl Artikel')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'kategorie_verteilung.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Jahrgangsverteilung
        if 'year' in self.data.columns:
            print("\nVerteilung nach Jahrgang:")
            jahr_verteilung = self.data['year'].value_counts().sort_index()
            print(jahr_verteilung)
            
            plt.figure(figsize=(10, 6))
            jahr_verteilung.plot(kind='line', marker='o', color='green', linewidth=2)
            plt.title('Publikationen nach Jahrgang', fontsize=14, fontweight='bold')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl Publikationen')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'jahrgang_verteilung.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    def text_vorverarbeitung(self):
        """
        Vorverarbeitung der Textdaten
        """
        print("\n=== 2. TEXT-VORVERARBEITUNG ===")
        
        # Textreinigung
        def text_bereinigen(text):
            if pd.isna(text):
                return ""
            # Kleinbuchstaben
            text = str(text).lower()
            # Sonderzeichen entfernen
            text = re.sub(r'[^\w\s]', '', text)
            # Mehrfache Leerzeichen entfernen
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Tokenisierung und Lemmatisierung
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        def tokenize_and_lemmatize(text):
            tokens = word_tokenize(text)
            # Stopwords entfernen und lemmatisieren
            tokens = [lemmatizer.lemmatize(token) for token in tokens 
                     if token.isalpha() and token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
        
        # Anwendung der Vorverarbeitung
        print("Bereinige Titel...")
        self.data['title_clean'] = self.data['title'].apply(text_bereinigen)
        
        print("Bereinige Abstracts...")
        self.data['abstract_clean'] = self.data['abstract'].apply(text_bereinigen)
        
        print("Tokenisierung und Lemmatisierung...")
        self.data['title_processed'] = self.data['title_clean'].apply(tokenize_and_lemmatize)
        self.data['abstract_processed'] = self.data['abstract_clean'].apply(tokenize_and_lemmatize)
        
        # Kombinierte Textdaten für Analyse
        self.data['combined_text'] = self.data['title_processed'] + ' ' + self.data['abstract_processed']
        
        print("Textvorverarbeitung abgeschlossen!")
        print(f"Beispiel verarbeiteter Text:")
        print(f"Original: {self.data['title'].iloc[0]}")
        print(f"Verarbeitet: {self.data['title_processed'].iloc[0]}")
        
    def feature_engineering(self):
        """
        Feature Engineering für Textdaten
        """
        print("\n=== 3. FEATURE ENGINEERING ===")
        
        # TF-IDF Vektorisierung
        print("TF-IDF Vektorisierung...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Top 1000 Features
            min_df=2,           # Mindestens 2 Dokumente
            max_df=0.95,        # Maximal 95% der Dokumente
            ngram_range=(1, 2)  # Unigramme und Bigramme
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_text'])
        
        print(f"TF-IDF Matrix Dimension: {self.tfidf_matrix.shape}")
        print(f"Anzahl Features: {len(self.vectorizer.get_feature_names_out())}")
        
        # Wichtigste Features anzeigen
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_sums = np.array(self.tfidf_matrix.sum(axis=0)).flatten()
        top_features_idx = tfidf_sums.argsort()[-20:][::-1]
        
        print("\nTop 20 wichtigste Features:")
        for i, idx in enumerate(top_features_idx):
            print(f"{i+1:2d}. {feature_names[idx]:20s} (Score: {tfidf_sums[idx]:.3f})")
        
        # Visualisierung der Feature-Verteilung
        plt.figure(figsize=(12, 6))
        plt.bar(range(20), tfidf_sums[top_features_idx], color='orange')
        plt.title('Top 20 TF-IDF Features', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Index')
        plt.ylabel('TF-IDF Score')
        plt.xticks(range(20), [feature_names[idx] for idx in top_features_idx], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_features.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def dimensionsreduktion(self):
        """
        Dimensionsreduktion mit PCA
        """
        print("\n=== 4. DIMENSIONSREDUKTION ===")
        
        # PCA für Dimensionsreduktion
        print("Führe PCA durch...")
        pca = PCA(n_components=0.95)  # 95% Varianz beibehalten
        self.pca_result = pca.fit_transform(self.tfidf_matrix.toarray())
        
        print(f"Originale Dimensionen: {self.tfidf_matrix.shape[1]}")
        print(f"Reduzierte Dimensionen: {self.pca_result.shape[1]}")
        print(f"Erklärte Varianz: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Varianz-Erklärung visualisieren
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), 'b-', linewidth=2)
        plt.title('Kumulative erklärte Varianz', fontsize=12, fontweight='bold')
        plt.xlabel('Anzahl Komponenten')
        plt.ylabel('Kumulative Varianz')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(1, min(21, len(pca.explained_variance_ratio_))), 
                pca.explained_variance_ratio_[:20], color='lightblue')
        plt.title('Erklärte Varianz pro Komponente', fontsize=12, fontweight='bold')
        plt.xlabel('Komponente')
        plt.ylabel('Erklärte Varianz')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pca_analyse.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # t-SNE für 2D Visualisierung
        print("Führe t-SNE für 2D Visualisierung durch...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(self.pca_result[:, :50])  # Erste 50 Komponenten
        
        # t-SNE Visualisierung
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                            c=pd.Categorical(self.data['categories']).codes, 
                            cmap='tab10', alpha=0.7, s=50)
        plt.title('t-SNE Visualisierung der wissenschaftlichen Artikel', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Komponente 1')
        plt.ylabel('t-SNE Komponente 2')
        
        # Legende
        legend1 = plt.legend(*scatter.legend_elements(), title="Kategorien")
        plt.gca().add_artist(legend1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_visualisierung.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def clustering_analyse(self):
        """
        Clustering-Analyse
        """
        print("\n=== 5. CLUSTERING-ANALYSE ===")
        
        # K-Means Clustering
        print("Führe K-Means Clustering durch...")
        
        # Optimale Anzahl Cluster finden
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.pca_result)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.pca_result, kmeans.labels_))
        
        # Elbow-Methode und Silhouette-Score visualisieren
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, 'bo-', linewidth=2)
        plt.title('Elbow-Methode für K-Means', fontsize=12, fontweight='bold')
        plt.xlabel('Anzahl Cluster (k)')
        plt.ylabel('Inertia')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2)
        plt.title('Silhouette-Score', fontsize=12, fontweight='bold')
        plt.xlabel('Anzahl Cluster (k)')
        plt.ylabel('Silhouette-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_optimierung.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Optimale Anzahl Cluster (basierend auf Silhouette-Score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimale Anzahl Cluster (basierend auf Silhouette-Score): {optimal_k}")
        
        # Finales K-Means Clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.clusters = final_kmeans.fit_predict(self.pca_result)
        
        # Cluster-Ergebnisse zum Datensatz hinzufügen
        self.data['cluster'] = self.clusters
        
        # Cluster-Analyse
        print(f"\nCluster-Verteilung:")
        cluster_verteilung = pd.Series(self.clusters).value_counts().sort_index()
        print(cluster_verteilung)
        
        # Cluster-Charakterisierung
        print(f"\nCluster-Charakterisierung:")
        for cluster_id in range(optimal_k):
            cluster_docs = self.data[self.data['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_docs)} Artikel):")
            
            # Häufigste Kategorien im Cluster
            if 'categories' in cluster_docs.columns:
                top_categories = cluster_docs['categories'].value_counts().head(3)
                print(f"  Top Kategorien: {dict(top_categories)}")
            
            # Häufigste Wörter im Cluster
            cluster_text = ' '.join(cluster_docs['combined_text'])
            words = cluster_text.split()
            word_freq = pd.Series(words).value_counts().head(5)
            print(f"  Häufigste Wörter: {dict(word_freq)}")
        
        # Visualisierung der Cluster
        plt.figure(figsize=(12, 5))
        
        # PCA-basierte Visualisierung
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                            c=self.clusters, cmap='tab10', alpha=0.7, s=50)
        plt.title('K-Means Cluster (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel('PCA Komponente 1')
        plt.ylabel('PCA Komponente 2')
        
        # t-SNE-basierte Visualisierung
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(self.pca_result[:, :50])
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                            c=self.clusters, cmap='tab10', alpha=0.7, s=50)
        plt.title('K-Means Cluster (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE Komponente 1')
        plt.ylabel('t-SNE Komponente 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_ergebnisse.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def trend_analyse(self):
        """
        Trend-Analyse und Interpretation
        """
        print("\n=== 6. TREND-ANALYSE ===")
        
        # Zeitliche Entwicklung der Cluster
        if 'year' in self.data.columns:
            print("Analysiere zeitliche Trends...")
            
            # Cluster-Entwicklung über Zeit
            cluster_time = pd.crosstab(self.data['year'], self.data['cluster'])
            
            plt.figure(figsize=(12, 6))
            cluster_time.plot(kind='bar', stacked=True, colormap='tab10')
            plt.title('Entwicklung der wissenschaftlichen Trends über Zeit', fontsize=14, fontweight='bold')
            plt.xlabel('Jahr')
            plt.ylabel('Anzahl Artikel')
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'trend_entwicklung.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Topic Modeling mit LDA
        print("Führe Topic Modeling durch...")
        lda = LatentDirichletAllocation(
            n_components=5,  # 5 Topics
            random_state=42,
            max_iter=10
        )
        
        # LDA auf TF-IDF Matrix anwenden
        lda_result = lda.fit_transform(self.tfidf_matrix)
        
        # Topics extrahieren
        feature_names = self.vectorizer.get_feature_names_out()
        
        print("\nIdentifizierte wissenschaftliche Topics:")
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        # Topic-Verteilung visualisieren
        topic_distribution = lda_result.mean(axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(topic_distribution) + 1), topic_distribution, color='lightgreen')
        plt.title('Verteilung der wissenschaftlichen Topics', fontsize=14, fontweight='bold')
        plt.xlabel('Topic')
        plt.ylabel('Durchschnittliche Topic-Verteilung')
        plt.xticks(range(1, len(topic_distribution) + 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'topic_verteilung.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def ergebnisse_zusammenfassung(self):
        """
        Zusammenfassung der Ergebnisse und Empfehlungen
        """
        print("\n=== 7. ERGEBNISSE UND EMPFEHLUNGEN ===")
        
        print("Zusammenfassung der wissenschaftlichen Trend-Analyse:")
        print("=" * 60)
        
        # Cluster-Zusammenfassung
        print(f"\n1. CLUSTER-ANALYSE:")
        print(f"   - {len(set(self.clusters))} Haupttrends identifiziert")
        print(f"   - Cluster-Verteilung: {dict(pd.Series(self.clusters).value_counts().sort_index())}")
        
        # Empfehlungen für akademische Kooperationen
        print(f"\n2. EMPFEHLUNGEN FÜR AKADEMISCHE KOOPERATIONEN:")
        
        for cluster_id in sorted(set(self.clusters)):
            cluster_docs = self.data[self.data['cluster'] == cluster_id]
            cluster_size = len(cluster_docs)
            
            print(f"\n   Cluster {cluster_id} ({cluster_size} Artikel):")
            
            # Top Kategorien
            if 'categories' in cluster_docs.columns:
                top_category = cluster_docs['categories'].mode().iloc[0]
                print(f"   - Hauptbereich: {top_category}")
            
            # Charakteristische Wörter
            cluster_text = ' '.join(cluster_docs['combined_text'])
            words = cluster_text.split()
            word_freq = pd.Series(words).value_counts().head(3)
            print(f"   - Charakteristische Themen: {', '.join(word_freq.index)}")
            
            # Kooperationsempfehlung
            print(f"   - Kooperationsempfehlung: Fokus auf {top_category} mit Schwerpunkt auf {word_freq.index[0]}")
        
        print(f"\n3. GESELLSCHAFTLICHE UND ETHISCHE AUSWIRKUNGEN:")
        print(f"   - Die Analyse zeigt aktuelle Forschungsschwerpunkte in der Wissenschaft")
        print(f"   - Identifizierte Trends können bei strategischen Entscheidungen helfen")
        print(f"   - Ethische Überlegungen: Transparenz bei der Datenverwendung gewährleisten")
        print(f"   - Nachhaltigkeit: Fokus auf umweltrelevante Forschungsthemen beachten")
        
        print(f"\n4. TECHNISCHE METHODEN:")
        print(f"   - Textvorverarbeitung: Tokenisierung, Lemmatisierung, Stopword-Entfernung")
        print(f"   - Feature Engineering: TF-IDF Vektorisierung mit N-Grammen")
        print(f"   - Dimensionsreduktion: PCA (95% Varianz beibehalten)")
        print(f"   - Clustering: K-Means mit Silhouette-Score Optimierung")
        print(f"   - Topic Modeling: Latent Dirichlet Allocation")
        
        # Ergebnisse speichern
        self.data.to_csv(os.path.join(OUTPUT_DIR, 'wissenschaftliche_trends_ergebnisse.csv'), index=False, encoding='utf-8')
        print(f"\nErgebnisse wurden in '{os.path.join(OUTPUT_DIR, 'wissenschaftliche_trends_ergebnisse.csv')}' gespeichert.")
        
        print(f"\nAnalyse abgeschlossen! Alle Visualisierungen wurden als PNG-Dateien gespeichert.")

def main():
    """
    Hauptfunktion für die Ausführung der Analyse
    """
    print("WISSENSCHAFTLICHE TRENDS ANALYSE")
    print("=" * 50)
    
    # Stelle sicher, dass der Output-Ordner existiert
    ensure_output_dir()
    
    # Analyse-Objekt erstellen
    analysator = WissenschaftlicheTrendsAnalyse()
    
    # Vollständige Analyse durchführen
    analysator.daten_laden()
    analysator.text_vorverarbeitung()
    analysator.feature_engineering()
    analysator.dimensionsreduktion()
    analysator.clustering_analyse()
    analysator.trend_analyse()
    analysator.ergebnisse_zusammenfassung()
    
    print("\n" + "=" * 50)
    print("ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 50)

if __name__ == "__main__":
    main() 