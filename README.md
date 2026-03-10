# AI Reputation 360 : Analyse de Sentiments & Topic Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange.svg) ![Deep Learning](https://img.shields.io/badge/Deep_Learning-CamemBERT-red.svg) ![Power BI](https://img.shields.io/badge/Dashboard-Power_BI-yellow.svg)

Ce projet est un pipeline complet de Data Science permettant d'analyser la réputation de trois grandes marques (**Allociné**, **Leboncoin**, **TCL Lyon**) à partir de **9 532 avis clients**.

## Fonctionnalités Clés
- **Scraping de Données** : Collecte automatisée et structurée d'avis clients.
- **Sentiment Analysis (Deep Learning)** : Utilisation du modèle **CamemBERT** pour classer les sentiments (Positif, Négatif, Neutre).
- **Topic Modeling** : Extraction automatique des thématiques majeures (films, cinémas, applications, service client) via **BERTopic**.
- **Indice de Confiance** : Évaluation de la fiabilité des prédictions de l'IA via des scores de probabilité.
- **Business Intelligence** : Dashboard interactif sous **Power BI** pour visualiser l'impact métier.

## Galerie du Dashboard Power BI
*Les visuels ci-dessous présentent les résultats de l'analyse interactive.*

### Vue d'ensemble (Sentiments & Thématiques)
![Répartition Globale des Sentiments](images/Capture%20d'écran%202026-02-22%20230057.png)
![Top Thèmes Extraits](images/Capture%20d'écran%202026-02-22%20230111.png)

### Analyses Temporelles & Tendances
![Évolution des Sentiments par Année](images/Capture%20d'écran%202026-02-22%20230129.png)
![Focus Tendances 2025-2026](images/Capture%20d'écran%202026-02-22%20230151.png)

### Analyses par Plateforme (Allociné, Leboncoin, TCL)
![Focus Thèmes Leboncoin](images/Capture%20d'écran%202026-02-22%20230205.png)
![Répartition Sentiments Marque A](images/Capture%20d'écran%202026-02-22%20230219.png)
![Évolution Temporelle Spécifique](images/Capture%20d'écran%202026-02-22%20230239.png)
![Focus Thèmes TCL Lyon](images/Capture%20d'écran%202026-02-22%20230258.png)
![Répartition Sentiments Marque B](images/Capture%20d'écran%202026-02-22%20230316.png)

## Analyse Technique (Notebook Python)
Le cœur de l'analyse est contenu dans le fichier `Analyse_Reputation_IA.ipynb`.
- **Modèle utilisé** : CamemBERT (modèle spécialisé pour le Français).
- **Validation** : Visualisation de l'indice de certitude du modèle pour garantir la qualité des insights.
- **Outils** : Pandas, Scikit-learn, Transformers et Seaborn.

## Structure du Projet
- `src/` : Scripts Python du pipeline (numérotés `01_` à `06_`).
- `Analyse_Reputation_IA.ipynb` : Notebook d'analyse et de visualisation.
- `data/` : Dossier contenant les datasets CSV (non versionné, voir ci-dessous).
- `images/` : Galerie de captures d'écran du dashboard.
- `results/` : Métriques, prédictions ML et visualisations BERTopic (HTML).

## Données
Les données d'avis clients sont scrapées depuis **Google Play** via le package `google-play-scraper`.
Elles ne sont **pas incluses dans le dépôt** (fichiers trop volumineux).

Pour les générer localement :
```bash
python src/01_load_data.py
```
Cela produira le fichier `data/raw/reviews_all.csv` (~9 500 avis).

## Installation
```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/AI-Reputation-360.git
cd AI-Reputation-360

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Usage
Exécuter le pipeline étape par étape :
```bash
python src/01_load_data.py        # Scraping des avis
python src/02_preprocessing.py    # Nettoyage et labellisation
python src/03_topic_modeling.py   # Extraction des thèmes (BERTopic)
python src/04_ml_baseline.py      # Entraînement ML (TF-IDF + LogReg)
python src/05_dl_finetuning.py    # Fine-tuning CamemBERT (optionnel, GPU)
python src/06_export_powerbi.py   # Export final pour Power BI
```

## Résultats
| Modèle | Accuracy | F1 Score | Temps |
|--------|----------|----------|-------|
| TF-IDF + Logistic Regression | 78.87% | 80.33% | ~1s |
| CamemBERT (fine-tuning, 1000 avis) | *voir notebook* | *voir notebook* | ~15 min (CPU) |