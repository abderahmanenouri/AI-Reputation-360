# 🤖 AI Reputation 360 : Analyse de Sentiments & Topic Modeling

Ce projet est un pipeline complet de Data Science permettant d'analyser la réputation de trois grandes marques (**Allociné**, **Leboncoin**, **TCL Lyon**) à partir de **9 532 avis clients**.

## 🚀 Fonctionnalités Clés
- **Scraping de Données** : Collecte automatisée et structurée d'avis clients.
- **Sentiment Analysis (Deep Learning)** : Utilisation du modèle **CamemBERT** pour classer les sentiments (Positif, Négatif, Neutre).
- **Topic Modeling** : Extraction automatique des thématiques majeures (films, cinémas, applications, service client) via **BERTopic**.
- **Indice de Confiance** : Évaluation de la fiabilité des prédictions de l'IA via des scores de probabilité.
- **Business Intelligence** : Dashboard interactif sous **Power BI** pour visualiser l'impact métier.

## 📊 Galerie du Dashboard Power BI
*Les visuels ci-dessous présentent les résultats de l'analyse interactive.*

### Vue d'ensemble (Sentiments & Thématiques)
![Répartition Globale des Sentiments](images/capture1.png)
![Top Thèmes Extraits](images/capture2.png)

### Analyses Temporelles & Tendances
![Évolution des Sentiments par Année](images/capture3.png)
![Focus Tendances 2025-2026](images/capture4.png)

### Analyses par Plateforme (Allociné, Leboncoin, TCL)
![Focus Thèmes Leboncoin](images/capture5.png)
![Répartition Sentiments Marque A](images/capture6.png)
![Évolution Temporelle Spécifique](images/capture7.png)
![Focus Thèmes TCL Lyon](images/capture8.png)
![Répartition Sentiments Marque B](images/capture9.png)

## 🧠 Analyse Technique (Notebook Python)
Le cœur de l'analyse est contenu dans le fichier `Analyse_Reputation_IA.ipynb`.
- **Modèle utilisé** : CamemBERT (modèle spécialisé pour le Français).
- **Validation** : Visualisation de l'indice de certitude du modèle pour garantir la qualité des insights.
- **Outils** : Pandas, Scikit-learn, Transformers et Seaborn.

## 📁 Structure du Projet
- `Analyse_Reputation_IA.ipynb` : Code source de l'analyse IA.
- `data/` : Dossier contenant les datasets CSV et JSON.
- `images/` : Galerie de captures d'écran du dashboard.
- `report_reputation.pbix` : Le rapport source Power BI.

## 🛠️ Installation
```bash
pip install pandas matplotlib seaborn transformers bertopic torch numpy