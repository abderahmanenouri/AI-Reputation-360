"""
AI Reputation 360 - Étape 4 : Machine Learning Baseline
Input : data/processed/reviews_with_themes.csv
Outputs : results/ml_predictions.csv, results/ml_metrics.csv
Modèle : TF-IDF + Logistic Regression
"""

import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- Configuration des chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = PROJECT_ROOT / "data" / "processed" / "reviews_with_themes.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🤖 Démarrage de l'entraînement Machine Learning (Baseline)...")

    # 1. Chargement des données
    try:
        df = pd.read_csv(DATA_INPUT)
        # Sécurité : on s'assure qu'il n'y a pas de valeurs nulles
        df = df.dropna(subset=['review_text_clean', 'sentiment_label'])
        print(f"📊 {len(df)} avis chargés pour l'entraînement.")
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier introuvable. Lancez le script 03 d'abord.")
        return

    # 2. Préparation des variables X (Texte) et y (Cible)
    X = df['review_text_clean'].astype(str)
    y = df['sentiment_label']

    # Découpage : 80% pour l'entraînement, 20% pour le test (l'examen final de l'IA)
    print("✂️ Découpage des données (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # Lancement du chronomètre
    start_time = time.time()

    # 3. Vectorisation (TF-IDF) : Transformation des mots en fréquences mathématiques
    print("🧮 Vectorisation TF-IDF en cours...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Entraînement du modèle (Logistic Regression)
    print("🧠 Entraînement du modèle Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    # Arrêt du chronomètre
    training_time = time.time() - start_time

    # 5. Évaluation (Le test sur les 20% d'avis qu'il n'a jamais vus)
    print("🎯 Évaluation des performances...")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "="*40)
    print("✅ RÉSULTATS DU MODÈLE BASELINE (ML)")
    print("="*40)
    print(f"⏱️ Temps d'entraînement : {training_time:.2f} secondes")
    print(f"🎯 Précision Globale (Accuracy) : {acc * 100:.2f}%")
    print(f"⚖️ Score F1 (Équilibre) : {f1 * 100:.2f}%")
    print("\n📊 Détail par classe :")
    print(classification_report(y_test, y_pred, target_names=['Négatif (0)', 'Neutre (1)', 'Positif (2)']))

    # 6. Sauvegarde des prédictions (pour fusionner plus tard dans Power BI)
    df_test_results = df.loc[indices_test].copy()
    df_test_results['ml_prediction'] = y_pred
    df_test_results.to_csv(RESULTS_DIR / "ml_predictions.csv", index=False)

    # 7. Sauvegarde des métriques (Pour le Benchmark)
    metrics_df = pd.DataFrame([{
        'Model': 'TF-IDF + Logistic Regression',
        'Accuracy': round(acc, 4),
        'F1_Score': round(f1, 4),
        'Training_Time_sec': round(training_time, 2)
    }])
    metrics_df.to_csv(RESULTS_DIR / "ml_metrics.csv", index=False)
    print(f"💾 Résultats et métriques sauvegardés dans {RESULTS_DIR}")

if __name__ == "__main__":
    main()