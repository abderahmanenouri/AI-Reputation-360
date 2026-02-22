"""
AI Reputation 360 - Étape 2 : Pré-traitement des données
Input : data/raw/reviews_all.csv
Output : data/processed/reviews_clean.csv
"""

import pandas as pd
from pathlib import Path
import re

# --- Configuration des chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "reviews_all.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_PATH = DATA_PROCESSED_DIR / "reviews_clean.csv"

def clean_text(text):
    """Nettoyage pour l'IA : on garde la ponctuation (utile pour CamemBERT) mais on enlève la pollution."""
    if not isinstance(text, str):
        return ""
    text = text.lower() # Tout en minuscules
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Supprime les liens web
    text = re.sub(r'\s+', ' ', text).strip() # Supprime les espaces multiples
    return text

def create_sentiment_label(rating):
    """
    Traduit les étoiles en langage Machine Learning :
    0 = Négatif (1-2 étoiles)
    1 = Neutre (3 étoiles)
    2 = Positif (4-5 étoiles)
    """
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

def main():
    print("🧹 Démarrage du nettoyage des données...")

    # 1. Chargement
    try:
        df = pd.read_csv(DATA_RAW)
        print(f"📊 Données initiales : {len(df)} lignes.")
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier introuvable. Lancez le script 01 d'abord.")
        return

    # 2. Suppression des doublons et des lignes vides
    df = df.drop_duplicates(subset=['review_text'])
    df = df.dropna(subset=['review_text', 'rating'])

    # 3. Nettoyage du texte
    print("🧽 Nettoyage du texte...")
    df['review_text_clean'] = df['review_text'].apply(clean_text)

    # 4. Filtre de qualité : suppression des avis < 10 caractères
    df['review_length'] = df['review_text_clean'].apply(len)
    df = df[df['review_length'] >= 10]

    # 5. Création des Labels (Cibles IA)
    print("🏷️ Création des labels de sentiment...")
    df['sentiment_label'] = df['rating'].apply(create_sentiment_label)

    # 6. Extraction des dates pour la Business Intelligence
    print("📅 Formatage des dates...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # 7. Exportation
    df.to_csv(EXPORT_PATH, index=False, encoding='utf-8')

    print("\n" + "="*40)
    print("✅ NETTOYAGE TERMINÉ AVEC SUCCÈS")
    print("="*40)
    print(f"📉 Lignes restantes après nettoyage : {len(df)} avis de haute qualité.")
    print(f"📁 Fichier sauvegardé dans : {EXPORT_PATH}")
    print("\n📊 Répartition des sentiments (Target) :")
    print(df['sentiment_label'].map({0: 'Négatif', 1: 'Neutre', 2: 'Positif'}).value_counts())

if __name__ == "__main__":
    main()