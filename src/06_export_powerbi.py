"""
AI Reputation 360 - Étape 6 : Préparation Finale pour Power BI
Input : data/processed/reviews_with_themes.csv
Output : data/powerbi_export/dashboard_data.csv
"""

import pandas as pd
from pathlib import Path

# --- Configuration des chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = PROJECT_ROOT / "data" / "processed" / "reviews_with_themes.csv"
POWERBI_DIR = PROJECT_ROOT / "data" / "powerbi_export"
POWERBI_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_PATH = POWERBI_DIR / "dashboard_data.csv"

def main():
    print("🪄 Préparation du fichier final pour Power BI...")

    # 1. Chargement des données contenant les thèmes de l'IA
    try:
        df = pd.read_csv(DATA_INPUT)
    except FileNotFoundError:
        print("❌ Fichier introuvable.")
        return

    # 2. Formatage lisible pour les graphiques
    # On traduit nos labels mathématiques (0, 1, 2) en vrai texte pour Power BI
    sentiment_dict = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    df['Sentiment'] = df['sentiment_label'].map(sentiment_dict)

    # 3. Formatage des dates (Crucial pour les graphiques d'évolution temporelle dans Power BI)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 4. On ne garde que les colonnes utiles pour le Business (On jette la tambouille technique)
    colonnes_a_garder = [
        'review_id', 'date', 'sector', 'company',
        'rating', 'Sentiment', 'theme_name', 'review_text_clean'
    ]
    df_final = df[colonnes_a_garder].copy()

    # 5. Renommage "Propre" des colonnes pour que ça fasse joli dans le Dashboard
    df_final.columns = [
        'ID_Avis', 'Date', 'Secteur', 'Entreprise',
        'Note_Etoiles', 'Sentiment_IA', 'Theme_IA', 'Texte_Avis'
    ]

    # 6. Exportation
    df_final.to_csv(EXPORT_PATH, index=False, encoding='utf-8-sig') # utf-8-sig gère mieux les accents dans Excel/PowerBI

    print("\n" + "="*40)
    print("✅ MASTER DATASET CRÉÉ AVEC SUCCÈS")
    print("="*40)
    print(f"📁 Ton fichier pour Power BI est prêt ici :\n{EXPORT_PATH}")
    print("\n🚀 PROCHAINE ÉTAPE : Ouvre Power BI Desktop, clique sur 'Obtenir les données' -> 'Texte/CSV', et sélectionne ce fichier !")

if __name__ == "__main__":
    main()