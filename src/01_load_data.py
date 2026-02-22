"""
AI Reputation 360 - Étape 1 : Collecte 100% Scraping
Sources Google Play : Leboncoin (E-commerce), Allociné (Divertissement), TCL Lyon (Services)
Output : data/raw/reviews_all.csv
"""

import pandas as pd
from google_play_scraper import Sort, reviews
from pathlib import Path
import uuid

# --- Configuration des chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

def scrape_app_reviews(app_id, sector, company, max_reviews=5000):
    """Fonction générique pour scraper n'importe quelle application."""
    print(f"📥 Scraping de l'application {company} ({sector})...")
    try:
        result, _ = reviews(
            app_id,
            lang='fr',
            country='fr',
            sort=Sort.NEWEST,
            count=max_reviews
        )
        df_raw = pd.DataFrame(result)

        # Standardisation des colonnes
        df_clean = pd.DataFrame({
            'review_id': [str(uuid.uuid4()) for _ in range(len(df_raw))],
            'sector': sector,
            'company': company,
            'review_text': df_raw['content'],
            'rating': df_raw['score'],
            'date': pd.to_datetime(df_raw['at'])
        })
        print(f"✅ {len(df_clean)} avis récupérés pour {company}.")
        return df_clean
    except Exception as e:
        print(f"❌ Erreur lors du scraping de {company} : {e}")
        return None

def main():
    print("🚀 Démarrage de la collecte 100% Scraping...")

    # Liste de nos cibles (App ID, Secteur, Nom de l'entreprise)
    apps_to_scrape = [
        ("fr.leboncoin", "E-commerce", "Leboncoin"),
        ("com.allocine.androidapp", "Entertainment", "Allocine_App"),
        ("com.micropole.android.tcl_mobile", "Services", "TCL_Lyon")
    ]

    datasets_list = []

    # Boucle pour scraper chaque application
    for app_id, sector, company in apps_to_scrape:
        df_app = scrape_app_reviews(app_id, sector, company, max_reviews=4000)
        if df_app is not None and not df_app.empty:
            datasets_list.append(df_app)

    # Fusion et Export
    print("\n🔄 Fusion des datasets...")
    if datasets_list:
        df_final = pd.concat(datasets_list, ignore_index=True)
        df_final.dropna(subset=['review_text'], inplace=True)

        export_path = DATA_RAW / "reviews_all.csv"
        df_final.to_csv(export_path, index=False, encoding='utf-8')

        print(f"🎉 SUCCÈS ! {len(df_final)} avis sauvegardés dans :\n{export_path}")
        print("\n📊 Répartition par secteur :")
        print(df_final['sector'].value_counts())
    else:
        print("❌ Échec : Aucun dataset n'a pu être chargé.")

if __name__ == "__main__":
    main()