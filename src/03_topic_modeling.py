"""
AI Reputation 360 - Étape 3 : Topic Modeling (BERTopic)
Input : data/processed/reviews_clean.csv
Output : data/processed/reviews_with_themes.csv + Graphiques HTML
"""

import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --- Configuration des chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
DATA_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reviews_with_themes.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🧠 Démarrage de l'Intelligence Artificielle (BERTopic)...")

    # 1. Chargement des données propres
    try:
        df = pd.read_csv(DATA_PROCESSED)
        # On s'assure que tout est bien en format texte
        texts = df['review_text_clean'].astype(str).tolist()
        print(f"📊 {len(texts)} avis chargés pour l'analyse thématique.")
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier introuvable. Lancez le script 02 d'abord.")
        return

    # 2. Création des Embeddings (Transformation du texte en vecteurs mathématiques)
    print("⚙️ Encodage des textes (Cette étape peut prendre quelques minutes...)")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=True)

    # 3. Entraînement de BERTopic
    print("🧩 Clustering et détection des thèmes...")
    # Règle dynamique d'ingénieur : min_topic_size = max(10, len(texts) // 500)
    min_size = max(10, len(texts) // 500)

    topic_model = BERTopic(
        min_topic_size=min_size,
        nr_topics=None # On laisse l'algo trouver le nombre optimal
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    # 4. Réduction des Outliers (Les avis inclassables = -1)
    print("🧹 Traitement des avis inclassables (Outliers)...")
    topics = topic_model.reduce_outliers(texts, topics, strategy="c-tf-idf")

    # 5. Création des labels lisibles (Ex: "livraison | colis | retard")
    print("🏷️ Génération des noms de thèmes...")
    def get_theme_label(topic_id):
        if topic_id == -1:
            return "Outliers (Divers)"
        words = topic_model.get_topic(topic_id)
        if not words:
            return "Inconnu"
        # On prend les 3 mots les plus forts du thème
        return " | ".join([w for w, _ in words[:3]])

    # Ajout des résultats dans notre DataFrame
    df["theme_id"] = topics
    df["theme_name"] = df["theme_id"].apply(get_theme_label)

    # Logs et Monitoring
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    outlier_pct = (topics.count(-1) / len(topics)) * 100
    print(f"\n✅ {n_topics} thèmes uniques découverts !")
    print(f"⚠️ Pourcentage d'Outliers final : {outlier_pct:.2f}%")

    # 6. Sauvegarde des données
    df.to_csv(DATA_OUTPUT, index=False, encoding='utf-8')
    print(f"💾 Base de données mise à jour sauvegardée : {DATA_OUTPUT}")

    # 7. Génération des Visualisations pour Power BI / Portfolio
    print("📈 Génération des graphiques interactifs...")
    topic_model.get_topic_info().to_csv(RESULTS_DIR / "topics_summary.csv", index=False)

    try:
        topic_model.visualize_topics().write_html(str(RESULTS_DIR / "topic_map.html"))
        topic_model.visualize_barchart().write_html(str(RESULTS_DIR / "topic_barchart.html"))
        topic_model.visualize_hierarchy().write_html(str(RESULTS_DIR / "topic_hierarchy.html"))
        print(f"✨ Fichiers HTML générés avec succès dans le dossier : {RESULTS_DIR}")
    except Exception as e:
        print(f"⚠️ Les graphiques n'ont pas pu être générés : {e}")

    print("\n" + "="*40)
    print("✅ TOPIC MODELING TERMINÉ AVEC SUCCÈS")
    print("="*40)

if __name__ == "__main__":
    main()