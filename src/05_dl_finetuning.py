"""
AI Reputation 360 - Étape 5 : Deep Learning (CamemBERT)
Modèle : fine-tuning de 'camembert-base'
Attention : Entraînement optimisé pour tourner sur CPU (limité à 1000 avis)
"""

import pandas as pd
import time
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments
import evaluate
from sklearn.metrics import classification_report

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = PROJECT_ROOT / "data" / "processed" / "reviews_with_themes.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

def compute_metrics(eval_pred):
    """Calcule la précision pendant l'entraînement."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("🧠 Initialisation de CamemBERT (Le Poids Lourd)...")

    # 1. Chargement des données
    df = pd.read_csv(DATA_INPUT).dropna(subset=['review_text_clean', 'sentiment_label'])

    # ✂️ RÉDUCTION DES DONNÉES POUR LE CPU
    df_sample = df.sample(n=1000, random_state=42)
    print(f"📉 [Mode CPU] Entraînement réduit à {len(df_sample)} avis pour limiter le temps de calcul.")

    # 2. Préparation
    X = df_sample['review_text_clean'].astype(str).tolist()
    y = df_sample['sentiment_label'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Tokenization (Traduction du français en langage IA)
    print("🔠 Tokenisation du texte avec CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

    # Conversion au format Dataset
    train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': y_train})
    test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': y_test})

    # 4. Chargement du Modèle
    print("📥 Chargement du cerveau CamemBERT...")
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=3)

    # 5. Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir='./results/model_checkpoints',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        logging_dir='./results/logs',
        logging_steps=10,
        eval_strategy="epoch",           # <--- LA CORRECTION EST ICI
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. ENTRAÎNEMENT
    print("\n🔥 DÉMARRAGE DE L'ENTRAÎNEMENT (Peut prendre 10 à 20 min...)")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # 7. ÉVALUATION FINALE
    print("\n🎯 Évaluation du modèle sur les données de test...")
    predictions_output = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)

    print("\n" + "="*40)
    print("✅ RÉSULTATS DE CAMEMBERT (DEEP LEARNING)")
    print("="*40)
    print(f"⏱️ Temps d'entraînement : {training_time / 60:.2f} minutes")
    print("\n📊 Détail par classe :")
    print(classification_report(y_test, y_pred, target_names=['Négatif (0)', 'Neutre (1)', 'Positif (2)']))

if __name__ == "__main__":
    main()