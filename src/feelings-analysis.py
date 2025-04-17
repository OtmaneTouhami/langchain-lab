"""
Lab : Classification de sentiments avec BERT
Auteur : Otmane TOUHAMI
Date   : 17/04/2025

Ce script charge un modèle BERT pré-entraîné pour l'analyse
de sentiments (5 classes : très négatif → très positif), 
tokenise un texte, fait une prédiction et affiche le résultat.
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification

def analyse_sentiment(text: str) -> str:
    """
    Prédit le sentiment d'un texte donné en entrée.
    Retourne l'étiquette de sentiment correspondante.
    """
    # Nom du modèle pré-entraîné (5 classes de sentiment)
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

    # Chargement du tokenizer et du modèle
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Passage éventuel sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenisation du texte
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Prédiction (désactivation du calcul de gradient)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Choix de la classe avec le score (logit) le plus élevé
    predicted_class = torch.argmax(logits, dim=1).item()

    # Étiquettes de sentiment dans l'ordre croissant
    sentiment_labels = [
        "Très négatif",
        "Négatif",
        "Neutre",
        "Positif",
        "Très positif"
    ]

    return sentiment_labels[predicted_class]

def main():
    exemples = [
        "This is the worst experience I've ever had.",
        "I love this product! It works perfectly.",
        "C'est pas mal, mais ça pourrait être mieux.",
        "Absolutely fantastic service and friendly staff!"
    ]

    for text in exemples:
        sentiment = analyse_sentiment(text)
        print(f"Texte : {text}")
        print(f"Sentiment prédit : {sentiment}")
        print("-" * 60)

if __name__ == "__main__":
    main()
