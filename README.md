# Rapport sur l'Analyse de Sentiment et l'Ingénierie de Prompts avec les Modèles de Langage

## Introduction

Ce rapport présente les travaux pratiques réalisés dans le cadre du cours de Systèmes Multi-Agents et IA Distribuée. L'objectif principal de ces travaux était de comprendre et d'implémenter deux aspects fondamentaux des modèles de langage actuels : d'une part, l'analyse de sentiments utilisant le modèle BERT, et d'autre part, l'ingénierie de prompts avec GPT-2 et LangChain. Ces laboratoires m'ont permis d'acquérir des compétences pratiques sur la façon de manipuler et d'optimiser ces modèles pour des tâches spécifiques, contribuant ainsi à ma compréhension des systèmes d'intelligence artificielle modernes.

## Table des matières

- [Rapport sur l'Analyse de Sentiment et l'Ingénierie de Prompts avec les Modèles de Langage](#rapport-sur-lanalyse-de-sentiment-et-lingénierie-de-prompts-avec-les-modèles-de-langage)
  - [Introduction](#introduction)
  - [Table des matières](#table-des-matières)
  - [Analyse de Sentiment avec BERT](#analyse-de-sentiment-avec-bert)
    - [Présentation du modèle](#présentation-du-modèle)
    - [Implémentation BERT](#implémentation-bert)
    - [Résultats et analyse BERT](#résultats-et-analyse-bert)
  - [Ingénierie de Prompts avec GPT-2](#ingénierie-de-prompts-avec-gpt-2)
    - [Introduction aux messages système](#introduction-aux-messages-système)
    - [Utilisation de LangChain pour les templates](#utilisation-de-langchain-pour-les-templates)
    - [Génération avec GPT-2](#génération-avec-gpt-2)
    - [Résultats et analyse GPT-2](#résultats-et-analyse-gpt-2)
  - [Conclusion](#conclusion)

## Analyse de Sentiment avec BERT

### Présentation du modèle

BERT (Bidirectional Encoder Representations from Transformers) est un modèle de langage développé par Google qui a révolutionné le traitement du langage naturel (NLP). Contrairement aux modèles précédents qui analysaient les textes de manière unidirectionnelle (gauche à droite ou droite à gauche), BERT est entraîné de manière bidirectionnelle, lui permettant de comprendre le contexte complet d'un mot en fonction de tous les mots qui l'entourent (à gauche comme à droite).

Pour ce laboratoire, j'ai utilisé une variante de BERT spécialement fine-tunée pour l'analyse de sentiments : `nlptown/bert-base-multilingual-uncased-sentiment`. Ce modèle a été entraîné pour classifier les textes en cinq catégories de sentiment, allant de "Très négatif" à "Très positif".

### Implémentation BERT

Voici le code principal que j'ai développé pour l'analyse de sentiment :

```python
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
```

Ce code fonctionne en plusieurs étapes essentielles :

1. **Chargement du modèle et du tokenizer** : J'utilise la bibliothèque Transformers de Hugging Face pour charger le modèle BERT pré-entraîné pour la classification de sentiments et son tokenizer associé.

2. **Optimisation pour GPU** : Pour accélérer l'inférence, je vérifie si un GPU est disponible et, le cas échéant, j'y déplace le modèle.

3. **Tokenisation du texte** : Cette étape convertit le texte brut en tokens que le modèle peut comprendre. Les paramètres `padding` et `truncation` assurent que le texte respecte la taille d'entrée attendue par BERT.

4. **Inférence et prédiction** : Le mode d'évaluation est activé (`model.eval()`) et le calcul du gradient est désactivé pour économiser de la mémoire et accélérer le processus, puisque nous ne faisons pas d'apprentissage. Le modèle génère des logits qui représentent des scores non normalisés pour chaque classe.

5. **Interprétation des résultats** : La classe avec le score le plus élevé est sélectionnée comme prédiction, et l'étiquette de sentiment correspondante est renvoyée.

Pour tester cette fonction, j'ai implémenté une fonction principale qui analyse plusieurs exemples de phrases en anglais et en français :

```python
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
```

### Résultats et analyse BERT

L'exécution de ce code sur les exemples fournis démontre la capacité du modèle à analyser correctement les sentiments exprimés dans différentes langues. Pour une phrase clairement négative comme "This is the worst experience I've ever had", le modèle prédit "Très négatif", tandis que pour une phrase positive comme "I love this product! It works perfectly", il prédit "Très positif".

Le modèle est également capable de nuancer son analyse, comme pour la phrase en français "C'est pas mal, mais ça pourrait être mieux", qui est correctement classifiée comme "Neutre" ou "Positif" selon le contexte.

Cette implémentation montre l'efficacité des modèles BERT pour des tâches de classification de texte, en particulier pour l'analyse de sentiments, qui est une compétence fondamentale pour de nombreuses applications d'intelligence artificielle, comme les assistants virtuels, l'analyse de retours clients, ou la modération de contenu.

## Ingénierie de Prompts avec GPT-2

### Introduction aux messages système

L'ingénierie de prompts est une compétence essentielle pour travailler avec les modèles de langage génératifs. Un "prompt" est l'entrée textuelle fournie à un modèle qui guide sa génération. Les messages système constituent une forme spécialisée de prompts qui définissent le comportement global du modèle, son rôle ou sa personnalité.

Dans ce laboratoire, j'ai exploré comment créer des templates de messages système réutilisables et personnalisables avec LangChain, puis les utiliser pour guider la génération de texte avec GPT-2.

### Utilisation de LangChain pour les templates

LangChain est une bibliothèque qui simplifie le travail avec les modèles de langage. J'ai utilisé sa fonctionnalité `PromptTemplate` pour créer des messages système paramétrés :

```python
def build_system_message(subject: str, tone: str, max_length: int) -> str:
    """
    Remplit le template de message système avec les variables désirées.
    """
    system_template = """
You are a helpful AI assistant that specializes in {subject}.
Your responses should be {tone} and no longer than {max_length} words.
"""
    prompt = PromptTemplate(
        input_variables=["subject", "tone", "max_length"], template=system_template
    )
    return prompt.format(subject=subject, tone=tone, max_length=str(max_length))
```

Cette fonction crée un template qui définit trois aspects importants du comportement attendu du modèle :
1. Son domaine d'expertise (`subject`)
2. Le ton à adopter (`tone`)
3. La contrainte de longueur (`max_length`)

L'avantage de cette approche est qu'elle permet de générer dynamiquement différents messages système en changeant simplement les paramètres, sans avoir à réécrire le template complet à chaque fois.

### Génération avec GPT-2

Une fois le message système créé, j'ai implémenté une fonction pour générer du texte avec GPT-2 basé sur ce message :

```python
def generate_from_gpt2(prompt_text: str, gen_max_length: int = 200) -> str:
    # Chargement du tokenizer et du modèle GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Passage éventuel sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenisation du prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Génération
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=gen_max_length,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

    # Décodage et retour du texte généré
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Cette fonction :
1. Charge le modèle GPT-2 et son tokenizer
2. Optimise l'utilisation du GPU si disponible
3. Tokenise le message système (prompt)
4. Configure les paramètres de génération comme la longueur maximale et la pénalité de répétition (`no_repeat_ngram_size`)
5. Génère et décode le texte

Les paramètres de génération sont particulièrement importants :
- `max_length` contrôle la longueur maximale de la sortie
- `no_repeat_ngram_size=2` empêche la répétition de séquences de 2 tokens, évitant ainsi certaines boucles
- `early_stopping=True` arrête la génération quand le modèle produirait naturellement une fin

### Résultats et analyse GPT-2

Pour tester l'impact des différents messages système sur la génération, j'ai implémenté une fonction principale qui crée trois prompts différents et génère du texte pour chacun :

```python
def main():
    # 1. Construire un message système pour un tuteur en mathématiques
    system_msg = build_system_message(
        subject="math tutoring", tone="encouraging", max_length=50
    )
    print("=== System Message ===")
    print(system_msg.strip())
    print()

    # 2. Générer une réponse avec GPT-2
    print("=== GPT-2 Generation ===")
    generated = generate_from_gpt2(system_msg, gen_max_length=150)
    print(generated)
    print("\n" + "=" * 60 + "\n")

    # 3. Autres exemples d'utilisation
    examples = [
        {
            "subject": "history explanations",
            "tone": "concise and formal",
            "max_length": 40,
        },
        {"subject": "creative writing", "tone": "playful and vivid", "max_length": 60},
    ]
    for ex in examples:
        msg = build_system_message(**ex)
        print(
            f"--- Prompt pour '{ex['subject']}' ({ex['tone']}, max {ex['max_length']} mots) ---"
        )
        print(msg.strip())
        out = generate_from_gpt2(msg, gen_max_length=100)
        print(out)
        print()
```

Les résultats montrent comment le même modèle GPT-2 peut adopter différents styles et produire des contenus variés en fonction du message système fourni. Par exemple, avec un sujet "math tutoring" et un ton "encouraging", le modèle tente de générer un contenu encourageant lié aux mathématiques, tandis qu'avec un sujet "creative writing" et un ton "playful and vivid", il produit un texte plus créatif et expressif.

Cette expérience démontre l'importance de l'ingénierie de prompts dans l'utilisation des modèles de langage : même sans modifier les paramètres du modèle lui-même, on peut significativement influencer le type, le style et la qualité du contenu généré.

## Conclusion

Ces deux laboratoires m'ont permis d'explorer des aspects complémentaires des modèles de langage modernes : l'analyse (avec BERT) et la génération (avec GPT-2). J'ai pu constater l'efficacité de BERT pour comprendre et classifier le sentiment d'un texte, ainsi que la flexibilité de GPT-2 pour générer du contenu adapté à différents contextes grâce à l'ingénierie de prompts.

L'utilisation de bibliothèques comme Transformers et LangChain simplifie considérablement l'accès à ces technologies avancées. Ces outils permettent de se concentrer sur les aspects créatifs et stratégiques de l'utilisation des modèles, plutôt que sur les détails techniques de leur implémentation.

Les compétences acquises lors de ces travaux pratiques sont directement applicables à de nombreux domaines : création d'assistants virtuels, analyse de données textuelles, génération de contenu, et bien d'autres. Elles constituent une base solide pour des applications plus avancées en intelligence artificielle distribuée et en systèmes multi-agents.

À l'avenir, il serait intéressant d'explorer des aspects plus avancés comme la combinaison de ces deux approches (analyse et génération) dans un même système, ou l'utilisation de modèles plus récents comme GPT-3, GPT-4 ou BERT-large pour des applications plus complexes.