"""
Lab : Prompt Engineering avec LangChain et GPT-2
Auteur : Otmane TOUHAMI
Date   : 17/04/2025

Ce script montre comment :
 1. Définir un System Message « template »
 2. Le remplir dynamiquement avec LangChain PromptTemplate
 3. Charger un modèle GPT-2 et générer du texte en se basant sur ce message
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain import PromptTemplate


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

    # 3. Autres exemples d’utilisation
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


if __name__ == "__main__":
    main()
