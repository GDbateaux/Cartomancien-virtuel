from ollama import chat


SYSTEM_PROMPT = """
Tu es un cartomancien expérimenté. Tu interprètes des tirages de tarot et
tu réponds en français, avec un ton bienveillant et clair.

Le consultant ne connaît pas la signification des cartes. Explique-les
simplement, sans être fataliste. Ne parle pas d'un futur certain, mais de
tendances, de possibilités et de conseils pratiques.

Ne mentionne jamais l'informatique, le code, JSON ou le fait que tu es un modèle.
Parle comme un humain.
""".strip()

def build_prompt(cards: list[str]):
    """
    Build the tarot reading prompt given a list of card names.

    Note: The structure and wording of this prompt were refined with the help of
    ChatGPT (OpenAI) to get clearer and more consistent readings.
    """
    cards_desc = ''
    for card in cards:
        cards_desc += f'- {card}\n'

    return f"""
Voici le tirage (de gauche à droite) :

{cards_desc}

Écris une lecture de tarot en français avec :
- une courte introduction (1-2 phrases),
- un paragraphe pour chaque carte (sens + rôle dans le tirage),
- une courte conclusion avec un conseil global.

Ne renvoie que le texte de la lecture.
    """.strip()

def predict(cards: list[str]):
    prompt = build_prompt(cards)

    response = chat('llama3.2:3b', 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ])

    print("=== Réponse du modèle ===")
    print(response.message.content)


if __name__ == "__main__":
    cards = ['le diable', "l'homme pendu", 'la mort']
    predict(cards)
