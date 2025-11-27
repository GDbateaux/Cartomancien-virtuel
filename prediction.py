import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(cards: list[str]):
    cards_desc = ''
    for card in cards:
        cards_desc += f'- {card}\n'

    prompt = f"""
You are a tarot reader. You receive the result of a tarot spread as structured data,
and you must generate a tarot reading in French.

The user does not know the meaning of the cards. Explain things clearly and kindly,
without being fatalistic. Avoid talking about "certain future"; talk instead about
tendencies, advice, and possible paths.

Here is the spread (from left to right):

{cards_desc}
Write a reading in French with:
- a short introduction (1-2 sentences),
- one short paragraph for each card explaining its meaning and its role in the spread,
- a short conclusion paragraph with global advice.

Do not mention JSON or technical details in your answer. Just write the reading.
    """
    return prompt

def predict(cards: list[str]):
    prompt = build_prompt(cards)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    print("=== Réponse du modèle ===")
    print(response.output_text)


if __name__ == "__main__":
    cards = ['le diable', "l'homme pendu", 'la mort']
    predict(cards)
