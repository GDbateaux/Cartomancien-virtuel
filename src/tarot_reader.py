import textwrap

from tts import speak

from ollama import chat


class TarotReader:
    """
    Tarot reading generation module.

    Note: The structure and wording of this prompt were refined with the help of
    ChatGPT (OpenAI) to get clearer and more consistent readings.
    """
    def __init__(self, model_name: str = 'llama3.2:3b'): 
        self.model_name = model_name
        self.SYSTEM_PROMPT = textwrap.dedent("""
            Tu es un cartomancien expérimenté. Tu interprètes des tirages de tarot et
            tu réponds en français, avec un ton bienveillant et clair.

            Le consultant ne connaît pas la signification des cartes. Explique-les
            simplement, sans être fataliste. Ne parle pas d'un futur certain, mais de
            tendances, de possibilités et de conseils pratiques.

            Ne mentionne jamais l'informatique, le code, JSON ou le fait que tu es un modèle.
            Parle comme un humain.
        """).strip()

    def build_prompt(self, cards: list[str]):
        cards_desc = ''
        for card in cards:
            cards_desc += f'- {card}\n'

        return textwrap.dedent(f"""
            Voici le tirage (de gauche à droite) :

            {cards_desc}

            Écris une lecture de tarot en français avec :
            - une courte introduction (1-2 phrases),
            - un paragraphe pour chaque carte (sens + rôle dans le tirage),
            - une courte conclusion avec un conseil global.

            Ne renvoie que le texte de la lecture.
        """).strip()

    def predict(self, cards: list[str]):
        prompt = self.build_prompt(cards)

        response = chat(self.model_name, 
            messages=[
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ])

        print('=== Réponse du modèle ===')
        print(response.message.content)
        return response.message.content

if __name__ == '__main__':
    cards = ['le diable', "l'homme pendu", 'la mort']
    reader = TarotReader()
    reading = reader.predict(cards)
    speak(reading)
