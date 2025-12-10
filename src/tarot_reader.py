import textwrap
import re

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
                                             
            Tes réponses doivent toujours être très concises :
              - 2 ou 3 phrases maximum,
              - pas de listes, pas de titres.

            Ne mentionne jamais l'informatique, le code, JSON ou le fait que tu es un modèle.
            Parle comme un humain.
        """).strip()
        self.predict(['fake card'])  # Warm-up the model

    def _build_prompt(self, cards: list[str]):
        cards_desc = ''
        for card in cards:
            cards_desc += f'- {card}\n'

        return textwrap.dedent(f"""
            Voici le tirage (de gauche à droite) :

            {cards_desc}

            Donne une interprétation globale en français qui :
              - reste cohérente avec les significations classiques de ces cartes,
              - relie les cartes entre elles dans une même idée (pas une explication technique),
              - se limite strictement à 2 ou 3 phrases,
              - inclut au moins une allusion explicite aux cartes (par exemple en les nommant
                ou en évoquant clairement leur énergie).

            N'utilise ni listes, ni titres, ni mise en forme spéciale.
            Ne renvoie que le texte de la lecture.
        """).strip()

    def predict(self, cards: list[str]):
        prompt = self._build_prompt(cards)

        response = chat(self.model_name, 
            messages=[
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ])
        return response.message.content

    def stream_predict(self, cards: list[str]):
        prompt = self._build_prompt(cards)

        response = chat(self.model_name, 
            messages=[
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            stream=True,)
        
        print('=== Réponse du modèle ===')
        sentence = ''
        for chunk in response:
            sentence += chunk.message.content
            print(chunk.message.content, end='', flush=True)

            if any(c in sentence for c in '.:?!'):
                yield sentence
                sentence = re.split(r'[\.\:\?\!]', sentence)[-1]
        yield sentence

if __name__ == '__main__':
    from tts import TTS

    reader = TarotReader()
    tts = TTS()
    test_cards = ['Le Bateleur', 'La Papesse', 'L\'Empereur']
    for prediction in reader.stream_predict(test_cards):
        tts.speak(prediction)
    #tts.speak(reader.predict(test_cards))
