import textwrap

from ollama import chat

from src.tarot_rag import TarotRag
from src.utils import load_prompt, project_root, Default


class TarotQuestions:
    """
    Note: The structure and wording of this prompt were refined with the help of
    ChatGPT (OpenAI) to get clearer and more consistent readings.
    """
    def __init__(self, model_name: str = 'llama3.2:3b', n_results = 4):
        self.model_name = model_name
        self.n_results = n_results
        self.tarot_rag = TarotRag()

        self.user_prompt_path = project_root() / 'data' / 'prompts' / 'questions_user.txt'
        system_prompt_path = project_root() / 'data' / 'prompts' / 'questions_system.txt'
        self.SYSTEM_PROMPT = load_prompt(system_prompt_path, textwrap.dedent("""
            Tu es un assistant expert du tarot.

            Règle d'utilisation :
            Tu n'as le droit d'affirmer des faits QUE s'ils sont explicitement présents dans les extraits fournis dans le message utilisateur.
            Tu n'utilises jamais tes connaissances générales.

            Décision :
            - Si la question n'est pas liée au tarot : réponds exactement une seule phrase :
            "Je ne peux pas répondre : cette question ne concerne pas le tarot."
            - Si la question est liée au tarot mais que l'information n'apparaît pas explicitement dans les extraits : réponds exactement une seule phrase :
            "Je ne peux pas répondre de façon fiable : je n'ai pas assez d'informations."
            - Sinon, réponds en français, en 1 à 3 phrases maximum, en un seul paragraphe.

            Contraintes :
            - Jamais de listes, jamais de titres.
            - N'écris jamais de parenthèses, ni ouvrantes ni fermantes.
            - Utilise uniquement des chiffres arabes 0-9, jamais de chiffres romains.
            - Ne mentionne jamais des extraits, textes, documents, sources, contexte, base de connaissances.
        """))

    def _build_prompt(self, question):
        retrieved_docs = self.tarot_rag.query_chroma(question, self.n_results)['documents']
        if retrieved_docs:
            context = '\n\n---\n\n'.join(retrieved_docs[0]) 
        else:
            context = "Aucun contexte pertinent n'a pu être trouvé dans la base de connaissances sur le tarot."

        prompt = load_prompt(self.user_prompt_path, textwrap.dedent("""
            Voici des extraits de textes de référence sur le tarot :
            
            {context}

            Question :
            {question}
        """), ['context', 'question'])

        # Code inspired by https://stackoverflow.com/questions/3536303/python-string-format-suppress-silent-keyerror-indexerror
        d = Default({'context': context, 'question': question})
        return prompt.format_map(d)
    
    def answer(self, question):
        prompt = self._build_prompt(question)

        response = chat(self.model_name, 
            messages=[
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ])
        return response.message.content

if __name__ == '__main__':
    tarot_questions = TarotQuestions()
    print(tarot_questions.answer('Combien il y a de cartes de tarot dans un jeu?'))
    print(tarot_questions.answer('Combien de pattes a une araignée?'))
