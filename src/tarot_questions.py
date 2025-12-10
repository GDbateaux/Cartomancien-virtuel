import textwrap

from tarot_rag import TarotRag

from ollama import chat


class TarotQuestions:
    """
    Note: The structure and wording of this prompt were refined with the help of
    ChatGPT (OpenAI) to get clearer and more consistent readings.
    """
    def __init__(self, model_name: str = 'llama3.2:3b', n_results = 4):
        self.model_name = model_name
        self.n_results = n_results
        self.tarot_rag = TarotRag()

        self.SYSTEM_PROMPT = textwrap.dedent("""
            Tu es un expert du tarot (histoire, structure du jeu, significations, usage divinatoire, etc.).
            Tu réponds toujours en français, de manière claire, sous la forme d'un court paragraphe.

            Tu dois t'appuyer STRICTEMENT sur les informations fournies dans la question.
            - Si la réponse se trouve dans ces informations, tu la formules clairement.
            - Si l'information n'est pas présente ou est insuffisante, tu dis que tu ne sais pas
              ou que ce n'est pas indiqué.
            - Si la question n'a rien à voir avec le tarot, tu expliques que ce n'est pas lié
              à ton domaine d'expertise.

            Contraintes de style :
            - Ne fais jamais de listes (pas de puces, pas de numérotation).
            - N'utilise pas de titres.
            - Ne parle pas de "contexte", de "documents" ou de "sources" dans ta réponse.
            - Ne commence pas par des expressions comme "Selon le contexte fourni" ou "D'après ces informations".
        """).strip()

    def _build_prompt(self, question):
        retrieved_docs = self.tarot_rag.query_chroma(question, self.n_results)['documents']
        if retrieved_docs:
            context = '\n\n---\n\n'.join(retrieved_docs[0]) 
        else:
            context = "Aucun contexte pertinent n'a pu être trouvé dans la base de connaissances sur le tarot."


        return textwrap.dedent(f"""
            Voici des extraits de textes de référence sur le tarot :

            {context}

            En t'appuyant uniquement sur ces informations, réponds à la question suivante
            en un court paragraphe, sans listes et sans mentionner de contexte, de documents
            ou de sources.

            Question :
            {question}
        """).strip()
    
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
    print(tarot_questions.answer('Combien de pates a une araignée?'))
