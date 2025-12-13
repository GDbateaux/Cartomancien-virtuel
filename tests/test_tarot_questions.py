from src.tarot_questions import TarotQuestions


tarot_rag = TarotQuestions()

def test_off_topic_returns():
    response = tarot_rag.answer('Combien de pattes a une araignée ?').strip()
    assert response == "Je ne peux pas répondre : cette question ne concerne pas le tarot."


def test_tarot_but_unknown_returns():
    response = tarot_rag.answer("Quelle est la couleur officielle du dos des cartes du tarot de Marseille édition X de 1930 ?").strip()
    assert response == "Je ne peux pas répondre de façon fiable : je n'ai pas assez d'informations."

def test_tarot_answerable():
    response = tarot_rag.answer('Combien y a-t-il de cartes dans un jeu de tarot ?')
    assert response not in [
        'Je ne peux pas répondre : cette question ne concerne pas le tarot.',
        "Je ne peux pas répondre de façon fiable : je n'ai pas assez d'informations."
    ]
    assert '(' not in response and ')' not in response
