import time

from types import SimpleNamespace

from src import tarot_reader as tarot_reader_module
from src.tarot_reader import TarotReader


def test_stream_time_prediction():
    tarot_reader = TarotReader()
    cards = ['le diable', "l'homme pendu", 'la mort']

    t1 = time.time()
    gen = tarot_reader.stream_predict(cards)

    first_sentence = next(gen)
    dt = time.time() - t1

    print(f"\nTime to first sentence: {dt} seconds\n")
    assert dt <= 10
    assert any(c in first_sentence for c in ".:?!")

""" def test_time_prediction():
    tarot_reader = TarotReader()
    cards = ['le diable', "l'homme pendu", 'la mort']

    t1 = time.time()
    tarot_reader.predict(cards)
    prediction_time = time.time() - t1
    assert prediction_time <= 3 """

def test_predict_calls_chat(monkeypatch):
    calls_content = {}

    class DummyResponse:
        def __init__(self, content: str):
            self.message = SimpleNamespace(content=content)

    def mockreturn(model_name, messages):
        calls_content['model_name'] = model_name
        calls_content['messages'] = messages
        return DummyResponse('fake response')

    monkeypatch.setattr(tarot_reader_module, 'chat', mockreturn)
    tarot_reader = TarotReader('llama3.2:3b')
    cards = ['le diable', "l'homme pendu", 'la mort']
    tarot_reader.predict(cards)

    assert calls_content['model_name'] == 'llama3.2:3b'
    assert len(calls_content['messages']) == 2
    
    system_msg, user_msg = calls_content['messages']

    assert system_msg['role'] == 'system'
    assert user_msg['role'] == 'user'

    for card in cards:
        assert card in user_msg['content']
    