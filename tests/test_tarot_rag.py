from src.tarot_rag import TarotRag


tarot_rag = TarotRag()


def test_load_chunks():
    chunks = tarot_rag._load_chunks()
    assert tarot_rag.collection.count() == len(chunks)
    assert len(chunks) > 0

def test_query_chroma_returns_expected_structure():
    n_results = 4
    results = tarot_rag.query_chroma('Combien de cartes il y a au tarot', n_results)
    assert 'documents' in results
    assert len(results['documents']) == 1
    assert len(results['documents'][0]) == n_results
    assert all(isinstance(d, str) for d in results['documents'][0])

def test_query_is_deterministic():
    q = 'Combien de cartes il y a au tarot'
    r1 = tarot_rag.query_chroma(q, 4)['documents'][0]
    r2 = tarot_rag.query_chroma(q, 4)['documents'][0]
    assert r1 == r2
