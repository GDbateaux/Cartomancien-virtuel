import chromadb

from pathlib import Path

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class TarotRag:
    """
    Retrieval-Augmented Generation (RAG) engine specialized for tarot knowledge.

    This implementation is inspired by:
    - the tutorial: https://medium.com/@arunpatidar26/rag-chromadb-ollama-python-guide-for-beginners-30857499d0a0
    - the official ChromaDB documentation.
    """
    def __init__(self, rebuild_index = True):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.DATA_DIR = Path(__file__).parent.parent / 'data' / 'tarot_data'

        self.CHROMA_DIR = Path(__file__).parent.parent / 'data' / 'chroma_tarot'
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.CHROMA_DIR)

        self.COLLECTION_NAME = 'tarot-wiki'
        if rebuild_index:
            try:
                self.chroma_client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass

            self.collection = self.chroma_client.create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            self._save_chunks_to_chroma()
        else:
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                )
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                )
                self._save_chunks_to_chroma()

    def _load_chunks(self):
        chunks = []

        for file_path in self.DATA_DIR.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for paragraph in f.read().split("\n\n"):
                    if len(paragraph.strip()) > 0:
                        chunks.append(paragraph.strip())
        return chunks

    def _save_chunks_to_chroma(self):
        chunks = self._load_chunks()

        ids = [f'idx{i}' for i in range(len(chunks))]
        self.collection.add(ids=ids, documents=chunks)
    
    def query_chroma(self, question, n_results = 2):
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results

if __name__ == '__main__':
    tarot_rag = TarotRag()
    result = tarot_rag.query_chroma('Combien il y a de cartes au tarot?', 3)
    print(result)
