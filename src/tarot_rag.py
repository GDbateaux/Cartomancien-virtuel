import chromadb

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.utils import project_root

class TarotRag:
    """
    RAG retrieval layer backed by ChromaDB.

    What it does:
    - Loads tarot reference text files from data/tarot_data/*.txt
    - Splits them into paragraph-sized chunks
    - Indexes the chunks into a Chroma collection using a multilingual
      embedding model
    - Provides a simple query method to retrieve the top-N most similar chunks
    """

    # Initialize Chroma and optionally rebuild the vector index from the text corpus.
    def __init__(self, rebuild_index = True):
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.DATA_DIR = project_root() / 'data' / 'tarot_data'

        self.CHROMA_DIR = project_root() / 'data' / 'chroma_tarot'
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

    # Load tarot documents and split them into paragraph chunks.
    def _load_chunks(self):
        chunks = []

        for file_path in self.DATA_DIR.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for paragraph in f.read().split("\n\n"):
                    if len(paragraph.strip()) > 0:
                        chunks.append(paragraph.strip())
        return chunks

    # Save the chunks into the Chroma collection.
    def _save_chunks_to_chroma(self):
        chunks = self._load_chunks()

        ids = [f'idx{i}' for i in range(len(chunks))]
        self.collection.add(ids=ids, documents=chunks)
    
    def query_chroma(self, question, n_results = 4):
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results

if __name__ == '__main__':
    tarot_rag = TarotRag()
    result = tarot_rag.query_chroma('Combien il y a de cartes au tarot?', 3)
    print(result)
