import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class SchemaRetriever:
    def __init__(self, schema_id: str):
        chroma_client = chromadb.PersistentClient(path="app/db/chroma_store")
        embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self.collection = chroma_client.get_collection(
            name=f"{schema_id}_schema",
            embedding_function=embedding_function
        )

    def retrieve(self, query: str, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["documents"][0], results["metadatas"][0]
