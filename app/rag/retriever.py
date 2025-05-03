import torch
from sentence_transformers.util import cos_sim

class SchemaRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.schema_chunks = embedder.get_schema_chunks()
        self.chunk_embeddings = embedder.embed_chunks()

    def retrieve(self, query: str, top_k: int = 2):
        query_embedding = self.embedder.model.encode(query, convert_to_tensor=True)
        scores = cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [self.schema_chunks[i] for i in top_indices]
