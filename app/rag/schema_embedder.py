from sentence_transformers import SentenceTransformer
import os

class SchemaEmbedder:
    def __init__(self, schema_file_path: str):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.schema_file_path = schema_file_path
        self.schema_text = self._load_schema()
        self.embeddings = self.model.encode(self._split_schema(), convert_to_tensor=True)

    def _load_schema(self) -> str:
        with open(self.schema_file_path, "r") as f:
            return f.read()

    def _split_schema(self):
        return [stmt.strip() for stmt in self.schema_text.split(";") if stmt.strip()]

    def get_schema_chunks(self):
        return self._split_schema()

    def embed_chunks(self):
        return self.embeddings
