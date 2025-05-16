import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def index_schema(schema_file_path: str, schema_id: str):
    with open(schema_file_path, "r") as f:
        schema_text = f.read()

    # Split schema into chunks
    schema_chunks = [chunk.strip() for chunk in schema_text.split(";") if chunk.strip()]

    chroma_client = chromadb.PersistentClient(path="app/db/chroma_store")
    embedding_function = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    collection = chroma_client.get_or_create_collection(
        name=f"{schema_id}_schema",
        embedding_function=embedding_function
    )

    for i, chunk in enumerate(schema_chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{schema_id}_chunk_{i}"],
            metadatas=[{
                "source": schema_file_path,
                "schema_id": schema_id,
                "example_query": "Example: SELECT * FROM ..."
            }]
        )

    print(f"? Indexed {len(schema_chunks)} chunks into ChromaDB as collection '{schema_id}_schema'.")

if __name__ == "__main__":
    index_schema("app/db/mock_schema.sql", schema_id="mock_schema")