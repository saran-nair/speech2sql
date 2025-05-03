from app.rag.schema_embedder import SchemaEmbedder
from app.rag.retriever import SchemaRetriever
from app.llm.sql_generator import LocalLLM

# Step 1: Load and embed schema
schema_path = "app/db/mock_schema.sql"
embedder = SchemaEmbedder(schema_file_path=schema_path)
retriever = SchemaRetriever(embedder)

# Step 2: Input query
query = "What is the total amount of orders placed by customers from Germany?"

# Step 3: Retrieve relevant schema parts
relevant_chunks = retriever.retrieve(query)
schema_context = "\n".join(relevant_chunks)

print("🔍 Retrieved schema context:")
print(schema_context)
print("="*60)

# Step 4: Generate SQL using a local model
llm = LocalLLM(model_name="microsoft/phi-2")  # Change to another model if needed
sql = llm.generate_sql(query, schema_context)

print("💡 Generated SQL query:")
print(sql)