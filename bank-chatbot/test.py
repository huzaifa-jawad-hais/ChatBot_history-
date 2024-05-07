from chatbot_utils.chunking.semantic_chunking import semantic_chunking
from chatbot_utils.embeddings.embedding import Ada_002
from chatbot_utils.vectorestore import vectorstore
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("PINECONE_API_KEY"))
chunker = semantic_chunking()
embed_model = Ada_002()
index_name = "bank-chatbot"
vectordatabse = vectorstore(index_name, embed_model=embed_model, chunk_model=chunker)
vectordatabse.delete_vectorstore()
vectordatabse.initialize_vectorstore()
vectordatabse.process_documents("AMINA_DOCS/")
# retriever = vectordatabse.return_vector_store_as_retriever()



