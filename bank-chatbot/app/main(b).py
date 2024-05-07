from typing import Union
import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import sys
from pathlib import Path
import os
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads

# Calculate the path to the root of the project (project_root)
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path if not already there
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from chatbot_utils.chunking.semantic_chunking import semantic_chunking
from chatbot_utils.embeddings.embedding import Ada_002
from chatbot_utils.vectorestore import vectorstore
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
import bot

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

from pydantic import BaseModel

# Define a model for the incoming request data
class QuestionData(BaseModel):
    question: str

@app.post("/message/")
def get_answer(question_data: QuestionData):  # Use the Pydantic model for automatic request parsing
    # No need to manually check if "question" is in question_data, as FastAPI handles validation
    
    question = question_data.question  # Extract the question directly
    chunker = semantic_chunking()
    embed_model = Ada_002()
    index_name = "bank-chatbot"
    vectordatabase = vectorstore(index_name, embed_model=embed_model, chunk_model=chunker)
    amina_bot = bot.bot(vectordatabase)
    answer = amina_bot.start_chat(question)
    return {"answer": answer}

# @app.post("/message/")
# def get_answer(question_data):
#     if "question" not in question_data:
#         raise HTTPException(status_code=400, detail="Question key not found in the provided JSON.")
    
#     # Extracting the question from the JSON data
#     question = question_data["question"]
#     chunker = semantic_chunking()
#     embed_model = Ada_002()
#     index_name = "bank-chatbot"
#     vectordatabse = vectorstore(index_name, embed_model=embed_model, chunk_model=chunker)
#     amina_bot = bot.bot(vectordatabse)
#     answer = amina_bot.start_chat(question)
#     return {"answer": answer}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()
    # Define required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    # Check if all required variables are present
    missing_vars = [var for var in required_vars if os.getenv(var) is None]
    if missing_vars:
    # Handle the case where some variables are missing
        raise ValueError(f"Error: The following required environment variables are missing from the .env file {missing_vars}")
    else:
        open_ai_key = os.environ["OPENAI_API_KEY"]
        pinecone_key = os.environ["PINECONE_API_KEY"]
        if open_ai_key == "add-key" and pinecone_key == "add-key":
            raise ValueError("Api keys have not been defined in the .env file")
    uvicorn.run(app, host="127.0.0.1", port=5000)