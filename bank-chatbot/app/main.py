from fastapi import FastAPI, Form, HTTPException, responses
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from pathlib import Path
import os
from fastapi.responses import JSONResponse
import sys
import re
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

# Correct path for the "Frontends" directory relative to the location of this file
static_files_path = Path(__file__).parent / "Frontends"
app.mount("/Frontends", StaticFiles(directory=static_files_path), name="Frontends")

# Redirect root access to index.html
@app.get("/")
def main():
    return RedirectResponse(url="/Frontends/index.html")

# Define a model for the form input using Pydantic
class QuestionData(BaseModel):
    question: str

# Endpoint to handle form submission
@app.post("/submit/", response_class=HTMLResponse)
async def submit_form(question: str = Form(...)):
    # Placeholder for actual logic to handle the question and generate a response
    answer_html = "<div><p>This is a response to your question.</p></div>"
    return answer_html

# Endpoint to return HTML content from a template
@app.post("/message/")
async def get_answer(question_data: QuestionData):
    question = question_data.question  # Extract the question directly
    chunker = semantic_chunking()
    embed_model = Ada_002()
    index_name = "bank-chatbot"
    vectordatabase = vectorstore(index_name, embed_model=embed_model, chunk_model=chunker)
    amina_bot = bot.bot(vectordatabase)
    response_message = amina_bot.start_chat(question)
    # Define a regular expression pattern to match URLs
    url_pattern = r'https?://\S+'

        # Find all URLs in the response string
    urls = re.findall(url_pattern, response_message)

    # Initialize a list of size 3 initialized to all zeros
    result = [0] * 3

    # Check if each URL matches the pattern and update the result list accordingly
    for index, url in enumerate(urls):
        if "Corporate" in url:
            result[0] = 1
        elif "Gold" in url:
            result[1] = 1
        elif "Platinum" in url:
            result[2] = 1

    print(result," //// ",response_message)
    template_html = ""
    if result[0] == 1:
        template_path = static_files_path / 'output_Corporate.html'
        template_html += template_path.read_text(encoding='utf-8')
    if result[1] == 1:
        template_path = static_files_path / 'output_Gold.html'
        template_html += template_path.read_text(encoding='utf-8')
    if result[2] == 1:
        template_path = static_files_path / 'output_platinium.html'
        template_html += template_path.read_text(encoding='utf-8')

    formatted_response = f'<div class="response-container"><p>{response_message}</p></div>'
    return JSONResponse(content={"answer": formatted_response + template_html})

# Run the app using Uvicorn if this file is executed directly
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Define required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]

    # Check if all required variables are present
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    # Handle the case where some variables are missing
    if missing_vars:
        raise ValueError(f"Error: The following required environment variables are missing from the .env file: {', '.join(missing_vars)}")

    # Ensure that placeholder values are not being used
    if os.getenv("OPENAI_API_KEY") == "add-key" or os.getenv("PINECONE_API_KEY") == "add-key":
        raise ValueError("API keys have not been defined in the .env file. Please update them from 'add-key'.")

    # Start the Uvicorn server
    uvicorn.run(app, host="127.0.0.1", port=5000)
