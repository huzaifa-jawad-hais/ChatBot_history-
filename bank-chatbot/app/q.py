import pandas as pd
import re
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
# Ensure that the project root is in the system path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from chatbot_utils.chunking.semantic_chunking import semantic_chunking
from chatbot_utils.embeddings.embedding import Ada_002
from chatbot_utils.vectorestore import vectorstore
import question  # Assuming 'question' is a module you have with necessary bot setup

# Load environment variables
load_dotenv()

# Define a function to process questions from an Excel file
def process_questions(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Setup the vector store and other components
    chunker = semantic_chunking()
    embed_model = Ada_002()
    index_name = "bank-chatbot"
    vectordatabase = vectorstore(index_name, embed_model=embed_model, chunk_model=chunker)
    amina_bot = question.bot(vectordatabase)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        question_text = row['Question']

        # Get response from the bot
        response, sim_qs = amina_bot.start_chat(question_text)

        # Update the DataFrame with similar questions extracted from the bot's response
        df.at[index, 'sim1'] = sim_qs[0] if len(sim_qs) > 0 else None
        df.at[index, 'sim2'] = sim_qs[1] if len(sim_qs) > 1 else None
        df.at[index, 'sim3'] = sim_qs[2] if len(sim_qs) > 2 else None

    # Save the updates back to the same Excel file
    df.to_excel(file_path, index=False)

if __name__ == "__main__":
    file_path = 'Book1.xlsx'  # Excel file in the same directory as the script
    process_questions(file_path)
