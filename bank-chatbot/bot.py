from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from operator import itemgetter
from dotenv import load_dotenv

import os
from langchain.prompts import PromptTemplate

from operator import itemgetter
import os

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

class bot:
    def __init__(self, vectorstore):
        self.vectordatabase = vectorstore
        self.vectordatabase.initialize_vectorstore()
        self.num_records_retrieve = 3
        self.chat_history = []

    def start_chat(self, question: str):
        self.update_chat_history(question, is_user=True)
        choice = self.process_question(question)
        answer = self.generate_answer(question, choice)
        self.update_chat_history(answer, is_user=False)
        return answer

    def process_question(self, question: str):
        template = """You are a chatbot designed for AMINA bank and are an expert at banking terminologies. 
        If the client is asking questions regarding accounts or account opening or account pricing schedule, reply with 1.
        If the user is specifically asking information on his bank account details, e.g., his credit/debit score, reply with 2.
        If the user is asking any other general questions related to banking or anything else reply with 3 
        Only return a single integer as stated.
        Question: {question}"""
        os.environ["LANGCHAIN_TRACING_V2"]="true"
        os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"]="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"
        os.environ["LANGCHAIN_PROJECT"]="pt-clear-ceramics-49"
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(temperature=0)
        final_rag_chain = (itemgetter("question") | prompt | llm | StrOutputParser())
        return int(final_rag_chain.invoke({"question": question}))
    
    def format_context(self, chat_history):
        return " ".join([f"{type}: {msg}" for msg, type in chat_history])
    
    def generate_answer(self, question, choice):
        if choice == 1:
            choice_template = """You are a chatbot designed for AMINA bank that provides the answers to client queries. Your job is to be helpful and courteous when
            responding to client queries.Above might be the previous history of chat with user.
            1. If the user uses some greeting without asking any other question, respond with 'Hi my name is Amina. What can I help you with today'
            2. if user asks a question or uses a greeting, alongside a question related to banking then 
             give an answer to the query. Query answering process is explained below.
            If asked about creating an account, the reply should include corporate, gold, platinum pricing schedules as given in context.
            
            Tone: conversational, corporate

            Now use the following context to answer the question. If you are not aware of the answer from the given context, please reply 
            'I am sorry, I don't have much information regarding your query. Please connect with our representative at 0323425255, and they will be able
            to assist you better'

            Context: {context}

            Question: {question}"""
            llm = ChatOpenAI(temperature=0)
            retriever = self.vectordatabase.return_vector_store_as_retriever(["Pricing"], self.num_records_retrieve)
            x = self.format_context(self.chat_history)  # Assuming a method to format the history nicely
            y = x + choice_template
            # Create a prompt with the current context and question
            
            PROMPT = PromptTemplate(
                template=y, 
                input_variables=["context", "query"],
            )
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs
            )
            # Send the enriched context and the question to the language model
            response = qa(question)
            answer = response["result"]

            print("Final answer: ", answer)
            return answer
        elif choice == 2:
            guard_rail = '''Hi, I am sorry but I cannot assist you with personal account information. Please contact our representative on information
            regarding account details at 0323425255'''
            answer = guard_rail
        elif choice == 3:
            choice_template = """You are a chabot designed for AMINA bank that provides the answers to client queries. Your job is to be helpful and courteous when
                respponding to client queries.Above might be a history of conversation with user.
                1. If the user uses some greeting without asking any other question, respond with 'Hi my name is Amina. What can I help you with today'
                2. if user asks a question  or uses a greeting, alongside a question related to banking then give an answer to query. Query answering process is explained below.
                
                tone: conversational , corporate

                Now use the following context to answer the question.If you are not aware of the answer from the given context, please reply 
                'I am sorry , I don't have much information regarding your query. Please connect with our representative at 0323425255, and they will be able
                to assist you better'
                
                {context}

                Question: {question}"""
            
            llm = ChatOpenAI(temperature=0)
            retriever = self.vectordatabase.return_vector_store_as_retriever(["Digital_Assets_&_Custody", "Terms_&_Conditions"],self.num_records_retrieve)

            x = self.format_context(self.chat_history)  # Assuming a method to format the history nicely
            y = x + choice_template
            # Create a prompt with the current context and question
            
            PROMPT = PromptTemplate(
                template=y, 
                input_variables=["context", "query"],
            )
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs
            )
            # Send the enriched context and the question to the language model
            response = qa(question)
            answer = response["result"]

            print("Final answer: ", answer)
            return answer
        else:
            return "I'm not sure how to handle this query."

    def update_chat_history(self, message, is_user=True):
        if is_user:
            self.chat_history.append(("User: " + message, "user"))
        else:
            self.chat_history.append(("Bot: " + message, "bot"))
        
        # Maintain only the last two Q&A pairs in the history
        if len(self.chat_history) > 4:
            self.chat_history = self.chat_history[-4:]



