from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from chatbot_utils.chunking.semantic_chunking import semantic_chunking
from chatbot_utils.embeddings.embedding import Ada_002
from chatbot_utils.vectorestore import vectorstore
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
import re
from langchain_core.messages import HumanMessage
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langsmith import Client

import os
from langchain.prompts import PromptTemplate

class BaseMessage:
    def __init__(self, role, content, name=None, additional_kwargs=None):
        self.role = role
        self.content = content
        self.name = name
        self.additional_kwargs = additional_kwargs if additional_kwargs else {}


# Now, when creating messages, you instantiate BaseMessage objects:

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
        # self.account_vectorstore = self.vectordatabase.return_vector_store_as_retriever(["Pricing"],self.num_records_retrieve)
        # self.general = self.vectordatabase.return_vector_store_as_retriever(["Digital_Assets_&_Custody","Terms_&_Conditions"], self.num_records_retrieve)
    
    def start_chat(self, question: str):
        
        template = """You are a chabot designed for AMINA bank and are an expert at banking terminologies. 
        If the client is asking questions regarding accounts or account opening or account pricing schedule, reply with 1.
        if the user is specifically asking information on his bank account details, e.g his credit/debit score, reply with 2
        if the user is asking any other general questions related to banking or anything else reply with 3 
        Only return a single integer as stated.
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(temperature=0)
        

        final_rag_chain = (
            
            { "question": itemgetter("question")}
            |prompt
            | llm
            | StrOutputParser()
        )

        final_ans = final_rag_chain.invoke({"question":question})
        print("Final ans: ", final_ans)
        choice = int(final_ans)
        answer = ""
        if choice == 1:
            choice_template = """You are a chabot designed for AMINA bank that provides the answers to client queries. Your job is to be helpful and courteous when
                respponding to client queries.
                1. If the user uses some greeting without asking any other question, respond with 'Hi my name is Amina. What can I help you with today'
                2. if user asks a question  or uses a greeting, alongside a question related to banking then first greet the client "Hi my name is AMINA"
                and then give an answer to query. Query answering process is explained below.
                If asked about creating an account reply should include corporate,gold,platinum pricing schedules.As given in context.
                At the end of responce  put an string "XXXXXXXXXXXXX" and below it generate 3 short question different to the question asked but related to the same category or topic and whose answers are present in the context provided.
                An example can be if asked about gold pricing schedule give platinum and corporate and one other question related to the question which in this case is about pricing schedules.
                Label 1. 2. 3. before each of questions. 
                
                tone: conversational , corporate

                Now use the following context to answer the question.If you are not aware of the answer from the given context, please reply 
                'I am sorry , I don't have much information regarding your query. Please connect with our representative at 0323425255, and they will be able
                to assist you better'
                
                {context}

                Question: {question}"""
            llm = ChatOpenAI(temperature=0)
            retriever = self.vectordatabase.return_vector_store_as_retriever(["Pricing"],self.num_records_retrieve)
            PROMPT = PromptTemplate(
                template=choice_template, 
                input_variables=["context", "query"],
                
            )

            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs
            )
    
            response = qa(question)
            answer = response["result"]

            print("Final ans: ", answer)
            
        
        elif choice == 2:
            guard_rail = '''Hi, I am sorry but I cannot assist you with personal account information. Please contact our representative on information
            regarding account details at 0323425255
            At the end of responce  put an string "XXXXXXXXXXXXX" and below it generate 3 short question different to the question asked but related to the same category or topic and whose answers are present in the context provided.
                An example can be if asked about gold pricing schedule give platinum and corporate and one other question related to the question which in this case is about pricing schedules.
                Label 1. 2. 3. before each of questions.'''
            answer = guard_rail
            
        elif choice == 3:
            choice_template = """You are a chabot designed for AMINA bank that provides the answers to client queries. Your job is to be helpful and courteous when
                respponding to client queries.
                1. If the user uses some greeting without asking any other question, respond with 'Hi my name is Amina. What can I help you with today'
                2. if user asks a question  or uses a greeting, alongside a question related to banking then first greet the client "Hi my name is AMINA"
                and then give an answer to query. Query answering process is explained below.
                At the end of responce  put an string "XXXXXXXXXXXXX" and below it generate 3 short question different to the question asked but related to the same category or topic and whose answers are present in the context provided.
                An example can be if asked about gold pricing schedule give platinum and corporate and one other question related to the question which in this case is about pricing schedules.
                Label 1. 2. 3. before each of questions.
                
                tone: conversational , corporate

                Now use the following context to answer the question.If you are not aware of the answer from the given context, please reply 
                'I am sorry , I don't have much information regarding your query. Please connect with our representative at 0323425255, and they will be able
                to assist you better'
                
                {context}

                Question: {question}"""
            os.environ["LANGCHAIN_TRACING_V2"]="true"
            os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
            os.environ["LANGCHAIN_API_KEY"]="ls__5d61e4c49e1142f3bde6b632cce55614"
            os.environ["LANGCHAIN_PROJECT"]="pt-clear-ceramics-49"
      
            retriever = self.vectordatabase.return_vector_store_as_retriever(["Digital_Assets_&_Custody", "Terms_&_Conditions"],self.num_records_retrieve)
            PROMPT = PromptTemplate(
                template=choice_template, 
                input_variables=["context", "query"],
                
            )

            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs
            )
    
        response = qa(question)
        answer = response["result"]
        sim_qs = re.findall(r'\d+\.\s(.+)', answer.split('XXXXXXXXXXXXX')[-1].strip())
        print(answer)
        return answer,sim_qs
