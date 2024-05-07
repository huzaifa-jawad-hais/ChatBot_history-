from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from chatbot_utils.chunking.semantic_chunking import semantic_chunking
from chatbot_utils.embeddings.embedding import Ada_002
from chatbot_utils.vectorestore import vectorstore
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
import bot

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("PINECONE_API_KEY"))

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
chunker = semantic_chunking()
embed_model = Ada_002()
index_name = "bank-chatbot"
vectordatabse = vectorstore(index_name, embed_model=embed_model, chunk_model=chunker)
amina_bot = bot.bot(vectordatabse)
amina_bot.start_chat("How to open accounts")
# vectordatabse.initialize_vectorstore()
# retriever = vectordatabse.return_vector_store_as_retriever()
# # Retrieve
# from langchain.prompts import ChatPromptTemplate

# # Multi Query: Different Perspectives
# template = """You are an AI language model assistant. Your task is to generate five 
# different versions of the given user question to retrieve relevant documents from a vector 
# database. By generating multiple perspectives on the user question, your goal is to help
# the user overcome some of the limitations of the distance-based similarity search. 
# Provide these alternative questions separated by newlines. Original question: {question}"""
# prompt_perspectives = ChatPromptTemplate.from_template(template)

# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI

# generate_queries = (
#     prompt_perspectives 
#     | ChatOpenAI(temperature=0) 
#     | StrOutputParser() 
#     | (lambda x: x.split("\n"))
# )
# question = "How can I open a corporate account?"
# answer = generate_queries.invoke({"question":question})
# print("generated queries: ", answer)
# question = "Can I open a corporate account and if so what is the cost associated with it?"
# retrieval_chain = generate_queries | retriever.map() | get_unique_union
# docs = retrieval_chain.invoke({"question":question})
# print(len(docs))
# print(docs)
# from operator import itemgetter
# from langchain_openai import ChatOpenAI
# from langchain_core.runnables import RunnablePassthrough

# # RAG
# template = """You are a chabot designed for AMINA banK, Introduce yourself as AMINA and be as helpful
# and courteous as possible. Now use the following context to answer the question.If you are not aware of the answer from the given context, please reply I dont have
# any information regarding this.

# {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# llm = ChatOpenAI(temperature=0)

# final_rag_chain = (
#     {"context": retrieval_chain, 
#      "question": itemgetter("question")} 
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# final_ans = final_rag_chain.invoke({"question":question})
# print("Final ans: ", final_ans)