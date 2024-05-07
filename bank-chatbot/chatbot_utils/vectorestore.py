import os
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain import vectorstores
from tqdm.auto import tqdm  # for progress bar
from langchain.embeddings.openai import OpenAIEmbeddings
import time
from typing import List
from .utils import read_pdf_file, read_json_file, read_text_file
import glob
from langchain.retrievers.self_query.base import SelfQueryRetriever
import json
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

class vectorstore:

    def __init__(self, index_name, embed_model, chunk_model, similarity_metric = 'cosine'):
        self.index_name = index_name
        self.embed_model = embed_model
        self.chunk_model = chunk_model
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.similarity_metric = similarity_metric
        self.namespaces = set()
        self.index = None
    
    def initialize_vectorstore(self):
    
        pc = Pinecone(api_key=self.pinecone_api_key)

        spec = ServerlessSpec(
            cloud="aws", region="us-west-2"
        )
        existing_indexes = [
            index_info["name"] for index_info in pc.list_indexes()
        ]

        if self.index_name in existing_indexes:
            self.index = pc.Index(self.index_name)
        else:
            pc.create_index(
                self.index_name,
                dimension=1536,  # dimensionality of ada 002
                metric=self.similarity_metric,
                spec=spec
            )
            time.sleep(3)
            self.index = pc.Index(self.index_name)
        return self.index
    
    def delete_vectorstore(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        existing_indexes = [
            index_info["name"] for index_info in pc.list_indexes()
        ]
        if self.index_name in existing_indexes:
            pc.delete_index(self.index_name)
            time.sleep(3)
            return True
        return False
    
    def process_documents(self, directory : str):
        if not os.path.exists(directory):
            raise ValueError(f"The directory '{directory}' does not exist.")
            return None
        files = []
        for root, _, _ in os.walk(directory):
            pdf_files = glob.glob(os.path.join(root, '*.pdf'))
            json_files = glob.glob(os.path.join(root, '*.json'))
            text_files = glob.glob(os.path.join(root, '*.txt'))
            print("pdf_files: ", pdf_files)
            print("json_file: ", json_files)
            files.extend(pdf_files + json_files+text_files)
        print("files: ",files)
        
        documents = []
        for file in files:
            split_filename = file.split(os.sep)
            print(split_filename)
            split_file_type = split_filename[-1].split(".")
            doc = None
            print("plit_file_type: ",split_file_type)
            if(len(split_file_type) > 1):
                if split_file_type[-1] == "json":
                   doc = read_json_file(file)
                elif split_file_type[-1] == "pdf":
                   doc = read_pdf_file(file)
                elif split_file_type[-1] == "txt":
                    doc = read_text_file(file)
            namespace_name = "index_name"
            # print("splitfilename: ",split_filename)
            if len(split_filename) > 1:
               namespace_name = split_filename[-2]
            #    print("namespace_name: ", namespace_name)
            #    print("filetype: ", split_file_type[-1])
               self.namespaces.add(namespace_name)
            doc_info = {}
            doc_info["name"] = split_filename[-1]
            doc_info["namespace"] =namespace_name
            doc_info["value"] = doc
            doc_info["type"] = split_file_type[-1]
            documents.append(doc_info)
        # Path to the JSON file
        json_file_path = "data.json"

        # Writing data to JSON file
        with open(json_file_path, mode='w') as file:
            json.dump(documents, file, indent=4)
        self.store_namespaces()
        list_of_documents_chunked = []
        for doc in documents:
            doc_chunked = self.chunk_model.chunk_documents(doc)
            list_of_documents_chunked = list_of_documents_chunked + doc_chunked
            print(doc)
            print()
        print("len docs: ", len(documents))
        print("len list of chunked docs: ", len(list_of_documents_chunked))
        vectors_to_upload = self.embed_model.create_embeddings(list_of_documents_chunked)
        self.__upload_documents(vectors_to_upload)

        return

    def store_namespaces(self):
        text_file_path = "namespaces.txt"

        # Writing namespaces to text file
        with open(text_file_path, mode='w') as file:
            for ns in self.namespaces:
                file.write(ns + '\n')

        print("Namespaces have been written to:", text_file_path)

    def __upload_documents(self, vectors: List):
        if self.index:
            self.index.upsert(
            vectors
            )
        else:
            raise ValueError("Error: Index not initialzed and Documents haven't been uploaded")
  
    def __history_of_docs_uploaded(self):
        pass
    def query_namespace(self, query_vector: str, namspaces: list, num_records = 3):
        docsearch = PineconeVectorStore.from_existing_index(self.index_name, self.embed_model.embed_model)

        docsearch.as_retriever(include_metadata = True, earch_kwargs={ "k": num_records,'filter': {
            'namespace': namspaces
            }})
        
        docs = docsearch.similarity_search(query_vector)
        return docs
    def return_vector_store_as_retriever(self, namspaces: list, num_records = 3):

        text_field = "text"
        vectorstore = PineconeVectorStore(  
            self.index, self.embed_model.embed_model, text_field
        )  
    
        ret = vectorstore.as_retriever(search_kwargs={ "k": num_records,'filter': {
            'namespace': { "$in": namspaces}
            }})
        return ret
        