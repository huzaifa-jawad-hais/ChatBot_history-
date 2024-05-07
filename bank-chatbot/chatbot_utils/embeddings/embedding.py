from abc import ABC, abstractmethod
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm
from typing import List, Dict

class embedding_strategy(ABC):
    @abstractmethod
    def __init__():
        pass
    @abstractmethod
    def create_embeddings(documents):
        pass

class Ada_002(embedding_strategy):
    
    def __init__(self):
        self.embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    def embed_query(self, query):
        embedding = self.embed_model.embed_documents(query)
        return embedding
   
    def create_embeddings(self, documents : List[Dict[str, str]]):
        batch_size = 100
        vectors_to_upload = []
        id = 0
        for i in tqdm(range(0, len(documents), batch_size)):
            i_end = min(len(documents), i+batch_size)
            # get batch of data
            data = documents[i:i_end]
            batch = [d["value"] for d in data]
            meta = [d["name"] for d in data]
            namespace = [d["namespace"] for d in data]
            # embed text
            embeds = self.embed_model.embed_documents(batch)

            # add to Pinecone without explicit IDs
            for m in range(len(embeds)):
                vector = {}
                vector["id"] = meta[m]+str(id)
                vector["values"] = embeds[m]
                filename = meta[m].split(".")
                print("file_name: ",filename)
                if(filename[-1] != "json"):
                    vector["metadata"] = {"text":batch[m], "namespace": namespace[m], "filename":meta[m]}
                vectors_to_upload.append(vector)
                id+=1
        return vectors_to_upload
        