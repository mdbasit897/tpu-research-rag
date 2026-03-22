import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# CHANGE 1: Import the base HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class ResearchRAG:
    def __init__(self, tpu_engine):
        self.llm = tpu_engine
        self.db_dir = "./data/chroma_db"

        # CHANGE 2: Force it to run purely on the CPU using the BGE specific wrapper
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

        print("Loading Embedding model onto CPU...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model loaded on CPU!")

        # Initialize Vector Store
        self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)