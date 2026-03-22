import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class ResearchRAG:
    def __init__(self, tpu_engine):
        self.llm = tpu_engine
        self.db_dir = "./data/chroma_db"

        print("Loading ONNX FastEmbed model (Bypassing PyTorch entirely)...")
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Embedding model loaded successfully!")

        self.vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
        )

    def process_and_store_document(self, file_path):
        print(f"Processing document: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Smaller chunks = less context per query = less HBM pressure
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,     # Down from 1000
            chunk_overlap=50,   # Down from 200
        )
        chunks = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks in vector database.")

    def query(self, user_question):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 2}  # Down from 4 — fewer chunks = shorter prompt
        )
        docs = retriever.invoke(user_question)

        # Hard-cap context at 1500 chars (~375 tokens) to stay within budget
        context_parts = []
        total_chars = 0
        for doc in docs:
            if total_chars + len(doc.page_content) > 1500:
                remaining = 1500 - total_chars
                if remaining > 100:
                    context_parts.append(doc.page_content[:remaining])
                break
            context_parts.append(doc.page_content)
            total_chars += len(doc.page_content)

        context = "\n\n".join(context_parts)

        augmented_prompt = f"""Context:
---------------------
{context}
---------------------
Answer this question using only the context above. Be concise.
Question: {user_question}"""

        return self.llm.generate_response(augmented_prompt)