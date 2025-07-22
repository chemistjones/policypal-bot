from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from loader import load_documents_from_directory, split_documents

from dotenv import load_dotenv
import os

# Load .env file with OPENAI_API_KEY
load_dotenv()

def build_vector_store(doc_dir: str, persist_path: str = "vector_index"):
    # Step 1: Load and split documents
    raw_docs = load_documents_from_directory(doc_dir)
    docs = split_documents(raw_docs)

    # Step 2: Embed and store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    # Step 3: Save to disk
    vector_store.save_local(persist_path)
    print(f"Vector store saved to: {persist_path}")

if __name__ == "__main__":

    build_vector_store("./acmetech_docs")
