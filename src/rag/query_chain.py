from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

def load_chain():
    # Load vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)

    # Set up retriever + LLM
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Create retrieval-based QA chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

if __name__ == "__main__":
    chain = load_chain()
    
    print("Ask AcmeTech anything about internal policy. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        result = chain.invoke(query)
        print(f"Bot: {result}")
