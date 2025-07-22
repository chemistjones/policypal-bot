import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

chain = load_chain()

st.title("PolicyPal Chatbot")
query = st.text_input("Ask AcmeTech policy questions:")
if query:
    answer = chain.invoke(query)
    st.markdown(answer)
