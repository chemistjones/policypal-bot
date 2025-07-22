from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pathlib import Path
from typing import List


def load_documents_from_directory(directory: str) -> List[Document]:
    loaders = []

    for file in Path(directory).glob("*"):
        if file.suffix == ".pdf":
            loaders.append(PyPDFLoader(str(file)))
        elif file.suffix == ".docx":
            loaders.append(Docx2txtLoader(str(file)))
        elif file.suffix == ".md":
            loaders.append(UnstructuredMarkdownLoader(str(file)))
        elif file.suffix == ".txt":
            loaders.append(TextLoader(str(file), encoding="utf-8"))
        elif file.suffix in [".yaml", ".yml"]:
            loaders.append(TextLoader(str(file), encoding="utf-8"))
        else:
            print(f"Unsupported file type: {file.name}")

    all_docs = []
    for loader in loaders:
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(Path(doc.metadata.get("source", "")))
        all_docs.extend(docs)

    return all_docs


def split_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


if __name__ == "__main__":
    from pprint import pprint

    docs = load_documents_from_directory("../../acmetech_docs")
    split_docs = split_documents(docs)

    print(f"Loaded {len(docs)} raw documents")
    print(f"Split into {len(split_docs)} chunks")
    pprint(split_docs[0].dict())
