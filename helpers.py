from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import MultiQueryRetriever

from langchain.schema import HumanMessage
import json

import re
from langchain_core.documents import Document

def get_pdf_pages(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    if not pages:
        print("Failed to load PDF.")
        return None
    return pages

def get_documents(doc_type, input):
    assert doc_type in ["url", "file"]
    documents = None
    if doc_type == "url":
        loader = WebBaseLoader()
        documents = loader.load_documents()
    elif doc_type == "file":
        documents = get_pdf_pages(input)
    return documents

def set_text_splitter(chunk_size=None, chunk_overlap=None):
    chunk_size = chunk_size or 4000
    chunk_overlap = chunk_overlap or 200
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

def split_docs(docs: list, text_splitter: RecursiveCharacterTextSplitter):
    documents = text_splitter.split_documents(docs)
    if not documents:
        print('No documents found')
        return None
    print(f'Found {len(documents)} documents')
    return documents

def documents_from_texts(texts: list[str], text_splitter):
    documents = text_splitter.create_documents(texts)
    return documents


def get_vectorstore(embedder, documents=None, local_vector_path=None):
    assert not(documents and local_vector_path), "Must provide either documents or local_vector_path, not both"
    if local_vector_path:
        vector = FAISS.load_local(f"vector-dbs/{local_vector_path}", embedder)
    else:
        print('Loading documents')
        vector = FAISS.from_documents(documents, embedder)
    return vector

def get_vectorstore_from_text(text: str, embedder):
    vector = FAISS.from_texts([text], embedder)
    return vector

def save_vector(vector, name="latest_vector"):
    vector.save_local(f'vector-dbs/{name}')

def get_document_chain(llm, prompt):
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

def get_retrieval_chain(vector, document_chain, advanced = False, llm = None):
    retriever = vector.as_retriever()
    if advanced:
        assert llm, "Must provide llm if advanced is True"
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def get_response(retrieval_chain, query, together = False):
    if together:
        messages = [HumanMessage(content=query)]
        response = retrieval_chain.invoke(messages)
        return response.content

    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

### CONSTANTS
rag_template = ChatPromptTemplate.from_template("""Answer the following question based only on the following context:

<context>
{context}
</context>

Question: {input}""")

def save_response_to_markdown_file(response_string, filename="response.md"):
    with open(filename, "w") as file:
        file.write(response_string)

def save_history_to_markdown_file(messages, filename="history.md"):
    with open(filename, "w") as file:
        for message in messages:
            file.write(f"{message}\n\n")

def read_sample():
    excerpt = ''
    with open('sample.txt', 'r') as file:
        excerpt = file.read()
        assert len(excerpt) > 0, "File is empty"
    return excerpt

def read_settings(config_file = "settings.json") -> dict:
    settings = {}
    with open(config_file, 'r') as file:
        settings = json.load(file)
        assert isinstance(settings, dict), "Settings file is not a dictionary"
    return settings

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    return cleaned_text

# get_rag_template()
# initialize_memory

def clean_docs(docs: list[Document]) -> list[Document]:
    """
    Clean documents, return List[Document]
    """
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    # Replace emojis, weird unicode characters, etc.
    return docs