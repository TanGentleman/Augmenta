# This file is for loading documents and indexing them to a vectorstore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, NotebookLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from os.path import exists, join
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_community.vectorstores import FAISS
from config import CHROMA_FOLDER, FAISS_FOLDER
from helpers import collection_exists
from models import Embedder

TEST_URL = "https://python.langchain.com/docs/integrations/vectorstores/faiss"
# This is just a descriptive name for the vector db folder
TEST_COLLECTION_NAME = "langchain_faiss_collection"
TEST_QUESTION = "How can I initialize a faiss vector db?"


def loader_from_arxiv_url(url: str) -> list[Document]:
    """
    Load documents from an arXiv URL, return List[Document]
    """
    # Get the number from links like "https://arxiv.org/pdf/2310.02170.pdf"
    valid_starts = ["https://arxiv.org/abs/", "https://arxiv.org/pdf/"]
    assert url.startswith(
        tuple(valid_starts)), "Make sure the link is a valid arXiv URL"
    doc_id = url.split("/")[-1].replace(".pdf", "")
    loader = ArxivLoader(query=doc_id)
    return loader


def loader_from_notebook_url(url: str):
    """
    Return a NotebookLoader from a url to a raw .ipynb file
    """
    assert url[-6:] == ".ipynb", "Make sure the link is a valid notebook"
    import requests

    def reformat_url(url):
        if url.count('/blob/') != 1:
            raise ValueError("Use the raw link to the notebook file")
        # remove /blob from the url
        url = url.replace('/blob', '')
        # append raw to the url
        url = url.replace('github.com', 'raw.githubusercontent.com')
        return url
    # URL to the raw text of the .ipynb file
    if not url.startswith("https://raw.github"):
        url = reformat_url(url)
    assert url.startswith("https://raw.githubuser")
    # Download the notebook as a file
    response = requests.get(url)
    local_file = "temp.ipynb"
    with open(local_file, 'wb') as f:
        f.write(response.content)
    loader = NotebookLoader(
        local_file,
        include_outputs=True,
        max_output_length=20,
        remove_newline=True,
    )
    return loader


def documents_from_url(url: str) -> list[Document]:
    """
    Load documents from a URL, return List[Document]
    """
    def is_link_valid(url):
        return url.startswith("http")
    assert is_link_valid(url), "Make sure the link is valid"
    print("Indexing url:", url)
    loader = None
    if url[-6:] == ".ipynb":
        loader = loader_from_notebook_url(url)
    elif url.startswith("https://arxiv.org/"):
        loader = loader_from_arxiv_url(url)
    else:
        loader = WebBaseLoader(url)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No documents found at url {url}")
    return docs


def documents_from_local_pdf(filepath) -> list[Document]:
    """
    Load a pdf from the "documents" folder
    Returns List[Document]
    """
    filepath = join("documents", filepath)
    assert exists(filepath), "Local PDF file not found"
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    # TODO: Add page number to each document as metadata
    if not docs:
        raise ValueError(f"Failed to read PDF and return documents.")
    return docs


def documents_from_text_file(filepath: str = "sample.txt") -> list[Document]:
    """
    Load a text file from the "documents" folder
    Returns List[Document]
    """
    filepath = join("documents", filepath)
    assert exists(filepath), "Local text file not found"
    loader = TextLoader(filepath)
    docs = loader.load()
    if not docs:
        raise ValueError(f"Local document {filepath} not found")
    return docs


def split_documents(
        docs: list[Document],
        chunk_size=4000,
        chunk_overlap=200) -> list[Document]:
    """
    Split documents into chunks, return List[Document]
    """
    chunked_docs = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)
    # Add the char count as metadata to each document
    for doc in chunked_docs:
        doc.metadata["char_count"] = len(doc.page_content)
    return chunked_docs


def get_chroma_vectorstore(collection_name: str, embedder: Embedder):
    """
    Get a Chroma vectorstore from a collection name, folder is created if it doesn't exist
    """
    filename = join(CHROMA_FOLDER, collection_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=filename,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True),
    )
    return vectorstore


def chroma_vectorstore_from_docs(
        collection_name: str,
        embedder: Embedder,
        docs: list[Document]):
    assert not collection_exists(
        collection_name, "chroma"), "Collection already exists"
    if docs is None:
        raise ValueError(
            "Collection not found. Provide documents to create a new collection")
    print('Indexing documents...')
    vectorstore = get_chroma_vectorstore(collection_name, embedder)
    vectorstore.add_documents(docs)
    return vectorstore


def load_chroma_vectorstore(collection_name, embedder):
    assert collection_exists(
        collection_name, "chroma"), "Collection does not exist"
    vectorstore = get_chroma_vectorstore(collection_name, embedder)
    # assert there are documents present
    return vectorstore


def faiss_vectorstore_from_docs(
        collection_name: str,
        embedder: Embedder,
        docs: list[Document]):
    assert not collection_exists(
        collection_name, "faiss"), "Collection already exists"
    filename = join(FAISS_FOLDER, collection_name)
    if docs is None:
        raise ValueError(
            "Collection not found. Provide documents to create a new collection")
    print('Indexing documents...')
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local(filename)
    return vectorstore


def load_faiss_vectorstore(collection_name: str, embedder: Embedder):
    assert collection_exists(
        collection_name, "faiss"), "Collection does not exist"
    filename = join(FAISS_FOLDER, collection_name)
    vectorstore = FAISS.load_local(
        filename, embedder, allow_dangerous_deserialization=True)
    return vectorstore


def main(
        url=TEST_URL,
        collection_name=TEST_COLLECTION_NAME,
        question=TEST_QUESTION):
    from models import get_openai_embedder_large
    embedder = get_openai_embedder_large()
    docs = documents_from_url(url)
    chunked_docs = split_documents(docs)
    vectorstore = chroma_vectorstore_from_docs(
        collection_name, embedder, chunked_docs)
    output = vectorstore.similarity_search(question, k=1)
    print(output)
    print('\nThe above document was found to be most relevant!')
    return output


if __name__ == "__main__":
    main()
