# This 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from helpers import clean_docs, database_exists, format_docs
from constants import get_music_template, get_rag_template, get_summary_template, get_eval_template
from config import EXPERIMENTAL_UNSTRUCTURED

# This file is for loading documents and indexing them to a vectorstore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, NotebookLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from os.path import exists, join
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_community.vectorstores import FAISS
from config import EXPERIMENTAL_UNSTRUCTURED, METADATA_MAP
from constants import CHROMA_FOLDER, FAISS_FOLDER
from helpers import database_exists

# if EXPERIMENTAL_UNSTRUCTURED:
#     try:
#         from unstructured.cleaners.core import clean_extra_whitespace
#         from langchain_community.document_loaders import UnstructuredPDFLoader
#     except ImportError:
#         print("ImportError: Unstructured functions in embed.py not be accessible")
#         raise ValueError("Set EXPERIMENTAL_UNSTRUCTURED to False to continue")


def loader_from_arxiv_url(url: str) -> ArxivLoader:
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


# def loader_from_file_unstructured(filepath: str):
#     """
#     Load documents from any file, return List[Document]
#     """
#     assert filepath.endswith(
#         ".pdf"), "Unstructured temporarily only supports PDFs"
#     LOAD_ELEMENTS = False
#     if LOAD_ELEMENTS:
#         element_loader = UnstructuredPDFLoader(
#             filepath,
#             mode="elements",
#             post_processors=[clean_extra_whitespace])
#         loader = element_loader
#     else:
#         loader = UnstructuredPDFLoader(
#             filepath,
#             post_processors=[clean_extra_whitespace])
#     return loader


def loader_from_notebook_url(url: str) -> NotebookLoader:
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


# def documents_from_arbitrary_file(filepath: str) -> list[Document]:
#     """
#     Load a pdf from the "documents" folder
#     Returns List[Document]
#     """
#     filepath = join("documents", filepath)
#     assert exists(filepath), "Local file not found"
#     element_loader = loader_from_file_unstructured(filepath)
#     docs = element_loader.load()
#     if not docs:
#         raise ValueError(f"Did not get docs from local document at {filepath}")
#     return docs


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
    excerpt_count = 0
    for doc in chunked_docs:
        source = doc.metadata.get("source")
        if METADATA_MAP and source in METADATA_MAP:
            doc.metadata["topic"] = METADATA_MAP[source]
        excerpt_count += 1
        doc.metadata["index"] = excerpt_count
        char_count = len(doc.page_content)
        doc.metadata["char_count"] = char_count
        if char_count > 20000:  # This number is arbitrary
            print("Warning: Document is really long, consider splitting it further.")
    return chunked_docs


def get_chroma_vectorstore(
        collection_name: str,
        embedder,
        exists=False):
    """
    Get a Chroma vectorstore from a collection name, folder is created if it doesn't exist
    """
    if exists is False:
        assert not database_exists(
            collection_name, "chroma"), "Collection does not exist"
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


def get_chroma_vectorstore_from_docs(
        collection_name: str,
        embedder,
        docs: list[Document]):
    assert not database_exists(
        collection_name, "chroma"), "Collection already exists"
    if docs is None:
        raise ValueError(
            "Collection not found. Provide documents to create a new collection")
    print('Indexing documents...')
    vectorstore = get_chroma_vectorstore(
        collection_name, embedder, exists=False)
    vectorstore.add_documents(docs)
    return vectorstore


def load_existing_chroma_vectorstore(collection_name, embedder):
    assert database_exists(
        collection_name, "chroma"), "Collection does not exist"
    vectorstore = get_chroma_vectorstore(
        collection_name, embedder, exists=True)
    # assert there are documents present
    return vectorstore


def get_faiss_vectorstore_from_docs(
        collection_name: str,
        embedder,
        docs: list[Document]):
    assert not database_exists(
        collection_name, "faiss"), "Collection already exists"
    assert docs, "No documents found"
    filename = join(FAISS_FOLDER, collection_name)
    if docs is None:
        raise ValueError(
            "Collection not found. Provide documents to create a new collection")
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local(filename)
    print('Vector database saved locally')
    return vectorstore


def load_existing_faiss_vectorstore(collection_name: str, embedder):
    assert database_exists(
        collection_name, "faiss"), "Collection does not exist"
    filename = join(FAISS_FOLDER, collection_name)
    vectorstore = FAISS.load_local(
        filename, embedder, allow_dangerous_deserialization=True)
    return vectorstore


def get_summary_chain(llm):
    """
    Returns a chain for summarization only.
    Can be invoked, like `chain.invoke("Excerpt of long reading:...")` to get a response.
    """
    chain = (
        {"excerpt": lambda x: x}
        | get_summary_template()
        | llm
        | StrOutputParser()
    )
    return chain


def eval_output_handler(output):
    """
    A JSON output parser that returns the response object.
    """
    def is_output_valid(output):
        """
        Checks if the output is valid.
        """
        structure_passed = bool(
            "index" in output and "meetsCriteria" in output)
        if structure_passed:
            print("Meets criteria:", output["meetsCriteria"])
        return structure_passed
    try:
        if isinstance(output, str):
            output_string = output
        else:
            output_string = output.content
        response_object = JsonOutputParser().parse(output_string)
        if not is_output_valid(response_object):
            raise ValueError("JSON output does not contain the required keys")
    except BaseException:
        raise ValueError("Eval chain did not return valid JSON")
    # print("Output:", output)
    # Perform JSON validation here
    # This function can also redirect to the next step in the pipeline
    return response_object


def get_eval_chain(llm):
    """
    Returns a chain for evaluating a given excerpt.

    This chain will NEED to be passed a dictionary with keys excerpt and criteria, not a string.
    """
    eval_prompt_template = get_eval_template()
    chain = eval_prompt_template | llm | eval_output_handler
    return chain


def music_output_handler(output):
    """
    A parser that returns a list of dictionaries.

    Each dictionary should have keys "title", "artist", and "album".
    """
    def is_output_valid(output):
        """
        Checks if the output is valid.
        """
        for item in output:
            if not all(key in item for key in ["title", "artist", "album"]):
                print(f"At least one song does not contain the required keys")
                return False
        return True

    try:
        if isinstance(output, str):
            output_string = output
        else:
            output_string = output.content
        print("Output:" + output_string)
        response_object = JsonOutputParser().parse(output_string)
        if not is_output_valid(response_object):
            raise ValueError("JSON output is not valid")
    except ValueError:
        print("Music chain did not return valid JSON")
        return None
        # raise SystemExit("Music chain did not return valid JSON")
    return response_object


def get_music_chain(llm, few_shot_examples=None):
    """
    Returns a chain for the music pipeline.
    """
    music_prompt_template = get_music_template()
    few_shot_string = ""
    if few_shot_examples:
        assert isinstance(few_shot_examples, list)
        for example in few_shot_examples:
            assert "input" in example and "output" in example
            few_shot_string += f"input: {example['input']}\noutput: {example['output']}\n\n"
    chain = (
        {"input": RunnablePassthrough(), "few_shot_examples": lambda x: few_shot_string}
        | music_prompt_template
        | llm
        | music_output_handler
    )
    return chain


def get_rag_chain(
        retriever,
        llm,
        format_fn=format_docs,
        system_message: str | None = None):
    """
    Returns a chain for the RAG pipeline.

    Can be invoked with a question, like `chain.invoke("How do I do x task using this framework?")` to get a response.
    Inputs:
    - retriever (contains vectorstore with documents) and llm
    - format_fn (callable): A function that takes a list of Document objects and returns a string.
    """
    # Get prompt template
    rag_prompt_template = get_rag_template(system_message)
    chain = (
        {"context": retriever | format_fn, "question": RunnablePassthrough()}
        | rag_prompt_template
        | llm
    )
    return chain


def input_to_docs(input: str) -> list[Document]:
    """
    Converts the input to a list of documents.
    """
    if input.startswith("http"):
        docs = documents_from_url(input)

    else:
        if EXPERIMENTAL_UNSTRUCTURED:
            assert False, "Experimental unstructured not supported"
            # docs = documents_from_arbitrary_file(input)
        else:
            if input.endswith(".txt"):
                docs = documents_from_text_file(input)
            else:
                assert input.endswith(
                    ".pdf"), "Invalid file type. Try enabling EXPERIMENTAL_UNSTRUCTURED."
                docs = documents_from_local_pdf(input)
    if not docs:
        print("No documents found")
        raise ValueError("No documents found")
    for doc in docs:
        doc.metadata["source"] = input
    docs = clean_docs(docs)
    return docs


def vectorstore_from_inputs(
        inputs: list[str],
        method: str,
        embedder,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int = 200):
    """
    Args:
    - inputs: list of strings (urls or filepaths to .pdf or .txt files)
    - method: "chroma" or "faiss"
    - embedder: the embedder to use
    - collection_name: the name of the collection to create/load
    - chunk_size: the size in characters of the chunks to split the documents into
    - chunk_overlap: the overlap between chunks
    Returns:
    - vectorstore: the vectorstore created from the inputs (FAISS or Chroma)
    """
    method = method.lower()
    assert method in ["chroma", "faiss"], "Invalid method"
    assert inputs, "No inputs provided"
    vectorstore = None
    if database_exists(collection_name, method):
        print(f"Collection {collection_name} exists, now loading")
        if method == "chroma":
            vectorstore = load_existing_chroma_vectorstore(
                collection_name, embedder)
        elif method == "faiss":
            vectorstore = load_existing_faiss_vectorstore(
                collection_name, embedder)
        assert vectorstore is not None, "Collection exists but not loaded properly"
        return vectorstore
    for i in range(len(inputs)):
        # In the future this can be parallelized
        input = inputs[i]
        if not input:
            print(f'Input {i} is empty, skipping')
            continue
        docs = input_to_docs(input)
        if not docs:
            print(f"No documents found for input {input}")
            continue
        docs = split_documents(docs, chunk_size, chunk_overlap)
        if i == 0:
            if method == "chroma":
                vectorstore = get_chroma_vectorstore_from_docs(
                    collection_name, embedder, docs)
            elif method == "faiss":
                vectorstore = get_faiss_vectorstore_from_docs(
                    collection_name, embedder, docs)
        else:
            assert vectorstore is not None, "Vectorstore not initialized"
            # This method should work for both Chroma and FAISS
            vectorstore.add_documents(docs)
    assert vectorstore is not None, "Vectorstore not initialized. Provide valid inputs."
    return vectorstore
