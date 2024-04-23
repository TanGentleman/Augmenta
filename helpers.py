from json import JSONDecodeError, load as json_load
from json import dump as json_dump
from re import sub as re_sub
from uuid import uuid4
from langchain_core.documents import Document
from os.path import exists as path_exists, join as path_join
from datetime import datetime

from config import VECTOR_DB_SUFFIX

# TODO:
# Add a ROOT_FILEPATH constant and use os.join to create the filepaths


def save_response_to_markdown_file(
        response_string: str,
        filename="response.md"):
    """
    Save a response string to a markdown file.

    Parameters:
    - response_string (str): The response string to be saved.
    - filename (str, optional): The name of the file. Defaults to "response.md".
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response_string)


def save_history_to_markdown_file(messages: list[str], filename="history.md"):
    """
    Save a list of messages to a markdown file.

    Parameters:
    - messages (list[str]): The list of messages to be saved.
    - filename (str, optional): The name of the file. Defaults to "history.md".
    """
    with open(filename, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(f"{message}\n\n")


def read_sample():
    """
    Read the sample.txt file and return the excerpt.
    """
    excerpt = ''
    with open('sample.txt', 'r') as file:
        excerpt = file.read()
        assert len(excerpt) > 0, "File is empty"
    return excerpt


def read_settings(config_file="settings.json") -> dict:
    """
    Read the settings file and return the settings as a dictionary.
    """
    settings = {}
    with open(config_file, 'r') as file:
        try:
            settings = json_load(file)
        except JSONDecodeError:
            print("Error reading settings file")
            exit(1)

        assert isinstance(settings, dict), "Settings file is not a dictionary"
    return settings


def clean_text(text):
    """
    Clean text, return str
    """
    cleaned_text = re_sub(r'\s+', ' ', text)
    cleaned_text = re_sub(r'[^\w\s]', '', cleaned_text)
    return cleaned_text


def clean_docs(docs: list[Document]) -> list[Document]:
    """
    Clean documents, return List[Document]
    """
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    # Replace emojis, weird unicode characters, etc.
    return docs

def process_docs(docs: list[Document]) -> list[Document]:
    """
    Process the documents and extract relevant content. This is set for Anthropic prompt library scraping.

    Parameters:
    - docs (list[Document]): A list of Document objects.

    Returns:
    - list[Document]: The processed list of Document objects.
    """
    left_string = "try it for yourself"
    right_string = "Example output"
    for doc in docs:
        start_index = doc.page_content.find(left_string)
        end_index = doc.page_content.find(right_string)
        if start_index != -1 and end_index != -1:
            extracted_content = doc.page_content[start_index + len(left_string):end_index]
            doc.page_content = extracted_content
            print('trimmed content')
        else:
            print("Role/example not found in document")
    return docs

def format_docs(docs: list[Document], save_excerpts=True, process_docs_fn=None) -> str:
    """
    Formats the list of documents into a single string as context to be passed to LLM.

    Parameters:
    - docs (list[Document]): A list of Document objects.
    - save_excerpts (bool, optional): Whether to save the excerpts to a markdown file. Defaults to True.
    """
    summaries = []
    # save documents here to excerpts.md
    context_string = ""
    if process_docs_fn:
        docs = process_docs_fn(docs)
    for doc in docs:
        # check if "Summary" is a key in the dict
        # print(doc.metadata.keys())
        # This block is the default for arxiv papers, but I can add these to
        # other docs
        if "Summary" in doc.metadata and "Title" in doc.metadata:
            summary = doc.metadata["Summary"]
            summary_string = f"Summary for {doc.metadata['Title']}:\n{summary}"
            if summary not in summaries:
                summaries.append(summary)
                context_string += summary_string + "\n\n"
        context_string += doc.page_content + "\n"
        # add source
        source_string = doc.metadata["source"] if "source" in doc.metadata else "Unknown source"
        # add page string if available
        if "page" in doc.metadata:
            source_string += f" (Page {doc.metadata['page']})"
        context_string += f"Source: {source_string}\n\n"
    if save_excerpts:
        with open("excerpts.md", "w") as f:
            f.write(f"Context:\n{context_string}")
    context_string = context_string.strip()
    return context_string


def collection_exists(collection_name: str, method: str) -> bool:
    """
    Check if a collection exists
    """
    filepath = path_join(f"{method}{VECTOR_DB_SUFFIX}", collection_name)
    return path_exists(filepath)


def get_current_time() -> str:
    """
    Get the current time in the format "YYYY-MM-DD"
    """
    return str(datetime.now().strftime("%Y-%m-%d"))


def scan_manifest(rag_settings):
    """
    Scan the manifest.json file for the collection name and return the doc_ids
    """
    doc_ids = []
    with open('manifest.json', 'r') as f:
        data = json_load(f)
        for item in data["databases"]:
            if item["collection_name"] == rag_settings["collection_name"]:
                doc_ids = item["metadata"]["doc_ids"]
                return doc_ids
    return doc_ids


def update_manifest(rag_settings, doc_ids=[]):
    """
    Update the manifest.json file with the new collection

    Parameters:
    - rag_settings (dict): A dictionary containing the RAG settings.

    Records:
    - unique id, collection name, metadata
    - metadata includes embedding model, method, chunk size, chunk overlap, inputs, timestamp
    """
    # assert rag_settings is appropriately formed
    data = {}
    with open('manifest.json', 'r') as f:
        data = json_load(f)
    assert isinstance(data, dict), "manifest.json is not a dict"
    if "databases" not in data:
        data["databases"] = []
    databases = data["databases"]
    # assert that the id is unique
    for item in databases:
        if item["collection_name"] == rag_settings["collection_name"]:
            # Make sure the embedding model is the same
            if item["metadata"]["embedding_model"] != rag_settings["embedding_model"].model_name:
                raise ValueError("Embedding model must match manifest")
            if item["metadata"]["multivector_enabled"] != rag_settings["multivector_enabled"]:
                raise ValueError("Incompatibility with multivector_enabled")

            # Override rag_settings

            # No need to update manifest.json
    # get unique id
    unique_id = str(uuid4())
    print()
    # This is temporary since embedding model
    embedding_model_fn = rag_settings["embedding_model"]
    embedding_model_name = embedding_model_fn.model_name
    manifest = {
        "id": unique_id,
        "collection_name": rag_settings["collection_name"],
        "metadata": {
            "embedding_model": embedding_model_name,
            "method": rag_settings["method"],
            "chunk_size": str(rag_settings["chunk_size"]),
            "chunk_overlap": str(rag_settings["chunk_overlap"]),
            "inputs": rag_settings["inputs"],
            "timestamp": get_current_time(),
            "multivector_enabled": rag_settings["multivector_enabled"],
            "multivector_method": rag_settings["multivector_method"],
            "doc_ids": doc_ids
        }
    }
    databases.append(manifest)
    with open('manifest.json', 'w') as f:
        json_dump(data, f, indent=4)
    print("Updated manifest.json")
