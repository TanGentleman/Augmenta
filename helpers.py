from json import load as json_load
from json import dump as json_dump
from re import sub as re_sub
from uuid import uuid4
from langchain_core.documents import Document
from os.path import exists, join
from datetime import datetime

from config import VECTOR_DB_SUFFIX

# TODO:
# Add a ROOT_FILEPATH constant and use os.join to create the filepaths


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


def read_settings(config_file="settings.json") -> dict:
    settings = {}
    with open(config_file, 'r') as file:
        settings = json_load(file)
        assert isinstance(settings, dict), "Settings file is not a dictionary"
    return settings


def clean_text(text):
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


def format_docs(docs: list[Document], save_excerpts=True) -> str:
    """
    Formats the list of documents into a single string.
    Used to format the docs into a string for context that is passed to the LLM.
    """
    summaries = []
    # save documents here to excerpts.md
    context_string = ""
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
        context_string += f"Source: {source_string}\n\n"
    if save_excerpts:
        with open("excerpts.md", "w") as f:
            f.write(f"Context:\n{context_string}")
    return context_string


def collection_exists(collection_name: str, method: str) -> bool:
    """
    Check if a collection exists
    """
    filepath = join(f"{method}{VECTOR_DB_SUFFIX}", collection_name)
    return exists(filepath)


def get_current_time() -> str:
    return str(datetime.now().strftime("%Y-%m-%d"))


def update_manifest(rag_settings):
    """
    Update the manifest.json file with the new collection

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
            if item["metadata"]["embedding_model"] != rag_settings["embedding_model"].model:
                raise ValueError("Embedding model must match manifest")
            # print("No need to update manifest.json")
            return
    # get unique id
    unique_id = str(uuid4())
    print()
    try:
        model_name = str(rag_settings["embedding_model"].model)
    except BaseException:
        print("Could not get model name from embedding model")
        model_name = "Unknown model"
    manifest = {
        "id": unique_id,
        "collection_name": rag_settings["collection_name"],
        "metadata": {
            "embedding_model": model_name,
            "method": rag_settings["method"],
            "chunk_size": str(rag_settings["chunk_size"]),
            "chunk_overlap": str(rag_settings["chunk_overlap"]),
            "inputs": rag_settings["inputs"],
            "timestamp": get_current_time()
        }
    }
    databases.append(manifest)
    with open('manifest.json', 'w') as f:
        json_dump(data, f, indent=4)
    print("Updated manifest.json")
