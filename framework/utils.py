from typing import Literal
from constants import CHROMA_FOLDER, FAISS_FOLDER, VECTOR_DB_SUFFIX
from json import JSONDecodeError, load as json_load
from json import dump as json_dump
from re import sub as regex_sub
from uuid import uuid4
from langchain_core.documents import Document
from datetime import datetime

# Get the root path of the repository
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
DATA_DIR = ROOT / "data"
LLM_RESPONSE_PATH = DATA_DIR / "llm-outputs" / "markdown"
TEXT_FILE_DIR = DATA_DIR / "txt"
CONFIG_DIR = ROOT / "config"
DB_DIR = DATA_DIR / "databases"
CHROMA_FOLDER_PATH = DB_DIR / CHROMA_FOLDER
FAISS_FOLDER_PATH = DB_DIR / FAISS_FOLDER

MANIFEST_FILEPATH = DATA_DIR / "manifest.json"

def copy_string_to_clipboard(string: str) -> str | None:
    """
    Copy a string to the clipboard.

    Parameters:
    - string (str): The string to be copied.
    """
    try:
        from pyperclip import copy
        copy(string)
        print("INFO: CLIPBOARD WRITE")
        return string
    except ImportError:
        print("pyperclip is not installed. Install it using 'pip install pyperclip'")
        return None


def get_clipboard_contents() -> str | None:
    """
    Get the contents of the clipboard.
    """
    try:
        from pyperclip import paste
        print("INFO: CLIPBOARD READ")
        return paste()
    except ImportError:
        print("pyperclip is not installed. Install it using 'pip install pyperclip'")
        return None


def save_string_as_markdown_file(
        response_string: str,
        filename="response.md"):
    """
    Save a response string to a markdown file.

    Parameters:
    - response_string (str): The response string to be saved.
    - filename (str, optional): The name of the file. Defaults to "response.md".
    """
    filepath = LLM_RESPONSE_PATH / filename
    if not filepath.exists():
        print(f"Creating new file: {filepath}")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(response_string)


def read_text_file(filename: str = "sample.txt") -> str:
    """
    Read a text file and return the contents as a string.

    Parameters:
    - filename (str): The name of the file to be read.
    """
    filepath = TEXT_FILE_DIR / filename
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return ""
    with open(filepath, "r") as file:
        print(f"Reading file: {filename}")
        return file.read()


def read_sample():
    """
    Read the sample.txt file and return the excerpt.
    """
    return read_text_file("sample.txt")


def read_settings(config_filename="settings.json") -> dict:
    """
    Read the settings file and return the settings as a dictionary.
    """
    settings = {}
    filepath = CONFIG_DIR / config_filename
    if not filepath.exists():
        print(f"File not found: {filepath}")
        print("Try tossing a .json file in the framework/config directory!")
        raise FileNotFoundError
    if not filepath.suffix == ".json":
        print("CRITICAL: settings file is not a .json file")
        raise ValueError
    with open(filepath, 'r') as file:
        try:
            settings = json_load(file)
        except JSONDecodeError:
            print("CRITICAL: Error reading settings file")
            raise JSONDecodeError
    assert isinstance(settings, dict), "Settings file is not a dictionary"
    return settings


def clean_text(text):
    """
    Clean text while preserving important symbols and punctuation, return str
    """
    # Replace multiple whitespace characters with a single space
    cleaned_text = regex_sub(r'\s+', ' ', text)

    # Keep important symbols and punctuation, remove unwanted characters
    cleaned_text = regex_sub(
        r'[^a-zA-Z0-9(),.!?;:\-@#$%^&*_+={}|[\]<>/`~ ]',
        '',
        cleaned_text)

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
    res_docs: list[Document] = []
    for doc in docs:
        start_index = doc.page_content.find(left_string)
        end_index = doc.page_content.find(right_string)
        if start_index != -1 and end_index != -1:
            extracted_content = doc.page_content[start_index +
                                                 len(left_string):end_index]
            doc.page_content = extracted_content
            res_docs.append(doc)
            if len(extracted_content) > 4000:
                print("Warning: Document is really long, consider splitting it further.")
        else:
            print("Role/example not found in document. NOT adding to docs.")
    return res_docs


def format_docs(
        docs: list[Document],
        save_excerpts=True,
        process_docs_fn=None,
        filename="excerpts.md") -> str:
    """
    Formats the list of documents into a single string as context to be passed to LLM.

    Parameters:
    - docs (list[Document]): A list of Document objects.
    - save_excerpts (bool, optional): Whether to save the excerpts to a markdown file. Defaults to True.
    """
    filename = fix_filename(filename)
    filepath = LLM_RESPONSE_PATH / filename

    summaries = []
    # save documents here to excerpts.md
    context_string = ""
    EXPERIMENTAL_PROCESSING = False
    if EXPERIMENTAL_PROCESSING:
        if process_docs_fn:
            docs = process_docs_fn(docs)
    excerpt_count = 0
    for doc in docs:
        excerpt_count += 1
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
        # add source
        source_string = doc.metadata["source"] if "source" in doc.metadata else "Unknown source"
        # TODO: Websites with gross urls can be snipped
        # in some cases having the page metadata is useful.
        # i currently don't have good a use case aside from creating child docs
        if "page" in doc.metadata:
            source_string += f" (Page {doc.metadata['page']})"
        if "index" in doc.metadata:
            source_string += f" (Index: {doc.metadata['index']})"
        else:
            # print("Warning: No index found in metadata")
            pass
        context_string += f"Source: {source_string}\n"
        context_string += doc.page_content + "\n\n"

    if save_excerpts:
        with open(filepath, "w") as f:
            f.write(f"Context:\n{context_string}")
    context_string = context_string.strip()
    return context_string


def database_exists(collection_name: str, method: str) -> bool:
    """
    Check if a vector database exists from the given collection name and method.
    """
    # filepath = path_join(f"{method}{VECTOR_DB_SUFFIX}", collection_name)
    if method == "faiss":
        filepath = FAISS_FOLDER_PATH / collection_name
    elif method == "chroma":
        filepath = CHROMA_FOLDER_PATH / collection_name
    else:
        raise ValueError("Invalid method")

    return filepath.exists()

def clear_database(collection_name: str, method: str) -> bool:
    """
    Clear the database from the given collection name and method.
    """
    if method == "faiss":
        filepath = FAISS_FOLDER_PATH / collection_name
    elif method == "chroma":
        filepath = CHROMA_FOLDER_PATH / collection_name
    else:
        raise ValueError("Invalid method")

    if filepath.exists():
        # Rename the folder to a temporary name
        temp_name = filepath.with_name(f"{collection_name}-trash")
        filepath.rename(temp_name)
        print(f"Renamed {filepath.name} to {temp_name.name}")
        return True
    print(f"WARNING: Database not found!")
    return False

def get_manifest_data(collection_name: str, method: str) -> dict | None:
    """
    Check if an entry exists in the manifest.json file.
    """
    filepath = MANIFEST_FILEPATH
    manifest_entry = {}
    try:
        with open(filepath, "r") as f:
            data = json_load(f)
            if not data:
                print("manifest.json is empty")
                return None
    except FileNotFoundError:
        print("manifest.json not found")
        with open(filepath, "w") as f:
            json_dump({"databases": []}, f, indent=4)
        return None
    for item in data["databases"]:
        if item["collection_name"] == collection_name and item["metadata"]["method"] == method:
            manifest_entry = item
            return manifest_entry
    return None

def get_timestamp() -> str:
    """
    Get the current time in the format "YYYY-MM-DD"
    """
    return str(datetime.now().strftime("%Y-%m-%d"))

def get_subfolder_names(filepath: Path | str) -> list[str]:
    """
    Get the collection names from the given path
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert filepath.exists(), "Path does not exist"
    return [name.name for name in filepath.iterdir() if name.is_dir()]

def get_db_collection_names(db_type: Literal["chroma", "faiss", "all"]) -> dict[Literal["chroma", "faiss"], list[str]]:
    """
    Get the collection names from the manifest.json file
    """
    assert db_type in ["chroma", "faiss", "all"], "Invalid method"
    chroma_collection_names = []
    faiss_collection_names = []
    # collection_names = []
    if db_type == "chroma":
        filepath = CHROMA_FOLDER_PATH
        if filepath.exists():
            print("Chroma databases")
            chroma_collection_names = get_subfolder_names(filepath)
    elif db_type == "faiss":
        filepath = CHROMA_FOLDER_PATH
        if filepath.exists():
            print("Faiss databases")
            faiss_collection_names = get_subfolder_names(filepath)
    elif db_type == "all":
        print("Collecting all databases!")
        chroma_collection_names = get_subfolder_names(CHROMA_FOLDER_PATH)
        faiss_collection_names += get_subfolder_names(FAISS_FOLDER_PATH)
    else:
        raise ValueError("Invalid database type")
    return {
        "chroma": chroma_collection_names,
        "faiss": faiss_collection_names
    }


def update_manifest(
        embedding_model_name: str,
        method: str,
        chunk_size: int,
        chunk_overlap: int,
        inputs: list[str],
        collection_name: str,
        doc_ids: list[str] = []):
    """
    Update the manifest.json file with the new collection

    Parameters:
    - rag_settings (dict): A dictionary containing the RAG settings.

    Records:
    - unique id, collection name, metadata
    - metadata includes embedding model, method, chunk size, chunk overlap, inputs, timestamp
    """
    # assert rag_settings is appropriately formed
    manifest_entry = get_manifest_data(collection_name, method)
    if manifest_entry:
        # NOTE: It is possible to override the manifest.json chunk here
        print("Entry already found in manifest.json. Returning False")
        return False
    filepath = MANIFEST_FILEPATH
    
    # assert that the id is unique
    # get unique id
    unique_id = str(uuid4())
    # This is temporary since embedding model
    manifest = {
        "id": unique_id,
        "collection_name": collection_name,
        "metadata": {
            "embedding_model": embedding_model_name,
            "method": method,
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap),
            "inputs": inputs,
            "timestamp": get_timestamp(),
            "doc_ids": doc_ids
        }
    }
    with open (filepath, "r") as f:
        data = json_load(f)

    assert isinstance(data, dict), "manifest.json is not a dict"
    assert "databases" in data, "databases key not found in manifest.json"
    assert isinstance(data["databases"], list), "databases key is not a list"
    data["databases"].append(manifest)
    with open(filepath, "w") as f:
        json_dump(data, f, indent=4)
    print("Updated manifest.json")


def save_config_as_json(data, filename: str):
    """
    Save the current config to a JSON file
    """
    filepath = CONFIG_DIR / filename
    with open(filepath, "w") as f:
        json_dump(data, f, indent=2)

def fix_filename(filename: str | Path) -> str:
    """
    Fixes the filename by ensuring it is an appropriate string format.

    Args:
        filename (str | Path): The filename to be fixed.

    Returns:
        str: The fixed filename.
    """
    if isinstance(filename, Path):
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        print("Warning: Converting path to just the filename")
        return str(filename.name)
    return filename