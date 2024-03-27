from json import load as json_load
from re import sub as re_sub
from langchain_core.documents import Document
from os.path import exists, join

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

def format_docs(docs: list[Document], save_excerpts = True) -> str:
    """
    Formats the list of documents into a single string.
    Used to format the docs into a string for context that is passed to the LLM.
    """
    summaries = []
    # save documents here to excerpts.md
    context_string = ""
    for doc in docs:
        # check if "Summary" is a key in the dict
        print(doc.metadata.keys())
        if "Summary" in doc.metadata and "Title" in doc.metadata:
            summary = doc.metadata["Summary"]
            summary_string = f"Summary for {doc.metadata['Title']}:\n{summary}"
            if summary not in summaries:
                summaries.append(summary)
                context_string += summary_string + "\n\n"
        context_string += doc.page_content + "\n\n"
    if save_excerpts:
        with open("excerpts.md", "w") as f:
            f.write(f"Context:\n{context_string}")
    return context_string

def collection_exists(collection_name: str, method: str) -> bool:
    """
    Check if a collection exists
    """
    filepath = join(f"{method}-vector-dbs", collection_name)
    return exists(filepath)