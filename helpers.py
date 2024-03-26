from langchain.schema import HumanMessage
import json

import re
from langchain_core.documents import Document

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