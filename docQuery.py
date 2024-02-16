# Load document
from models import get_together_mix, get_together_quen
from helpers import get_pdf_pages, save_response_to_markdown_file, set_text_splitter, split_docs
from helpers import get_vectorstore, save_vector, get_document_chain, get_retrieval_chain, get_response
from helpers import rag_template
from langchain_together.embeddings import TogetherEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from os.path import exists
from config import EMBEDDINGS_STEP, ENABLE_SYSTEM_MESSAGE, PDF_FILENAME, EMBEDDING_CONTEXT_SIZE
from config import CHUNK_SIZE, CHUNK_OVERLAP, USE_ADVANCED, CHOSEN_MODEL, RAG_SYSTEM_MESSAGE
from langchain_core.prompts import ChatPromptTemplate
# PDF_FILENAME = "unit4_8k.pdf"
# EMBEDDING_CONTEXT_SIZE = '8k'
# CHUNK_SIZE = 7500
# CHUNK_OVERLAP = 200
# ACTIVE_LLM = get_together_mix()

ACTIVE_LLM = CHOSEN_MODEL()

def get_embedder(context_size = EMBEDDING_CONTEXT_SIZE):
    assert context_size in ['2k', '8k'], "context_size must be either 2k or 8k"
    model = f"togethercomputer/m2-bert-80M-{context_size}-retrieval"
    embedder = TogetherEmbeddings(model=model)
    return embedder

def create_embeddings_for_pdf(documents, name, embedding_context_size = EMBEDDING_CONTEXT_SIZE):
    embedder = get_embedder(embedding_context_size)
    vector = get_vectorstore(embedder, documents=documents)
    save_vector(vector, name)

def get_embeddings_filename(filepath):
    return f"{filepath[:-4]}-embeddings"

def main(vector_path = PDF_FILENAME, query = None, embedding_context_size = EMBEDDING_CONTEXT_SIZE, advanced=USE_ADVANCED):
    # Assumes embeddings have already been made
    assert query, "query must be provided"
    embeddings_filename = get_embeddings_filename(vector_path)
    embedder = get_embedder(embedding_context_size)
    vector = get_vectorstore(documents=None, embedder=embedder, local_vector_path = embeddings_filename)
    document_chain = get_document_chain(ACTIVE_LLM, rag_template)
    if advanced:
        retrieval_chain = get_retrieval_chain(vector, document_chain, advanced=True, llm=ACTIVE_LLM)
    else:
        retrieval_chain = get_retrieval_chain(vector, document_chain)
    # messages = []
    # if ENABLE_SYSTEM_MESSAGE:
    #     messages.append(("system", RAG_SYSTEM_MESSAGE))
    # messages.append(("human", "{input}"))
    # template = ChatPromptTemplate.from_messages(messages)
    # chain = template | retrieval_chain
    response = retrieval_chain.invoke({"input": query})
    save_response_to_markdown_file(response["answer"], "response.md")
    return response["answer"]

# Read file sample.txt
excerpt = ''
with open('sample.txt', 'r') as file:
    excerpt = file.read()

# BACKGROUND = "Respond as a graduate class instructor, being clear, concise, and comprehensive."
BACKGROUND = "You are an AI assistant for a graduate class. Respond being clear, concise, and comprehensive."
# BACKGROUND = "Explain:"
QUERY = f'''{BACKGROUND}
"""
{excerpt}
"""
'''
QUERY = excerpt

def make_embeddings():
    # Check if embeddings have already been made
    filename = get_embeddings_filename(PDF_FILENAME)
    if exists(f"vector-dbs/{filename}"):
        print("Embeddings already exist")
        return
    pages = get_pdf_pages(PDF_FILENAME)
    assert pages, "pdf must return document pages"
    # Split document
    splitter = set_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = split_docs(pages, splitter)
    create_embeddings_for_pdf(documents, filename)

if EMBEDDINGS_STEP:
    make_embeddings()
else:
    main(query=QUERY, advanced=USE_ADVANCED)

#TODO: Add support for Together
# Step 1. Append system message and prompt to messages array
# Step 2. Get retrieval chain
# Step 3. Get response