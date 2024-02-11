# Load document
from helpers import get_pdf_pages, set_text_splitter, split_docs
from helpers import get_vectorstore, save_vector, get_document_chain, get_retrieval_chain, get_response
# from helpers import mistral as mistral_old
from helpers import rag_prompt, CallbackManager, StreamingStdOutCallbackHandler
from dotenv import load_dotenv
load_dotenv()
from langchain_together.embeddings import TogetherEmbeddings
from langchain_openai import ChatOpenAI
# from os import getenv
from langchain.schema import HumanMessage

PDF_FILENAME = "chromosome.pdf"
DEFAULT_EMBEDDING_CONTEXT_SIZE = '2k'
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 300


DEFAULT_ADVANCED = False
USE_ADVANCED = False

# Add support for Together

from models import get_together_mix#, together_quen, mistral, beagle

ACTIVE_LLM = get_together_mix()

def get_embedder(context_size = DEFAULT_EMBEDDING_CONTEXT_SIZE):
    assert context_size in ['2k', '8k'], "context_size must be either 2k or 8k"
    model = f"togethercomputer/m2-bert-80M-{context_size}-retrieval"
    embedder = TogetherEmbeddings(model=model)
    return embedder

def create_embeddings_for_pdf(documents, name, embedding_context_size = DEFAULT_EMBEDDING_CONTEXT_SIZE):
    assert embedding_context_size in ['2k', '8k'], "context_size must be either 2k or 8k"
    embedder = get_embedder(embedding_context_size)
    # vector = get_vector(documents, together_2k_embeddings)
    vector = get_vectorstore(documents, embedder)
    save_vector(vector, name)

def get_embeddings_filename(filepath):
    return f"{filepath[:-4]}-embeddings"

# def get_vectorstore()

def main(vector_path = PDF_FILENAME, query = None, embedding_context_size = DEFAULT_EMBEDDING_CONTEXT_SIZE, advanced=DEFAULT_ADVANCED):
    # Assumes embeddings have already been made
    embeddings_filename = get_embeddings_filename(vector_path)
    embedder = get_embedder(embedding_context_size)
    vector = get_vectorstore(documents=None, embedder=embedder, local_vector_path = embeddings_filename)
    document_chain = get_document_chain(ACTIVE_LLM, rag_prompt)
    # document_chain = get_document_chain(together_mix, rag_prompt)
    if advanced:
        retrieval_chain = get_retrieval_chain(vector, document_chain, advanced=True, llm=ACTIVE_LLM)
    else:
        retrieval_chain = get_retrieval_chain(vector, document_chain)
    # Query
    # query = """Respond as a graduate class instructor, being clear, concise, and comprehensive.
    # Elaborate on the processes utilized in path integration and the systems that allow such operation."""
    response = get_response(retrieval_chain, query)

# Read file sample.txt
excerpt = ''
with open('sample.txt', 'r') as file:
    excerpt = file.read()

BACKGROUND = "Respond as a graduate class instructor, being clear, concise, and comprehensive."
# BACKGROUND = "Explain the following production in a comprehensive manner."
# BACKGROUND = "How would I adjust this production rules without using a cond statement so that the unique letter of the 3 is identified and added to the imaginal buffer?"
NEW_QUERY = f'''{BACKGROUND}
"""
{excerpt}
"""
'''
QUERY = """Respond as a graduate class instructor, being clear, concise, and comprehensive.
How could the turing test be adapted in the current day with complex LLMs?"""


def make_embeddings():
    pages = get_pdf_pages(PDF_FILENAME)
    assert pages, "pdf must return document pages"
    # Split document
    splitter = set_text_splitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    # splitter = set_text_splitter()
    documents = split_docs(pages, splitter)
    create_embeddings_for_pdf(documents, name = get_embeddings_filename(PDF_FILENAME))

EMBEDDINGS_STEP = True
if EMBEDDINGS_STEP:
    make_embeddings()
else:
    main(query=NEW_QUERY, advanced=USE_ADVANCED)

#TODO: Add support for Together
# Step 1. Append system message and prompt to messages array
# Step 2. Get retrieval chain
# Step 3. Get response