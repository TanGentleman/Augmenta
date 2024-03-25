# This file is for running a retrieval augmented generation on an existing vector db
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from helpers import clean_docs
from constants import RAG_TEMPLATE
import embed

# These are the urls that get ingested as documents. Should be a list of strings.
DEFAULT_URLS = [
    "https://python.langchain.com/docs/integrations/vectorstores/faiss",
]
# This the folder name within chroma-vector-dbs or faiss-dbs where the vector db is stored
DEFAULT_COLLECTION_NAME = "langchain_faiss_collection"
# This is the question you want to ask (retriever will choose chunks of the documents as context to answer the question)
DEFAULT_QUESTION = """How can I add documents to an existing faiss vector db?"""

def format_docs(docs: list[Document], save_excerpts = True) -> str:
    """
    Formats the list of documents into a single string.
    Used to format the docs into a string for context that is passed to the LLM.
    """
    # save documents here to excerpts.md
    context = "\n\n".join(doc.page_content for doc in docs)
    if save_excerpts:
        with open("excerpts.md", "w") as f:
            f.write(f"Context:\n{context}")
    return context

def get_rag_chain(retriever, llm):
    """
    Input: retriever (contains vectorstore with documents) and llm
    Returns a chain for the RAG pipeline.
    Can be invoked with a question, like `chain.invoke("How do I do x task using this framework?")` to get a response.
    """
    # Get prompt template
    rag_prompt_template = RAG_TEMPLATE
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_template
        | llm
    )
    return chain


def input_to_docs(input: str) -> list[Document]:
    """
    Converts the input to a list of documents.
    """
    if input.startswith("http"):
        docs = embed.documents_from_url(input)
    elif input.endswith(".txt"):
        docs = embed.documents_from_text_file(input)
    else:
        # Loading pdf
        docs = embed.documents_from_local_pdf(input)
    if not docs:
        print("No documents found")
        return None
    docs = clean_docs(docs)
    return docs

def vectorstore_from_inputs(inputs: str | list[str], method: str, embedder, collection_name: str):
    method = method.lower()
    assert method in ["chroma", "faiss"], "Invalid method"
    """
    Args:
    - inputs: list of strings (urls or filepaths to .pdf or .txt files)
    - method: "chroma" or "faiss"
    - embedder: the embedder to use
    - collection_name: the name of the collection to create/load
    Returns:
    - vectorstore: the vectorstore created from the inputs (FAISS or Chroma)
    """
    vectorstore = None
    if isinstance(inputs, str):
        inputs = [inputs]
    for i in range(len(inputs)):
        docs = input_to_docs(inputs[i])
        docs = embed.split_documents(docs)
        if i == 0:
            if method == "chroma":
                vectorstore = embed.create_chroma_vectorstore(embedder, collection_name, docs)
            elif method == "faiss":
                vectorstore = embed.create_faiss_vectorstore(embedder, collection_name, docs)
        else:
            assert vectorstore is not None, "Vectorstore not initialized"
            # This method should work for both Chroma and FAISS
            vectorstore.add_documents(docs)
    return vectorstore

# def main(inputs: str | list[str] = DEFAULT_URLS, collection_name: str = DEFAULT_COLLECTION_NAME, question: str = DEFAULT_QUESTION):
#     # from models import get_openai_embedder_small, get_claude_sonnet
#     from models import get_openai_embedder_large, get_claude_opus
#     embedder = get_openai_embedder_large()
#     llm = get_claude_opus()
#     vectorstore = vectorstore_from_inputs(inputs, "chroma", embedder, collection_name)
#     retriever = vectorstore.as_retriever()
#     # Can add optional arguments like search_kwargs={"score_threshold": 0.5}
#     chain = get_rag_chain(retriever, llm)
#     output = chain.invoke(question)
#     return output

# if __name__ == "__main__":
#     main()