# This file is for running a retrieval augmented generation on an existing vector db
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from operator import itemgetter
from helpers import clean_docs

import embed

# These are the urls that get ingested as documents. Should be a list of strings.
DEFAULT_URLS = [
    "https://python.langchain.com/docs/integrations/vectorstores/faiss",
]
# This the folder name within chroma-vector-dbs or faiss-dbs where the vector db is stored
DEFAULT_COLLECTION_NAME = "langchain_faiss_collection"
# This is the question you want to ask (retriever will choose chunks of the documents as context to answer the question)
DEFAULT_QUESTION = """How can I add documents to an existing faiss vector db?"""

def format_docs(docs: list[Document]) -> str:
    """
    Formats the list of documents into a single string.
    Used to format the docs into a string for context that is passed to the LLM.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_template():
    """
    Fetches the RAG template for the prompt.
    This template expects to be passed values for both context and question.
    """
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    rag_prompt_template = ChatPromptTemplate.from_template(template)
    rag_prompt_template.messages.insert(0, 
        SystemMessage(
            content="You are an AI programming assistant. Use the document excerpts to respond to the best of your ability."
        )
    )
    return rag_prompt_template

def get_chain(retriever, llm, memory: ConversationBufferMemory | None = None):
    """
    Input: retriever (contains vectorstore with documents) and llm
    Returns a chain for the RAG pipeline.
    Can be invoked with a question, like `chain.invoke("How do I do x task using this framework?")` to get a response.
    """
    # Get prompt template
    rag_prompt_template = get_rag_template()
    # Set memory
    if memory is None:
        memory = ConversationBufferMemory(
        return_messages=True, input_key="question", output_key="answer"
    )
    # Load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | loaded_memory
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    # Save memory
    # This saves the history of the conversation
    # Return memory along with the chain
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
    - inputs: list of strings (urls or filepaths to pdfs [will support text files soon])
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
#     chain = get_chain(retriever, llm)
#     output = chain.invoke(question)
#     return output

# if __name__ == "__main__":
#     main()