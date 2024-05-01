# This file is for running a retrieval augmented generation on an existing
# vector db
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from helpers import clean_docs, database_exists, format_docs
from constants import get_rag_template, get_summary_template, get_eval_template
import embed

# assert embed.EXPERIMENTAL_UNSTRUCTURED is False, "Uncomment this line in rag.py to continue using Unstructured"
EXPERIMENTAL_UNSTRUCTURED = embed.EXPERIMENTAL_UNSTRUCTURED


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
    A dummy output parser that returns the output as is.
    """
    def is_output_valid(output):
        """
        Checks if the output is valid.
        """
        return "index" in output and "meetsCriteria" in output
    try:
        response_object = JsonOutputParser().parse(output.content)
        if not is_output_valid(response_object):
            raise ValueError("JSON output does not contain the required keys")
    except BaseException:
        raise ValueError("Eval chain did not return valid JSON")
    # print("Output:", output)
    # Perform JSON validation here
    # This function can also redirect to the next step in the pipeline
    print("yay!")
    return response_object


def get_eval_chain(llm):
    """
    Returns a chain for evaluating a given excerpt.

    This chain will NEED to be passed a dictionary with keys excerpt and criteria, not a string.
    """
    eval_prompt_template = get_eval_template()
    chain = (eval_prompt_template | llm | eval_output_handler)
    return chain


def get_rag_chain(
        retriever,
        llm,
        format_fn: callable = format_docs,
        system_message: str = None):
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
        docs = embed.documents_from_url(input)

    else:
        if EXPERIMENTAL_UNSTRUCTURED:
            docs = embed.documents_from_arbitrary_file(input)
        else:
            if input.endswith(".txt"):
                docs = embed.documents_from_text_file(input)
            else:
                assert input.endswith(
                    ".pdf"), "Invalid file type. Try enabling EXPERIMENTAL_UNSTRUCTURED."
                docs = embed.documents_from_local_pdf(input)
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
            vectorstore = embed.load_chroma_vectorstore(
                collection_name, embedder)
        elif method == "faiss":
            vectorstore = embed.load_faiss_vectorstore(
                collection_name, embedder)
        assert vectorstore is not None, "Collection exists but not loaded properly"
        return vectorstore
    for i in range(len(inputs)):
        # In the future this can be parallelized
        input = inputs[i]
        if not input:
            continue
        docs = input_to_docs(input)
        if not docs:
            print(f"No documents found for input {input}")
            continue
        docs = embed.split_documents(docs, chunk_size, chunk_overlap)
        if i == 0:
            if method == "chroma":
                vectorstore = embed.chroma_vectorstore_from_docs(
                    collection_name, embedder, docs)
            elif method == "faiss":
                vectorstore = embed.faiss_vectorstore_from_docs(
                    collection_name, embedder, docs)
        else:
            assert vectorstore is not None, "Vectorstore not initialized"
            # This method should work for both Chroma and FAISS
            vectorstore.add_documents(docs)
    assert vectorstore is not None, "Vectorstore not initialized. Provide valid inputs."
    return vectorstore
