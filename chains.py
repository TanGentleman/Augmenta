from typing import Any, Callable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from constants import get_music_template, get_rag_template, get_summary_template, get_eval_template
from helpers import format_docs

class SimpleChain:
    """
    A simple chain class that can be used to create chains.
    """
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input_data: str | list | dict):
        """
        Invokes the chain with the given input data.
        """
        return self.chain.invoke(input_data)
    
    def stream(self, input_data: str | list | dict):
        """
        Streams the chain with the given input data.
        """
        return self.chain.stream(input_data)

class RAGChain(SimpleChain):
    """
    A chain that can be used to interact with the RAG pipeline.
    """
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input_data: str | list | dict):
        """
        Invokes the chain with the given input data.
        """
        return self.chain(input_data)

def get_object_from_response(string: str, validity_fn: Callable[[dict], bool] | None = None) -> Any:
    """
    A JSON output parser that returns the response object.
    """
    if not isinstance(string, str):
        print("CRITICAL: Expected string, got", type(string))
        try:
            string = string.content
            assert isinstance(string, str)
        except:
            raise ValueError("Could not convert input to string")
    
    if validity_fn is None:
        def is_output_valid(output_object: dict) -> bool:
            """
            Checks if the output is valid.
            """
            structure_passed = bool(
                "index" in output_object and "meetsCriteria" in output_object)
            if structure_passed:
                print("Meets criteria:", output_object["meetsCriteria"])
            return structure_passed
        print("No validity function provided. Using default.")
        validity_fn = is_output_valid
    try:
        response_object = JsonOutputParser().parse(string)
        print("Checking validity of response object")
        if not validity_fn(response_object):
            print("JSON output does not contain the required keys")
            raise ValueError("JSON output does not contain the required keys")
    except ValueError:
        raise ValueError("Eval chain did not return valid JSON")
    # print("Output:", output)
    return response_object


def get_eval_chain(llm, validity_fn: Callable[[dict], bool] | None = None) -> SimpleChain:
    """
    Returns a chain for evaluating a given excerpt.

    This chain will NEED to be passed a dictionary with keys excerpt and criteria, not a string.
    """
    eval_prompt_template = get_eval_template()
    # How can I pass the validity function to get_object_from_response
    # and then use it in the chain?
    # Response: I can pass it as an argument to the function.

    {"first_string": eval_prompt_template | llm | StrOutputParser() }

    chain = SimpleChain(eval_prompt_template | llm | StrOutputParser() | (lambda x: get_object_from_response(string = x, validity_fn = validity_fn)))
    return chain

def music_output_handler(response_string: str):
    """
    A parser that returns a list of dictionaries.

    Each dictionary should have keys "title", "artist", and "album".
    """
    def is_output_valid(output_object: dict) -> bool:
        """
        Checks if the output is valid.
        """
        for item in output_object:
            if not all(key in item for key in ["title", "artist", "album"]):
                print(f"At least one song does not contain the required keys")
                return False
        return True
    return get_object_from_response(response_string, is_output_valid)

def get_music_chain(llm, few_shot_examples=None):
    """
    Returns a chain for the music pipeline.
    """
    music_prompt_template = get_music_template()
    few_shot_string = ""
    if few_shot_examples:
        assert isinstance(few_shot_examples, list)
        for example in few_shot_examples:
            assert "input" in example and "output" in example
            few_shot_string += f"input: {example['input']}\noutput: {example['output']}\n\n"
    chain = (
        {"input": RunnablePassthrough(), "few_shot_examples": lambda x: few_shot_string}
        | music_prompt_template
        | llm
        | StrOutputParser()
        | music_output_handler
    )
    return chain

def get_rag_chain(
        retriever,
        llm,
        format_fn=format_docs,
        system_message: str | None = None):
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

def get_summary_chain(llm) -> SimpleChain:
    """
    Returns a chain for summarization only.
    Can be invoked, like `chain.invoke("Excerpt of long reading:...")` to get a response.
    """
    chain = SimpleChain(
        {"excerpt": lambda x: x}
        | get_summary_template()
        | llm
        | StrOutputParser()
    )
    return chain