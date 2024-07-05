from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from models.models import get_together_bigmix, get_together_qwen

SAMPLE_QWEN_QUERY = "Tell me a story about Qwen the warrior!"
SAMPLE_LLAMA_QUERY = "What is a llama?"


classifier_chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `Llama`, `Qwen`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | get_together_qwen()
    | StrOutputParser()
)

llama_chain = PromptTemplate.from_template(
    """You are an expert on llamas. \
Always answer questions starting with "As ye wise llama once said". \
Respond to the following question:

Question: {question}
Answer:"""
) | get_together_qwen()
qwen_chain = PromptTemplate.from_template(
    """You are an expert in wizardly matters of Qwen. \
Always answer questions starting with "As Father Alibaba told me". \
Respond to the following question:

Question: {question}
Answer:"""
) | get_together_qwen()
general_chain = PromptTemplate.from_template(
    """Respond to the following question:

Question: {question}
Answer:"""
) | get_together_bigmix()


def route_topic_to_chain(info):
    if "llama" in info["topic"].lower():
        return llama_chain
    elif "qwen" in info["topic"].lower():
        return qwen_chain
    else:
        return general_chain

full_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | RunnableLambda(
    route_topic_to_chain
)

def test_first_chain(query = SAMPLE_LLAMA_QUERY):
    if not query:
        print("No query provided")
        raise ValueError
    response_string = ""
    for chunk in classifier_chain.stream({"question": query}):
        print(chunk, end = "")
        response_string += chunk
    print("\n")
    print("Passed!") if response_string else print("Failed!")
    return response_string

def main(query = SAMPLE_QWEN_QUERY):
    if not query:
        print("No query provided")
        raise ValueError
    response_string = ""
    for chunk in full_chain.stream({"question": query}):
        print(chunk.content, end = "")
        response_string += chunk.content
    print("\n")
    print("Passed!") if response_string else print("Failed!")
    return response_string

main(SAMPLE_LLAMA_QUERY)