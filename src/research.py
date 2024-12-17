"""
STORM-inspired research assistant for generating comprehensive articles.
Based on https://arxiv.org/abs/2402.14207
"""
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
import asyncio

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph, START
from langgraph.pregel import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver


from dotenv import load_dotenv

from augmenta.classes import get_llm_fn

load_dotenv()

### ENV
import getpass
import os


def _set_env(var: str):
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var + ":")


# _set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
###
MAX_SEARCH_RESULTS = 10


### LLMS
try:
    gpt_4o_mini = get_llm_fn("gpt4o-mini").get_llm()
    gpt_4o = get_llm_fn("gpt4o").get_llm()
    samba = get_llm_fn("samba").get_llm()
    llama = get_llm_fn("llama").get_llm()
    gpt_3_5 = get_llm_fn("gpt3.5-turbo").get_llm()
except Exception as e:
    raise ValueError(f"Error initializing LLMs: {e}")


### Generate Initial Outline
direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Wikipedia writer. Write an outline for a Wikipedia page about a user-provided topic. Be comprehensive and specific.",
        ),
        ("user", "{topic}"),
    ]
)


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


generate_outline_direct = direct_gen_outline_prompt | llama.with_structured_output(
    Outline
)

### Expand Topics
gen_related_topics_prompt = ChatPromptTemplate.from_template(
    """I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

Please list the as many subjects and urls as you can.

Topic of interest: {topic}
"""
)


class RelatedSubjects(BaseModel):
    topics: List[str] = Field(
        description="Comprehensive list of related subjects as background research.",
    )


expand_chain = gen_related_topics_prompt | llama.with_structured_output(
    RelatedSubjects
)

### Generate Perspectives
class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(
        description="Name of the editor.", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )
    role: str = Field(
        description="Role of the editor in the context of the topic.",
    )
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )


gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You need to select a diverse (and distinct) group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\
    You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.

    Wiki page outlines of related topics for inspiration:
    {examples}""",
        ),
        ("user", "Topic of interest: {topic}"),
    ]
)

gen_perspectives_chain = gen_perspectives_prompt | gpt_3_5.with_structured_output(
    Perspectives
)

### Wikipedia
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable

wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)


def format_doc(doc, max_length=1000):
    related = "- ".join(doc.metadata["categories"])
    return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[
        :max_length
    ]


def format_docs(docs):
    return "\n\n".join(format_doc(doc) for doc in docs)


@as_runnable
async def survey_subjects(topic: str):
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    return await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})

### Expert Dialogue
def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references


def update_editor(editor, new_editor):
    # Can only set at the outset
    if not editor:
        return new_editor
    return editor


class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]

### Dialog Roles
gen_qn_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced Wikipedia writer and want to edit a specific page. \
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
Please only ask one question at a time and don't ask what you have asked before.\
Your questions should be related to the topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

Stay true to your specific perspective:

{persona}""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.model_dump(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | llama
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}

### Answer Questions
class Queries(BaseModel):
    queries: List[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's questions.",
    )


gen_queries_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. Query the search engine to answer the user's questions.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)
gen_queries_chain = gen_queries_prompt | llama.with_structured_output(Queries, include_raw=True)

### AnswerWithCitations
class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="Comprehensive answer to the user's question with citations.",
    )
    cited_urls: List[str] = Field(
        description="List of urls cited in the answer.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )


gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants\
 to write a Wikipedia page on the topic you know. You have gathered the related information and will now use the information to form a response.

Make your response as informative as possible and make sure every sentence is supported by the gathered information.
Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response. Include the cited_urls exactly as they are in the source.""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

gen_answer_chain = gen_answer_prompt | gpt_4o_mini.with_structured_output(
    AnswerWithCitations, include_raw=True
).with_config(run_name="GenerateAnswer")

### Search Engine

search_engine = TavilySearchResults(max_results=MAX_SEARCH_RESULTS)

### Gen Answer
import json

from langchain_core.runnables import RunnableConfig


async def gen_answer(
    state: InterviewState,
    config: Optional[RunnableConfig] = None,
    name: str = "Subject_Matter_Expert",
    max_str_len: int = 15000,
):
    swapped_state = swap_roles(state, name)  # Convert all other AI messages
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_engine.abatch(
        queries["parsed"].queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    # all_query_results = {
    #     res["url"]: res["content"] for results in successful_results for res in results
    # }
    all_query_results = {}
    for results in successful_results:
        for res in results:
            if isinstance(res, dict) and "url" in res and "content" in res:
                all_query_results[res["url"]] = res["content"]
    # We could be more precise about handling max token length if we wanted to here
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].tool_calls[0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    # Only update the shared state with the final answer to avoid
    # polluting the dialogue history with intermediate messages
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    # Save the retrieved information to a the shared state for future reference
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references}

# Construct the Interview Graph
max_num_turns = 5
from langgraph.pregel import RetryPolicy


def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"


builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=5))
builder.add_node("answer_question", gen_answer, retry=RetryPolicy(max_attempts=5))
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.add_edge(START, "ask_question")
interview_graph = builder.compile(checkpointer=False).with_config(
    run_name="Conduct Interviews"
)

### Refine Outline
refine_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
You need to make sure that the outline is comprehensive and specific. \
Topic you are writing about: {topic} 

Old outline:

{old_outline}""",
        ),
        (
            "user",
            "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Wikipedia outline:",
        ),
    ]
)

# Using turbo preview since the context can get quite long
refine_outline_chain = refine_outline_prompt | gpt_4o.with_structured_output(
    Outline
)

### Generate Sections
class SubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class WikiSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: List[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


section_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Wikipedia writer. Complete your assigned WikiSection from the following outline:\n\n"
            "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
        ),
        ("user", "Write the full WikiSection for the {section} section."),
    ]
)

### Generate Final Article
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:\n\n"
            "{draft}\n\nStrictly follow Wikipedia format guidelines.",
        ),
        (
            "user",
            'Write the complete Wiki article using markdown format. Organize citations using footnotes like "[1]",'
            " avoiding duplicates in the footer. Include URLs in the footer.",
        ),
    ]
)

writer = writer_prompt | gpt_4o | StrOutputParser()

### Final Flow
class ResearchState(TypedDict):
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[InterviewState]
    # The final sections output
    sections: List[WikiSection]
    article: str
async def main(main_topic: str = "Education Crisis in the US"):
    import logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)


    # example_topic = "Education Crisis in the US"
    example_topic = main_topic
    print(f"Researching topic: {example_topic}")
    
    # Generate initial outline
    initial_outline = await generate_outline_direct.ainvoke({"topic": example_topic})
    logger.info(f"Generated initial outline")
    print(f"Generated initial outline")
    # print(initial_outline.as_str)
    
    # Get perspectives
    perspectives = await survey_subjects.ainvoke(example_topic)
    logger.info(f"Got perspectives")
    print(f"Got perspectives")
    # Generate question
    # messages = [HumanMessage(f"So you said you were writing an article on {example_topic}?")]
    # question = await generate_question.ainvoke({
    #     "editor": perspectives.editors[0],
    #     "messages": messages,
    # })
    # print(question["messages"][0].content)
    
    # Generate queries
    # queries = await gen_queries_chain.ainvoke(
    #     {"messages": [HumanMessage(content=question["messages"][0].content)]}
    # )
    # print(queries["parsed"].queries)
    
    # Generate answer
    # example_answer = await gen_answer({
    #     "messages": [HumanMessage(content=question["messages"][0].content)]
    # })
    # print(example_answer["messages"][-1].content)
    
    # Conduct interview
    initial_state = {
        "editor": perspectives.editors[0],
        "messages": [
            AIMessage(
                content=f"So you said you were writing an article on {example_topic}?",
                name="Subject_Matter_Expert",
            )
        ],
    }

    final_step = None
    async for step in interview_graph.astream(initial_state):
        name = next(iter(step))
        print(name)
        print("-- ", str(step[name]["messages"])[:300])
        final_step = step
    
    final_state = next(iter(final_step.values()))
    logger.info(f"Conducted interviews")
    print(f"Conducted interviews")

    # Refine outline
    refined_outline = await refine_outline_chain.ainvoke({
        "topic": example_topic,
        "old_outline": initial_outline.as_str,
        "conversations": "\n\n".join(
            f"### {m.name}\n\n{m.content}" for m in final_state["messages"]
        ),
    })
    # print(refined_outline.as_str)
    logger.info(f"Refined outline")
    print(f"Refined outline")
    
    # Setup embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    reference_docs = [
        Document(page_content=v, metadata={"source": k})
        for k, v in final_state["references"].items()
    ]
    vectorstore = InMemoryVectorStore.from_documents(
        reference_docs,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(k=3)
    logger.info(f"Vector database setup complete.")
    print(f"Vector database setup complete.")


    # Write section
    async def retrieve(inputs: dict):
        docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
        formatted = "\n".join(
            [
                f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
        return {"docs": formatted, **inputs}

    section_writer = (
        retrieve
        | section_writer_prompt
        | gpt_4o.with_structured_output(WikiSection)
    )
    
    section = await section_writer.ainvoke({
        "outline": refined_outline.as_str,
        "section": refined_outline.sections[1].section_title,
        "topic": example_topic,
    })
    logger.info(f"Wrote section")
    print(f"Wrote section")
    # print(section.as_str)
    
    # Generate final article
    article_content = ""
    async for tok in writer.astream({"topic": example_topic, "draft": section.as_str}):
        article_content += tok
        # print(tok, end="")
    
    logger.info(f"Generated final article")
    print(f"Generated final article")
    
    # Save article content to markdown file
    with open("article.md", "w", encoding="utf-8") as f:
        f.write(article_content)
    logger.info("Saved article content to article.md")
    print("Saved article content to article.md")
    # Create PDF
    from fpdf import FPDF

    class MarkdownPDF(FPDF):
        def __init__(self):
            super().__init__()
            # Add UTF-8 support
            self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
            self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
            self.set_auto_page_break(auto=True, margin=15)
            
        def markdown_text(self, text, font_size=12):
            # Clean the text of problematic characters
            text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
            
            lines = text.split('\n')
            for line in lines:
                if not line.strip():
                    self.ln(5)  # Reduced paragraph spacing
                    continue
                    
                # Headers with compact spacing
                if line.startswith('###'):
                    self.ln(2)
                    self.set_font('DejaVu', 'B', font_size + 2)
                    self.multi_cell(0, 8, line.replace('###', '').strip())
                    self.ln(3)
                elif line.startswith('##'):
                    self.ln(3)
                    self.set_font('DejaVu', 'B', font_size + 4)
                    self.multi_cell(0, 8, line.replace('##', '').strip())
                    self.ln(4)
                elif line.startswith('#'):
                    self.ln(4)
                    self.set_font('DejaVu', 'B', font_size + 6)
                    self.multi_cell(0, 8, line.replace('#', '').strip())
                    self.ln(5)
                else:
                    self.set_font('DejaVu', '', font_size)
                    self.multi_cell(0, 6, line.strip())
                    self.ln(2)  # Reduced line spacing

    def create_pdf(example_topic, article_content, initial_outline):
        try:
            # Create PDF
            pdf = MarkdownPDF()
            pdf.add_page()

            # Title page
            pdf.set_font('DejaVu', 'B', 24)
            pdf.ln(60)  # Add space at top
            pdf.cell(0, 10, example_topic, ln=True, align='C')
            pdf.ln(20)
            
            # Add date
            pdf.set_font('DejaVu', '', 12)
            from datetime import datetime
            date_str = datetime.now().strftime("%B %d, %Y")
            pdf.cell(0, 10, date_str, ln=True, align='C')
            
            # Start article content on new page
            pdf.add_page()
            pdf.markdown_text(article_content)
            
            # Add appendix with initial outline
            pdf.add_page()
            pdf.set_font('DejaVu', 'B', 16)
            pdf.ln(10)
            pdf.cell(0, 10, "Appendix: Initial Research Outline", ln=True)
            pdf.ln(5)
            pdf.markdown_text(initial_outline.as_str)

            # Save PDF with sanitized filename
            filename = f"research_{example_topic.lower().replace(' ', '_')[:30]}.pdf"
            pdf.output(filename)
            logger.info(f"PDF created: {filename}")
            
            # Open the file in the default PDF viewer
            if os.path.exists(filename):
                os.system(f"open '{filename}'")
            return filename
        except Exception as e:
            logger.error(f"Failed to create and open PDF: {e}")
            return None

    create_pdf(example_topic, article_content, initial_outline)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate a research article on a given topic')
    parser.add_argument('topic', type=str, nargs='?', default="Education Crisis in the US",
                      help='Topic to research (default: "Education Crisis in the US")')
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.topic))
    except KeyboardInterrupt:
        print("Aborting!")
