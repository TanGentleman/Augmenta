
import pyperclip
try:
    import gnureadline
except ImportError:
    pass
from dotenv import load_dotenv
load_dotenv()

from uuid import uuid4
from os import get_terminal_size
from textwrap import fill

from langchain.schema import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore

import utils
# from utils import copy_string_to_clipboard, database_exists, get_clipboard_contents, get_db_collection_names, process_docs, get_doc_ids_from_manifest, save_string_as_markdown_file, read_sample, update_manifest
from constants import DEFAULT_QUERY, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES, PROMPT_CHOOSER_SYSTEM_MESSAGE, RAG_COLLECTION_TO_SYSTEM_MESSAGE, SUMMARY_TEMPLATE, SYSTEM_MESSAGE_CODES
from config.config import MAX_CHARACTERS_IN_PARENT_DOC, MAX_PARENT_DOCS, SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE, EXPLAIN_EXCERPT, FILTER_TOPIC
from classes import Config
from models.models import LLM_FN, LLM
from chains import get_summary_chain, get_rag_chain, get_eval_chain
from rag import input_to_docs, get_chroma_vectorstore_from_docs, get_faiss_vectorstore_from_docs, load_existing_faiss_vectorstore, load_existing_chroma_vectorstore, split_documents


TERMINAL_WIDTH = get_terminal_size().columns

def print_adjusted(
        text: str,
        end='\n',
        flush=False,
        width=TERMINAL_WIDTH) -> None:
    '''
    Prints text with adjusted line wrapping
    '''
    print(fill(text, width=width), end=end)
    # lines = text.splitlines()
    # for line in lines:
    #     # If line is longer than terminal width, wrap to fit
    #     if len(line) > width:
    #         wrapped_line = fill(line, width=width)
    #         line = wrapped_line
    #     print(line, end=end, flush=flush)
    # Easier method


PROCESSING_DOCS_FN = utils.process_docs

ID_KEY = "doc_id"

REFORMATTING_PROMPTS = [".paste", ".read"]
COMMAND_LIST = [
    "del",
    "quit",
    "exit",
    "switch",
    "refresh",
    "save",
    "saveall",
    "info",
    "rag",
    "reg",
    ".s",
    ".names",
    ".copy",
    ".rm",
    ".eval"]


class Chatbot:
    """
    An interactive RAG-capable chatbot class.

    Parameters:
    - config (Config, optional): The configuration object. Holds ChatSettings and RagSettings.

    Attributes:
    - config (Config): The configuration object.
    - chat_model (LLM): The chat model.
    - rag_model (LLM): The RAG model.
    - backup_model (LLM): The backup model.
    - parent_docs (list[Document]): The parent documents.
    - doc_ids (list[str]): The document IDs.
    - rag_chain (Runnable): The RAG chain.
    - retriever (MultiVectorRetriever): The retriever.
    - count (int): How many exchanges (back and forth messages) in history.
    - exit (bool): The exit flag for the chat loop.
    - messages: The message history used in context for non-RAG chat.

    Methods:
    - chat: The main chat loop.
    - initialize_rag: Initializes the retriever and RAG chain.
    - refresh_rag_state: Refreshes the RAG state.
    - refresh_config: Refreshes the configuration.
    - _set_doc_ids: Sets document IDs.
    - get_child_docs: Generates child documents.
    - _get_vectorstore: Retrieves or creates a vectorstore.
    - run_eval_tests_on_vectorstore: Runs evaluation tests on the vectorstore.
    - get_retriever: Retrieves the retriever.
    - messages_to_string: Converts messages to a single string.
    - get_clipboard: Retrieves clipboard content.
    - set_clipboard: Sets clipboard content.
    - handle_command: Handles commands.
    - get_chat_response: Gets a chat response.
    - get_rag_response: Gets a RAG response.
    - _pop_last_exchange: Deletes the last exchange.
    """

    def __init__(self, config=None):
        """
        Initializes the Chatbot with the given configuration.

        Parameters:
        - config (Config, optional): The configuration object. Defaults from settings.json if not provided.
        """
        if config is None:
            config = Config()

        # TODO: Migrate filter topic into rag_settings
        self.filter_topic = FILTER_TOPIC

        self.config = config
        self.chat_model = None
        self.rag_model = None
        self.backup_model = None

        self.parent_docs = None
        self.doc_ids = []
        self.rag_chain = None
        self.retriever = None
        self.response_count = 0
        self.exit = False
        self.messages = []

        if self.config.rag_settings.rag_mode:
            self.initialize_rag()
        else:
            self.chat_model = LLM(self.config.chat_settings.primary_model)
            self.set_messages()

    def set_messages(self, messages: list[BaseMessage] | None = None):
        """
        Sets the starting messages based on the configuration.
        """
        if messages is not None:
            assert isinstance(messages, list)
            assert all(isinstance(m, BaseMessage) for m in messages)
            self.messages = messages
            self.response_count = len(messages) // 2
            return
        
        messages = []
        if not self.config.rag_settings.rag_mode:
            if self.config.chat_settings.enable_system_message:
                messages.append(SystemMessage(content=self.config.chat_settings.system_message))
            else:
                print('System message disabled')

        self.messages = messages
        self.response_count = 0
        return

    def initialize_rag(self) -> bool:
        """
        Initializes the retriever and RAG chain.

        Raises:
            ValueError: If Doc ID is not initialized.
            Exception: If RAG mode is not enabled.

        Returns:
            bool: Whether the initialization was successful.
        """
        if self.retriever is not None:
            print('Retriever already exists. Use "refresh" to clear it first')
            return False
        if not self.config.rag_settings.rag_mode:
            print('RAG mode not enabled')
            raise Exception('RAG mode not enabled')
        assert self.config.rag_settings.rag_mode, "RAG mode not enabled"
        self.rag_model = LLM(self.config.rag_settings.rag_llm)

        # Get doc_ids
        if self.config.rag_settings.multivector_enabled:
            doc_ids = utils.get_doc_ids_from_manifest(self.config.rag_settings.collection_name)
            if self.config.rag_settings.database_exists and not doc_ids:
                raise ValueError("Doc IDs not initialized")
            self.doc_ids = doc_ids

        self.retriever = self.get_retriever()
        rag_system_message = RAG_COLLECTION_TO_SYSTEM_MESSAGE.get(
            self.config.rag_settings.collection_name, "default")
        self.rag_chain = get_rag_chain(
            self.retriever,
            self.rag_model.llm,
            system_message=rag_system_message)
        self.set_messages()
        utils.update_manifest(
            embedding_model_name=self.config.rag_settings.embedding_model.model_name,
            method=self.config.rag_settings.method,
            chunk_size=self.config.rag_settings.chunk_size,
            chunk_overlap=self.config.rag_settings.chunk_overlap,
            inputs=self.config.rag_settings.inputs,
            collection_name=self.config.rag_settings.collection_name,
            doc_ids=self.doc_ids)
        
        # Save current config to active.json
        self.config.save_to_json()
        return True

    def refresh_rag_state(self):
        """
        Refreshes the RAG state.
        """
        if not self.config.rag_settings.rag_mode:
            print('RAG mode not enabled')
        self.retriever = None
        self.rag_chain = None
        self.doc_ids = []
        self.parent_docs = None
        self.set_messages()

    def refresh_config(self, config: Config | None = None):
        """
        Refreshes the configuration.

        Args:
            config (Config, optional): The configuration object. Defaults to None.
        """
        if config is None:
            # Reload config from settings.json
            # Override with the current rag_mode setting
            config_override = {}
            config_override["RAG"] = {}
            config_override["RAG"]["rag_mode"] = self.config.rag_settings.rag_mode
            config = Config(config_override=config_override)
        self.config = config
        self.chat_model = None
        self.backup_model = None

        if self.config.rag_settings.rag_mode:
            self.initialize_rag()
        else:
            self.chat_model = LLM(self.config.chat_settings.primary_model)
            self.set_messages()
            self.response_count = 0

    def _set_doc_ids(self):
        assert self.config.rag_settings.rag_mode, "RAG mode must be on"
        assert self.config.rag_settings.multivector_enabled, "Multivector not enabled"
        assert self.parent_docs, "Parent docs not initialized"

        if self.doc_ids:
            print("Doc IDs already initialized")
        else:
            self.doc_ids = [str(uuid4()) for _ in self.parent_docs]
        assert self.doc_ids, "Doc IDs not created"

    def _get_child_docs(self) -> list[Document]:
        # This function has limits for doc count/size set in constants.py
        assert self.config.rag_settings.multivector_enabled, "Multivector not enabled"
        assert isinstance(
            self.rag_model, LLM), "RAG LLM not initialized"
        # When qa is supported, this will check method
        assert self.parent_docs and len(
            self.parent_docs) < MAX_PARENT_DOCS, f"Temporary limit of {MAX_PARENT_DOCS} parent Documents"
        print('Now estimating token usage for child documents')
        for doc in self.parent_docs:
            if "char_count" not in doc.metadata:
                print(
                    'Char count not found in metadata. This means parent docs were never split')
            if len(doc.page_content) > MAX_CHARACTERS_IN_PARENT_DOC:
                raise ValueError(
                    'Document too long, split before making child documents')
        parent_texts = [doc.page_content + '\nsource: ' +
                        doc.metadata["source"] for doc in self.parent_docs]
        summarize_chain = get_summary_chain(self.rag_model.llm)
        child_texts = summarize_chain.batch(
            parent_texts, {"max_concurrency": 5})
        assert len(self.parent_docs) == len(
            child_texts), "Parent and child texts do not match"

        assert self.doc_ids, "Doc IDs not initialized"

        child_docs = []
        for i, text in enumerate(child_texts):
            new_doc = Document(
                page_content=text, metadata={
                    ID_KEY: self.doc_ids[i]})
            child_docs.append(new_doc)
        return child_docs

    def index_docs(self, processing_docs_fn=PROCESSING_DOCS_FN) -> list[Document]:
        """
        Indexes documents for the vectorstore.

        Returns:
            list[Document]: The indexed documents.
        """
        assert self.config.rag_settings.rag_mode, "RAG mode not enabled"
        inputs = self.config.rag_settings.inputs
        collection_name = self.config.rag_settings.collection_name
        # Check if collection exists
        if utils.database_exists(collection_name, self.config.rag_settings.method):
            raise ValueError(f"Collection {collection_name} already exists")
        docs = []
        for i in range(len(inputs)):
            if not inputs[i]:
                print(f'Input {i} is empty, skipping')
                continue
            new_docs = input_to_docs(inputs[i])
            if not new_docs:
                print(f'No documents found in input {i}')
                continue
            docs.extend(new_docs)
            print("Indexed", inputs[i])

        assert docs, "No documents to create collection"
        if RAG_COLLECTION_TO_SYSTEM_MESSAGE.get(collection_name) == PROMPT_CHOOSER_SYSTEM_MESSAGE:
            if processing_docs_fn:
                print("Prompt chooser detected, processing documents...")
                docs = processing_docs_fn(docs)
        docs = split_documents(docs,
                            self.config.rag_settings.chunk_size,
                            self.config.rag_settings.chunk_overlap)
        if not docs:
            print('No documents generated!')
            return []
        return docs

    def _get_vectorstore(self) -> Chroma | FAISS:
        """
        Retrieves or creates a vectorstore.

        Args:
            processing_docs_fn (function, optional): Function to process documents before indexing.

        Returns:
            Chroma | FAISS: The vectorstore.

        Raises:
            AssertionError: If RAG mode is not enabled or embedding model is not an instance of LLM_FN.
        """
        assert self.config.rag_settings.rag_mode, "RAG mode not enabled"
        collection_name = self.config.rag_settings.collection_name
        method = self.config.rag_settings.method
        embedding_model_fn = self.config.rag_settings.embedding_model
        assert isinstance(embedding_model_fn, LLM_FN)
        embedder = embedding_model_fn.get_llm()

        vectorstore = None

        if utils.database_exists(collection_name, method):
            print(f"Loading existing Vector DB: {collection_name}")
            if method == "chroma":
                vectorstore = load_existing_chroma_vectorstore(collection_name, embedder)
            elif method == "faiss":
                vectorstore = load_existing_faiss_vectorstore(collection_name, embedder)
            assert vectorstore is not None, "Collection exists but not loaded properly"
            if self.config.rag_settings.multivector_enabled:
                docs = self.index_docs()
                assert docs, "No documents to make parent docs"
                print("Warning: Parent docs regenerated from inputs")
                assert len(self.parent_docs) == len(self.doc_ids), "Parent docs and doc IDs do not match length"
                for doc in docs:
                    if doc.metadata["char_count"] > 20000:
                        print('WARNING: Parent documents very long, split to avoid expensive large contexts')
                self.parent_docs = docs
            return vectorstore
        else:
            print('No collection found, now indexing documents')
            docs = self.index_docs()
            if self.config.rag_settings.multivector_enabled:
                for doc in docs:
                    if doc.metadata["char_count"] > 20000:
                        raise ValueError('Document too long, split before making child documents')
                self.parent_docs = docs
                self._set_doc_ids() # These are new doc IDs
                child_docs = self._get_child_docs()
                docs = child_docs
            if method == "chroma":
                vectorstore = get_chroma_vectorstore_from_docs(collection_name, embedder, docs)
            elif method == "faiss":
                vectorstore = get_faiss_vectorstore_from_docs(collection_name, embedder, docs)
            assert vectorstore is not None, "Vectorstore not created properly"
            return vectorstore
        
    def run_eval_tests_on_vectorstore(self, vectorstore, similarity_query: str, criteria: str = "This document is about dolphins", k_excerpts: int = 1, enable_llm_eval: bool = False):
        """
        Runs evaluation tests on the vectorstore.

        Args:
            vectorstore: The vectorstore to evaluate.
            similarity_query (str): The query to retrieve similar documents.
            criteria (str, optional): The criteria to evaluate the document. Defaults to "This document is about dolphins".
            k_excerpts (int, optional): Number of excerpts to retrieve. Defaults to 1.
            enable_llm_eval (bool, optional): Whether to enable LLM evaluation. Defaults to False.
        """
        if not criteria:
            print("Error: Criteria not provided")
            criteria = "This document is about dolphins"
        print('Running evaluation tests on vectorstore')
        filter = {}
        if self.filter_topic is not None:
            filter = {"topic": self.filter_topic}

        docs: list[Document] = vectorstore.similarity_search(similarity_query, k=k_excerpts, filter=filter)
        assert len(docs) == k_excerpts, f"Document count must be {k_excerpts}"
        print(len(docs), "documents found")

        for doc in docs:
            index = doc.metadata.get("index")
            source = doc.metadata.get("source")
            topic = doc.metadata.get("topic")
            char_count = doc.metadata.get("char_count")

            topic_string = f"(Topic: {topic})" if topic else ""
            print(f"Document {index} ({source}) (Word count: {round(char_count/5)}) {topic_string}")
            print_adjusted(doc.page_content)
            if char_count > 3000:
                print(f"Warning: Document {index} ({source}) is {char_count} chars long!")

        if not enable_llm_eval:
            return

        print('\n\nGetting evaluation from criteria!')
        eval_chain = get_eval_chain(self.rag_model.llm)
        eval_dict = {"excerpt": docs[0].page_content, "criteria": criteria}
        res = eval_chain.invoke(eval_dict)
        print(res)
        if res["meetsCriteria"] is True:
            print("Now fetching neighbor document chunk")
            index = docs[0].metadata["index"]
            new_index = index + 1 if index < len(docs) - 1 else index - 1
            temp_search_kwargs = {"k": 1, "filter": {"index": new_index}}
            new_docs: list[Document] = vectorstore.similarity_search("", search_kwargs=temp_search_kwargs)
            if new_docs:
                print_adjusted(new_docs[0].page_content)
                eval_dict["excerpt"] = new_docs[0].page_content
                new_res = eval_chain.invoke(eval_dict)
                print(new_res)
        print('yay!')

    def get_retriever(self) -> MultiVectorRetriever | VectorStoreRetriever:
        """
        Retrieves the retriever for the chatbot.

        Returns:
        - MultiVectorRetriever | VectorStoreRetriever: The retriever object.
        """
        vectorstore = self._get_vectorstore()
        # self.run_eval_tests_on_vectorstore(vectorstore)
        search_kwargs = {}
        search_kwargs["k"] = self.config.rag_settings.k_excerpts
        if self.filter_topic is not None:
            search_kwargs["filter"] = {'topic': self.filter_topic}
        # search_kwargs["filter"] = {'page': 0}
        if self.config.rag_settings.multivector_enabled:
            assert self.parent_docs, "Parent docs not initialized"
            assert self.doc_ids, "Doc IDs not initialized"
            multi_retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=InMemoryByteStore(),
                id_key=ID_KEY,
                search_kwargs=search_kwargs,
            )
            multi_retriever.docstore.mset(
                list(zip(self.doc_ids, self.parent_docs)))
            return multi_retriever
        else:
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            return retriever

    def messages_to_string(self, messages: list[SystemMessage | AIMessage | HumanMessage]):
        message_string = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                message_string += f"SYSTEM: {message.content}\n"
            elif isinstance(message, AIMessage):
                message_string += f"AI: {message.content}\n"
            elif isinstance(message, HumanMessage):
                message_string += f"HUMAN: {message.content}\n"
        return message_string

    def get_clipboard(self) -> str | None:
        return utils.get_clipboard_contents()

    def set_clipboard(self, text: str) -> None:
        return utils.copy_string_to_clipboard(text)

    def handle_command(self, prompt):
        assert prompt in COMMAND_LIST, "Invalid command"
        print("Executing command")
        if prompt == "del":
            self._pop_last_exchange()
            return
        elif prompt == "quit" or prompt == "exit":
            print('Exiting.')
            self.exit = True
            return
        elif prompt == "switch":
            if self.config.rag_settings.rag_mode:
                print("Cannot switch models in RAG mode")
                return
            assert isinstance(
                self.chat_model, LLM), "Chat model not initialized"
            if isinstance(self.backup_model, LLM):
                self.chat_model, self.backup_model = self.backup_model, self.chat_model
            elif self.backup_model is None:
                self.backup_model = self.chat_model
                try:
                    self.chat_model = LLM(
                        self.config.chat_settings.backup_model)
                    print(
                        f'Switching to backup model {self.chat_model.model_name}')
                    return
                except BaseException:
                    print('Error switching to backup model')
                    raise SystemExit
            else:
                raise ValueError("Backup model is neither None nor LLM")
        elif prompt == "refresh":
            self.refresh_config()
            return
        elif prompt == "save":
            if len(self.messages) < 2:
                print('No responses to save')
                return
            utils.save_string_as_markdown_file(self.messages[-1].content)
            # TODO: Modify this to work with RAG mode
            print('Saved response to response.md')
            return
        elif prompt == "saveall":
            message_string = self.messages_to_string(self.messages)
            utils.save_string_as_markdown_file(message_string)
            print(f'Saved {self.response_count} exchanges to history.md')
            return
        elif prompt == "info":
            print(f'RAG mode: {self.config.rag_settings.rag_mode}')
            if self.config.rag_settings.rag_mode is False:
                print(f'Exchanges: {self.response_count}')
                if self.config.chat_settings.enable_system_message:
                    print(
                        f'System message: {self.config.chat_settings.system_message}')
                print(f'LLM: {self.chat_model.model_name}')
                return
            else:
                # TODO: Make sure important fields are consistent with manifest
                assert isinstance(
                    self.rag_model, LLM), "RAG LLM not initialized"

                rag_system_message = RAG_COLLECTION_TO_SYSTEM_MESSAGE.get(
                    self.config.rag_settings.collection_name, "default")
                print(f'System message: {rag_system_message}')
                print(f'RAG LLM: {self.rag_model.model_name}')
                print(
                    f'Using vectorstore: {self.config.rag_settings.collection_name}')
                print(
                    f'Embedding model: {self.config.rag_settings.embedding_model.model_name}')
                print(f'Method: {self.config.rag_settings.method}')
                print(f'Chunk size: {self.config.rag_settings.chunk_size}')
                print(
                    f'Chunk overlap: {self.config.rag_settings.chunk_overlap}')
                print(
                    f'Excerpts in context: {self.config.rag_settings.k_excerpts}')
                if self.config.rag_settings.multivector_enabled:
                    print(
                        f'Multivector enabled! Using method: {self.config.rag_settings.multivector_method}')
                return
        elif prompt == "rag":
            if self.config.rag_settings.rag_mode:
                print(
                    'Already in RAG mode. Type reg to switch back to chat mode, or refresh to reload the configuration')
                return
            self.config.rag_settings.rag_mode = True
            self.initialize_rag()
            return
        elif prompt == "reg":
            if not self.config.rag_settings.rag_mode:
                print(
                    'Already in chat mode. Type rag to switch to RAG mode, or refresh to reload the configuration')
                return
            self.config.rag_settings.rag_mode = False
            # NOTE: Reload chat model from settings
            self.chat_model = LLM(self.config.chat_settings.primary_model)
            # TODO: Check if side effects are necessary here
            self.retriever = None
            self.rag_chain = None
            self.set_messages()
            return
        elif prompt == ".s":
            if self.config.rag_settings.rag_mode:
                print('Cannot set system message in RAG mode')
                # TODO: Maybe: Implement rag chain with custom system message.
                return
            # Print the current codes
            print(f"Available codes:")
            for k, v in SYSTEM_MESSAGE_CODES.items():
                print(f"- {k}: {v[:50]}{'[...]'if len(v)>50 else ''}")
            user_system_message = input(
                'Enter a code or type a system message: ')
            if not user_system_message:
                print('No input given, try again')
                return
            if user_system_message == "None":
                self.config.chat_settings.enable_system_message = False
                print('System message disabled')
                self.messages = []
                self.response_count = 0
                return
            # Check against SYSTEM_MESSAGE_CODES
            if user_system_message in SYSTEM_MESSAGE_CODES:
                print("System message code detected")
                user_system_message = SYSTEM_MESSAGE_CODES[user_system_message]
            self.config.chat_settings.system_message = user_system_message
            self.messages = [
                SystemMessage(
                    content=user_system_message)]
            self.response_count = 0
            print('Chat history cleared')
            return
        elif prompt == ".names":
            # Get collection names from database
            db_method = self.config.rag_settings.method
            print("Fetching collection names for method:", db_method)
            collection_names = utils.get_db_collection_names(method=db_method)
            # sort collection names
            collection_names.sort()
            print("Collection names:")
            for name in collection_names:
                print("-", name)
        elif prompt == ".copy":
            # Copy response to clipboard
            if len(self.messages) < 2:
                print('No responses to save')
            else:
                clipboard_text = self.messages[-1].content
                self.set_clipboard(clipboard_text)
            return
        elif prompt == ".eval":
            if not self.config.rag_settings.rag_mode:
                print('Must evaluate in RAG mode')
                return
            user_input = input(
                'Type a query to retrieve similar docs for: ').strip()
            if user_input:
                # criteria = input('Type a criteria to evaluate the document: ').strip()
                criteria = "The language of this document is English"
                self.run_eval_tests_on_vectorstore(
                    self.retriever.vectorstore, user_input, criteria)
            return

        elif prompt == ".rm":
            if self.config.rag_settings.rag_mode:
                print('Cannot remove messages in RAG mode')
                return
            # Allow user to choose which exchange to delete
            if len(self.messages) < 2:
                print('No messages to delete')
                return
            print('Choose an exchange to delete:')
            exchange_count = 0
            for i, message in enumerate(self.messages):
                if self.config.chat_settings.enable_system_message:
                    if i % 2 == 1:
                        exchange_count += 1
                else:
                    if i % 2 == 0:
                        exchange_count += 1
                message_suffix = "" if len(message.content) < 50 else "..."
                print(exchange_count, message.content[:50] + message_suffix)
            try:
                index = input('Enter the index of the exchange to delete: ')
                if "-" in index:
                    start, end = map(int, index.split("-"))
                    if start <= 0 or end > len(self.messages) // 2:
                        print('Invalid index range')
                        return
                    # Adjust for exchange count
                    if self.config.chat_settings.enable_system_message:
                        start = (start * 2 - 1)
                        end = (end * 2)
                    else:
                        start = (start * 2 - 2)
                        end = (end * 2 - 1)
                    for i in range(start, end + 1):
                        self.messages.pop(start)
                        self.response_count -= i % 2
                    print('Deleted exchanges')
                    return
                else:
                    index = int(index)
                    if index <= 0 or index > len(self.messages) // 2:
                        if index == 0:
                            print('Cannot delete system message')
                        print('Invalid index')
                        return
                    # Adjust for exchange count
                    if self.config.chat_settings.enable_system_message:
                        index_to_pop = (index * 2 - 1)
                    else:
                        index_to_pop = (index * 2 - 2)
                    self.messages.pop(index_to_pop)
                    self.messages.pop(index_to_pop)
                    self.response_count -= 1
                    print('Deleted exchange')
                    return
            except ValueError:
                print('Invalid input')
        else:
            print('Invalid command: ', prompt)
            return
        print("Command executed! (Return before this point.)")
        return

    def get_chat_response(self, prompt: str, stream: bool = False) -> AIMessage | None:
        """
        Gets a chat response from the chat model.

        Args:
        - prompt (str): The user's input prompt.
        - stream (bool): Whether to stream the response.

        Returns:
        - AIMessage: The AI's response message.
        """
        assert self.chat_model is not None, "Chat model not initialized"
        self.messages.append(HumanMessage(content=prompt))
        print(f'Fetching response #{self.response_count + 1}!')
        try:
            if stream:
                response_string = ""
                if self.chat_model.is_ollama:
                    for chunk in self.chat_model.stream(self.messages):
                        print(chunk, end="", flush=True)
                        response_string += chunk
                else:
                    for chunk in self.chat_model.stream(self.messages):
                        print(chunk.content, end="", flush=True)
                        response_string += chunk.content
                print()
                if not response_string:
                    raise ValueError('No response generated')
                response = AIMessage(content=response_string)
            else:
                response = self.chat_model.invoke(self.messages)
                if self.chat_model.is_ollama:
                    assert isinstance(response, str), "Response not str"
                    print_adjusted(response)
                    response = AIMessage(content=response)
                else:
                    assert isinstance(response, AIMessage), "Response not AIMessage"
                    print_adjusted(response.content)
        except KeyboardInterrupt:
            print('Keyboard interrupt, aborting generation.')
            self.messages.pop()
            return None
        except Exception as e:
            print(f'Error!: {e}')
            self.messages.pop()
            return None
        self.messages.append(response)
        self.response_count += 1
        return response

    def get_rag_response(self, prompt: str, stream: bool = False):
        assert self.rag_model is not None, "RAG LLM not initialized"
        assert self.rag_chain is not None, "RAG chain not initialized"
        self.messages.append(HumanMessage(content=prompt))
        print(f'RAG engine response #{self.response_count + 1}!')
        try:
            if stream:
                response_string = ""
                if self.rag_model.is_ollama:
                    for chunk in self.rag_chain.stream(prompt):
                        print(chunk, end="", flush=True)
                        response_string += chunk
                else:
                    for chunk in self.rag_chain.stream(prompt):
                        print(chunk.content, end="", flush=True)
                        response_string += chunk.content
                print()
                response = AIMessage(content=response_string)
            else:
                response = self.rag_chain.invoke(prompt)
                if self.rag_model.is_ollama:
                    assert isinstance(response, str), "Response not str"
                    print_adjusted(response)
                    response = AIMessage(content=response)
                else:
                    assert isinstance(
                        response, AIMessage), "Response not AIMessage"
                    print_adjusted(response.content)

        except KeyboardInterrupt:
            print('Keyboard interrupt, aborting generation.')
            return None
        except Exception as e:
            print(f'Error!: {e}')
            return None
        self.messages.append(response)
        self.response_count += 1
        return response

    def chat(self, prompt=None, persistence_enabled=True):
        """
        Main chat loop for the chatbot.

        Args:
        - prompt (str, optional): Initial prompt for the chatbot.
        - persistence_enabled (bool): Whether to enable persistent chat mode.

        Returns:
        - list: The message history.
        """
        force_prompt = False
        forced_prompt = ""
        save_response = False
        max_responses = MAX_CHAT_EXCHANGES
        if prompt is not None:
            force_prompt = True
            forced_prompt = prompt
        if persistence_enabled is False:
            if prompt is None:
                prompt = DEFAULT_QUERY
            max_responses = 1
            if SAVE_ONESHOT_RESPONSE:
                save_response = True
        while self.exit is False:
            if self.response_count >= max_responses:
                print(f'Max exchanges reached: {self.response_count}')
                self.exit = True
                continue
            if force_prompt:
                prompt = forced_prompt
                force_prompt = False
                forced_prompt = ""
            else:
                prompt = input("Enter your query: ")
            if prompt in REFORMATTING_PROMPTS:
                if prompt == ".paste":
                    # use clipboard content as prompt
                    prompt = self.get_clipboard()
                    if not prompt:
                        print('No text in clipboard! Try again.')
                        continue
                elif prompt == ".read":
                    # Read sample.txt as prompt
                    # TODO: Add some checks for string content of sample.txt
                    prompt = utils.read_sample()
                else:
                    print('Reformatting command not yet implemented in .chat method')
                # Do not continue here
            stripped_prompt = prompt.strip()
            if not stripped_prompt:
                print('No input given, try again')
                continue
            if len(prompt) > MAX_CHARS_IN_PROMPT:
                print(
                    f'Input too long, max characters is {MAX_CHARS_IN_PROMPT}')
                continue

            if stripped_prompt in COMMAND_LIST:
                self.handle_command(stripped_prompt)
                continue
            # Generate response
            if self.config.rag_settings.rag_mode:
                self.get_rag_response(prompt, stream=True)
            else:
                res = self.get_chat_response(prompt, stream=True)
                if res is None:
                    print('No response generated')
                    continue
        if save_response:
            utils.save_string_as_markdown_file(self.messages[-1].content)
            print('Saved response to response.md')
        return self.messages

    def _pop_last_exchange(self) -> None:
        if len(self.messages) < 2:
            print('No messages to delete')
            return
        self.messages.pop()
        self.messages.pop()
        print('Deleted last exchange')
        self.response_count -= 1


def run_chat(config: Config | None = None):
    chatbot = Chatbot(config)
    chatbot.chat()


def main():
    run_chat()


def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive chat')
    parser.add_argument(
        'prompt',
        type=str,
        nargs='?',
        help='Prompt for the LLM')
    parser.add_argument(
        '-np',
        '--not-persistent',
        action='store_true',
        help='Disable persistent chat mode')
    parser.add_argument(
        '-rag',
        '--rag-mode',
        action='store_true',
        help='Enable RAG mode')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='Specify a model for chat')
    parser.add_argument(
        '-c',
        '--collection',
        type=str,
        help='Specify a collection name for RAG mode')

    # Add -i flag for making inputs in the form of a list[str]
    parser.add_argument(
        '-i',
        '--inputs',
        nargs='+',
        help='List of inputs for RAG mode')
    args = parser.parse_args()

    config_override = {}
    # How can I make sure this typechecks correctly?

    config_override["RAG"] = {}
    config_override["chat"] = {}

    if args.collection:
        args.rag_mode = True
        config_override["RAG"]["collection_name"] = args.collection

    if args.inputs:
        print('Found inputs. RAG mode enabled')
        assert isinstance(args.inputs, list)
        # assert all(isinstance(i, str) for i in args.inputs)
        args.rag_mode = True
        config_override["RAG"]["inputs"] = args.inputs

    if args.rag_mode:
        # Currently no way to disable rag mode from CLI if True in
        # settings.json
        config_override["RAG"]["rag_mode"] = args.rag_mode

    if args.model:
        if args.rag_mode:
            config_override["RAG"]["rag_llm"] = args.model
        else:
            config_override["chat"]["primary_model"] = args.model

    config = Config(config_override=config_override)
    prompt = args.prompt

    persistence_enabled = not args.not_persistent
    if prompt is None and persistence_enabled is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = utils.read_sample()
            if EXPLAIN_EXCERPT:
                excerpt_as_prompt = SUMMARY_TEMPLATE.format(
                    excerpt=excerpt_as_prompt)
            prompt = excerpt_as_prompt

    try:
        chatbot = Chatbot(config)
        chatbot.chat(prompt, persistence_enabled=persistence_enabled)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting.')
        raise SystemExit


if __name__ == "__main__":
    main_cli()