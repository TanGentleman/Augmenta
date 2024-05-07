from uuid import uuid4
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from helpers import database_exists, get_db_collection_names, process_docs, get_doc_ids_from_manifest, save_response_to_markdown_file, save_history_to_markdown_file, read_sample, update_manifest
from constants import DEFAULT_QUERY, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES, PROMPT_CHOOSER_SYSTEM_MESSAGE, RAG_COLLECTION_TO_SYSTEM_MESSAGE, SUMMARY_TEMPLATE, SYSTEM_MESSAGE_CODES
from config import MAX_CHARACTERS_IN_PARENT_DOC, MAX_PARENT_DOCS, SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE, EXPLAIN_EXCERPT, FILTER_TOPIC, ALLOW_MULTI_VECTOR
from classes import Config
from models import LLM_FN, LLM
from rag import get_summary_chain, input_to_docs, get_rag_chain, get_eval_chain
from embed import get_chroma_vectorstore_from_docs, get_faiss_vectorstore_from_docs, load_existing_faiss_vectorstore, load_existing_chroma_vectorstore, split_documents
from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore
from os import get_terminal_size
from textwrap import fill
try:
    import gnureadline
except ImportError:
    pass

if ALLOW_MULTI_VECTOR:
    from langchain.retrievers.multi_vector import MultiVectorRetriever
else:
    MultiVectorRetriever = None

try:
    from pyperclip import paste as clipboard_paste, copy as clipboard_copy
    CLIPBOARD_ACCESS_ENABLED = True
except ImportError:
    print('pyperclip not installed, clipboard access disabled')
    CLIPBOARD_ACCESS_ENABLED = False


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


PROCESSING_DOCS_FN = process_docs

ID_KEY = "doc_id"

REFORMATTING_PROMPTS = ["paste", "read"]
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
    ".sr",
    ".rm"]


class Chatbot:
    """
    This class is an interactive RAG-capable chatbot.

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
    - chat: The main chat loop. It's as simple as `from chat import Chatbot; Chatbot().chat()`
    """

    def __init__(self, config=None):
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
        self.count = 0
        self.exit = False
        self.messages = []

        if self.config.rag_settings.rag_mode:
            self.ingest_documents()
        else:
            self.chat_model = LLM(self.config.chat_settings.primary_model)
            self.set_starting_messages()

    def set_starting_messages(self):
        messages = []
        if self.config.rag_settings.rag_mode:
            messages = []
        else:
            if self.config.chat_settings.enable_system_message:
                messages.append(
                    SystemMessage(
                        content=self.config.chat_settings.system_message))
            else:
                print('System message is disabled')
        self.messages = messages
        self.count = 0

    def ingest_documents(self):
        """
        Initializes the retriever and RAG chain.

        Raises:
            ValueError: Doc ID not initialized
        """
        if self.retriever is not None:
            print('Retriever already exists. Use "refresh" to clear it first')
            return
        assert self.config.rag_settings.rag_mode, "RAG mode not enabled"
        self.rag_model = LLM(self.config.rag_settings.rag_llm)
        # get doc_ids
        if self.config.rag_settings.multivector_enabled:
            doc_ids = get_doc_ids_from_manifest(
                self.config.rag_settings.collection_name)
            if self.config.rag_settings.database_exists and not doc_ids:
                raise ValueError("Doc IDs not initialized")
            self.doc_ids = doc_ids
        self.retriever = self.get_retriever()
        collection_code = RAG_COLLECTION_TO_SYSTEM_MESSAGE.get(
            self.config.rag_settings.collection_name, "default")
        rag_system_message = RAG_COLLECTION_TO_SYSTEM_MESSAGE[collection_code]
        self.rag_chain = get_rag_chain(
            self.retriever,
            self.rag_model.llm,
            system_message=rag_system_message)

        self.set_starting_messages()
        update_manifest(
            embedding_model_name=self.config.rag_settings.embedding_model.model_name,
            method=self.config.rag_settings.method,
            chunk_size=self.config.rag_settings.chunk_size,
            chunk_overlap=self.config.rag_settings.chunk_overlap,
            inputs=self.config.rag_settings.inputs,
            # NOTE: These values can be wrong if a db is loaded after manifest was reset
            # Afaik, there's no decent way to solve this, but it wouldn't affect a stable version
            collection_name=self.config.rag_settings.collection_name,
            doc_ids=self.doc_ids)
        # Update active json file
        self.config.save_to_json()

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
            config_override["rag_mode"] = self.config.rag_settings.rag_mode
            config = Config(config_override=config_override)
        self.config = config
        self.backup_model = None
        self.retriever = None
        self.rag_chain = None
        self.doc_ids = []
        self.parent_docs = None

        if self.config.rag_settings.rag_mode:
            # NOTE: I may have to set self.chat_model = None here. Not sure if
            # it's necessary.
            self.ingest_documents()
            self.chat_model = None
            self.backup_model = None
        else:
            self.chat_model = LLM(self.config.chat_settings.primary_model)
            self.backup_model = None
            self.set_starting_messages()
            self.count = 0

    def set_doc_ids(self):
        assert self.config.rag_settings.rag_mode, "RAG mode must be on"
        assert self.config.rag_settings.multivector_enabled, "Multivector not enabled"
        assert self.parent_docs, "Parent docs not initialized"

        if self.doc_ids:
            print("Doc IDs already initialized")
        else:
            self.doc_ids = [str(uuid4()) for _ in self.parent_docs]
        assert self.doc_ids, "Doc IDs not created"

    def get_child_docs(self):
        # This function has limits for doc count/size set in constants.py
        assert self.config.rag_settings.multivector_enabled, "Multivector not enabled"
        assert isinstance(
            self.rag_model, LLM), "RAG LLM not initialized"
        # When qa is supported, this will check method
        assert self.parent_docs and len(
            self.parent_docs) < MAX_PARENT_DOCS, "Temporary limit of 8 parent Documents"
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

    def get_vectorstore(
            self,
            processing_docs_fn=PROCESSING_DOCS_FN) -> Chroma | FAISS:
        # The processing_docs_fn can be used to format or clean up the
        # documents before indexing.
        assert self.config.rag_settings.rag_mode, "RAG mode not enabled"
        collection_name = self.config.rag_settings.collection_name
        method = self.config.rag_settings.method
        embedding_model_fn = self.config.rag_settings.embedding_model
        assert isinstance(embedding_model_fn, LLM_FN)
        # This is used instead of the LLM object
        embedder = embedding_model_fn.get_llm()

        vectorstore = None
        inputs = self.config.rag_settings.inputs
        docs = []
        # If collection exists, load it
        if database_exists(collection_name, method):
            print(f"Loading existing Vector DB: {collection_name}")
            if method == "chroma":
                vectorstore = load_existing_chroma_vectorstore(
                    collection_name, embedder)
            elif method == "faiss":
                vectorstore = load_existing_faiss_vectorstore(
                    collection_name, embedder)
            assert vectorstore is not None, "Collection exists but not loaded properly"
            if self.config.rag_settings.multivector_enabled:
                for i in range(len(inputs)):
                    if not inputs[i]:
                        print(f'Input {i} is empty, skipping')
                        continue
                    docs.extend(input_to_docs(inputs[i]))
                assert docs, "No documents to make parent docs"
                docs = split_documents(docs,
                                       self.config.rag_settings.chunk_size,
                                       self.config.rag_settings.chunk_overlap)
                # TODO: assertion to make sure parent docs are
                # correctly formed
                self.parent_docs = docs
                assert len(
                    self.parent_docs) == len(
                    self.doc_ids), "Parent docs and doc IDs do not match length"
            return vectorstore
        else:
            # Ingest documents
            print('No collection found, now ingesting documents')
            for i in range(len(inputs)):
                # In the future this can be parallelized
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
            if RAG_COLLECTION_TO_SYSTEM_MESSAGE.get(
                    collection_name) == PROMPT_CHOOSER_SYSTEM_MESSAGE:
                # This is for indexing anthropic url prompt library data
                if processing_docs_fn:
                    print("Prompt chooser detected, processing documents...")
                    docs = processing_docs_fn(docs)
            docs = split_documents(docs,
                                   self.config.rag_settings.chunk_size,
                                   self.config.rag_settings.chunk_overlap)
            if self.config.rag_settings.multivector_enabled:
                # Make sure the parent docs aren't too wordy
                for doc in docs:
                    # This number is arbitrary for now
                    if doc.metadata["char_count"] > 20000:
                        raise ValueError(
                            'Document too long, split before making child documents')

                self.parent_docs = docs
                # Add child docs to vectorstore instead
                self.set_doc_ids()
                docs = self.get_child_docs()
            if method == "chroma":
                vectorstore = get_chroma_vectorstore_from_docs(
                    collection_name, embedder, docs)
            elif method == "faiss":
                vectorstore = get_faiss_vectorstore_from_docs(
                    collection_name, embedder, docs)
            assert vectorstore is not None, "Vectorstore not created properly"
            return vectorstore

    def run_eval_tests_on_vectorstore(self, vectorstore):
        # This function is for testing purposes
        print('Running evaluation tests on vectorstore')
        EVAL_QUERY = "What animal is this about?"
        EVAL_K_EXCERPTS = 1
        filter = {}
        if self.filter_topic is not None:
            filter = {"topic": self.filter_topic}

        # This is a test for similarity search
        docs = vectorstore.similarity_search(
            EVAL_QUERY, k=EVAL_K_EXCERPTS, filter=filter)
        # print(docs[0])
        # There should only be one document
        assert len(docs) == 1, "There should be exactly one document"
        print(len(docs), "documents found")
        # Optional scan through documents
        for doc in docs:
            index = doc.metadata.get("index")
            source = doc.metadata.get("source")
            topic = doc.metadata.get("topic")
            char_count = doc.metadata.get("char_count")
            print(
                f"Document {index} ({source}) ({topic}) ({char_count} chars long)")
            print_adjusted(doc.page_content)
            print('\n\n')
            if char_count > 2000:
                print(
                    f"Warning: Document {index} ({source}) is {char_count} chars long!")
        # This is a test for evaluation
        assert isinstance(self.rag_model, LLM), "RAG LLM not initialized"
        eval_chain = get_eval_chain(self.rag_model.llm)
        eval_dict = {
            "excerpt": docs[0].page_content,
            "criteria": "The topic of this excerpt is dolphins."}
        res = eval_chain.invoke(eval_dict)
        print(res)
        if res["meetsCriteria"] is True:
            print("Now fetching related documents")
            index = docs[0].metadata["index"]
            if index > 0:
                new_index = index + 1
            else:
                new_index = index + 1
            temp_search_kwargs = {"k": 1, "filter": {"index": new_index}}
            new_docs = vectorstore.similarity_search(
                "", search_kwargs=temp_search_kwargs)
            if new_docs:
                print_adjusted(new_docs[0].page_content)
                eval_dict["excerpt"] = new_docs[0].page_content
                new_res = eval_chain.invoke(eval_dict)
                print(new_res)
        print('yay!')
        raise SystemExit("Test complete")

    def get_retriever(self) -> MultiVectorRetriever | VectorStoreRetriever:  # type: ignore
        vectorstore = self.get_vectorstore()
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
            )
            multi_retriever.docstore.mset(
                list(zip(self.doc_ids, self.parent_docs)))
            return multi_retriever
        else:
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            return retriever

    def messages_to_strings(self, messages):
        messages = [msg.type.upper() + ": " + msg.content for msg in messages]
        return messages

    def get_clipboard(self) -> str | None:
        if CLIPBOARD_ACCESS_ENABLED:
            prompt = clipboard_paste().strip()
            return prompt
        return None

    def set_clipboard(self, text: str) -> None:
        if CLIPBOARD_ACCESS_ENABLED:
            clipboard_copy(text)
            print('Copied response to clipboard')
        else:
            print('Clipboard access disabled')

    def handle_command(self, prompt):
        assert prompt in COMMAND_LIST, "Invalid command"
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
            save_response_to_markdown_file(self.messages[-1].content)
            # TODO: Modify this to work with RAG mode
            print('Saved response to response.md')
            return
        elif prompt == "saveall":
            message_strings = self.messages_to_strings(self.messages)
            save_history_to_markdown_file(message_strings)
            print(f'Saved {self.count} exchanges to history.md')
            for message in message_strings:
                print_adjusted(f"{message}\n\n")
            return
        elif prompt == "info":
            print(f'RAG mode: {self.config.rag_settings.rag_mode}')
            if self.config.rag_settings.rag_mode is False:
                print(f'Exchanges: {self.count}')
                if self.config.chat_settings.enable_system_message:
                    print(
                        f'System message: {self.config.chat_settings.system_message}')
                print(f'LLM: {self.chat_model.model_name}')
                return
            else:
                # TODO: Make sure important fields are consistent with manifest
                assert isinstance(
                    self.rag_model, LLM), "RAG LLM not initialized"

                collection_code = RAG_COLLECTION_TO_SYSTEM_MESSAGE.get(
                    self.config.rag_settings.collection_name)
                if collection_code is None:
                    collection_code = "default"
                rag_system_message = RAG_COLLECTION_TO_SYSTEM_MESSAGE[collection_code]
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
            self.ingest_documents()
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
            self.set_starting_messages()
            return
        elif prompt == ".s":
            if self.config.rag_settings.rag_mode:
                print('Cannot set system message in RAG mode')
                # TODO: Maybe: Implement rag chain with custom system message.
                return
            # Print the current codes
            print(f"Available codes:")
            for k, v in SYSTEM_MESSAGE_CODES.items():
                print(f"- {k}: {v[:50]}[...]")
            user_system_message = input(
                'Enter a code or type a system message: ')
            if not user_system_message:
                print('No input given, try again')
                return
            # Check against SYSTEM_MESSAGE_CODES
            if user_system_message in SYSTEM_MESSAGE_CODES:
                print("System message code detected")
                user_system_message = SYSTEM_MESSAGE_CODES[user_system_message]
            self.config.chat_settings.system_message = user_system_message
            self.messages = [
                SystemMessage(
                    content=user_system_message)]
            self.count = 0
            print('Chat history cleared')
            return
        elif prompt == ".names":
            # Get collection names from database
            db_method = self.config.rag_settings.method
            print("Fetching collection names for method:", db_method)
            collection_names = get_db_collection_names(method=db_method)
            # sort collection names
            collection_names.sort()
            print("Collection names:")
            for name in collection_names:
                print("-", name)
        elif prompt == ".sr":
            # Save response to clipboard
            if len(self.messages) < 2:
                print('No responses to save')
            else:
                clipboard_text = self.messages[-1].content
                self.set_clipboard(clipboard_text)
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
                print(exchange_count, message.content[:50])
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
                        self.count -= i % 2
                    print('Deleted exchanges')
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
                    self.count -= 1
                    print('Deleted exchange')
            except ValueError:
                print('Invalid input')
        else:
            print('Invalid command: ', prompt)
            return

    def get_chat_response(self, prompt: str, stream: bool = False):
        assert self.chat_model is not None, "Chat model not initialized"
        self.messages.append(HumanMessage(content=prompt))
        print(f'Fetching response #{self.count + 1}!')
        try:
            if stream:
                response_string = ""
                if self.chat_model.model_name == "local-ollama3":
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
                if self.chat_model.model_name == "local-ollama3":
                    assert isinstance(response, str), "Response not str"
                    print_adjusted(response)
                    response = AIMessage(content=response)
                else:
                    assert isinstance(
                        response, AIMessage), "Response not AIMessage"
                    print_adjusted(response.content)
        except KeyboardInterrupt:
            print('Keyboard interrupt, aborting generation.')
            self.messages.pop()
            return
        except Exception as e:
            print(f'Error!: {e}')
            self.messages.pop()
            return
        self.messages.append(response)
        self.count += 1
        return response

    def get_rag_response(self, prompt: str, stream: bool = False):
        assert self.rag_model is not None, "RAG LLM not initialized"
        assert self.rag_chain is not None, "RAG chain not initialized"
        self.messages.append(HumanMessage(content=prompt))
        print(f'RAG engine response #{self.count + 1}!')
        try:
            if stream:
                response_string = ""
                if self.rag_model.model_name == "local-ollama3":
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
                if self.rag_model.model_name == "local-ollama3":
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
        self.count += 1
        return response

    def chat(self, prompt=None, persistence_enabled=True):
        force_prompt = False
        forced_prompt = ""
        save_response = False
        max_exchanges = MAX_CHAT_EXCHANGES
        if prompt is not None:
            force_prompt = True
            forced_prompt = prompt
        if persistence_enabled is False:
            if prompt is None:
                prompt = DEFAULT_QUERY
            max_exchanges = 1
            if SAVE_ONESHOT_RESPONSE:
                save_response = True
        while self.exit is False:
            if self.count >= max_exchanges:
                print(f'Max exchanges reached: {self.count}')
                self.exit = True
                continue
            if force_prompt:
                prompt = forced_prompt
                force_prompt = False
                forced_prompt = ""
            else:
                prompt = input("Enter your query: ")
            if prompt in REFORMATTING_PROMPTS:
                if prompt == "paste":
                    # use clipboard content as prompt
                    prompt = self.get_clipboard()
                    if not prompt:
                        print('No text in clipboard! Try again.')
                        continue
                elif prompt == "read":
                    # Read sample.txt as prompt
                    # TODO: Add some checks for string content of sample.txt
                    prompt = read_sample()
                # Do not continue here

            if not prompt.strip():
                print('No input given, try again')
                continue
            if len(prompt) > MAX_CHARS_IN_PROMPT:
                print(
                    f'Input too long, max characters is {MAX_CHARS_IN_PROMPT}')
                continue

            if prompt in COMMAND_LIST:
                self.handle_command(prompt)
                continue
            # Generate response
            if self.config.rag_settings.rag_mode:
                self.get_rag_response(prompt, stream=True)
            else:
                self.get_chat_response(prompt, stream=True)
        if save_response:
            save_response_to_markdown_file(self.messages[-1].content)
            print('Saved response to response.md')

    def _pop_last_exchange(self):
        if len(self.messages) < 2:
            print('No messages to delete')
            return
        self.messages.pop()
        self.messages.pop()
        print('Deleted last exchange')
        self.count -= 1


# Argparse implementation
if __name__ == "__main__":
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

    # Add -i flag for making inputs in the form of a list[str]
    parser.add_argument(
        '-i',
        '--inputs',
        nargs='+',
        help='List of inputs for RAG mode')
    args = parser.parse_args()

    config_override = {}
    # How can I make sure this typechecks correctly?

    config_override["rag_config"] = {}
    config_override["chat_config"] = {}

    if args.inputs:
        print('Found inputs. RAG mode enabled')
        assert isinstance(args.inputs, list)
        # assert all(isinstance(i, str) for i in args.inputs)
        args.rag_mode = True
        config_override["rag_config"]["inputs"] = args.inputs

    if args.rag_mode:
        # This means there is no way to disable rag mode from CLI if set in
        # settings.json
        config_override["rag_config"]["rag_mode"] = args.rag_mode

    if args.model:
        if args.rag_mode:
            config_override["rag_config"]["rag_llm"] = args.model
        else:
            config_override["chat_config"]["primary_model"] = args.model

    config = Config(config_override=config_override)
    if config.rag_settings.rag_mode and config.rag_settings.multivector_enabled is True:
        if MultiVectorRetriever is None:
            print('MultiVectorRetriever not supported. Check config.py and chat.py.')
            raise SystemExit
    prompt = args.prompt

    persistence_enabled = not args.not_persistent
    if prompt is None and persistence_enabled is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = read_sample()
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
