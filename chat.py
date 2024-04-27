from typing import Any
from uuid import uuid4
from langchain.schema import HumanMessage, SystemMessage
from helpers import database_exists, process_docs, scan_manifest, save_response_to_markdown_file, save_history_to_markdown_file, read_sample, update_manifest
from constants import DEFAULT_QUERY, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES, SUMMARY_TEMPLATE
from config import MAX_CHARACTERS_IN_PARENT_DOC, MAX_PARENT_DOCS, SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE, EXPLAIN_EXCERPT
from classes import Config
from models import MODEL_DICT, LLM_FN, LLM
from rag import get_summary_chain, input_to_docs, get_rag_chain
from embed import chroma_vectorstore_from_docs, faiss_vectorstore_from_docs, load_faiss_vectorstore, load_chroma_vectorstore, split_documents
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

PROCESSING_DOCS_FN = None
# PROCESSING_DOCS_FN = process_docs

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
    "reg"]


class Chatbot:
    """
    This class is an interactive RAG-capable chatbot.

    Parameters:
    - config (Config, optional): The configuration object. Defaults to None.

    Attributes:
    - config (Config): The configuration object.
    - settings (dict): The chat settings. See classes.ChatSchema
    - rag_settings (dict): The RAG settings. See classes.RAGSchema
    - chat_model (LLM): The chat model.
    - backup_model (LLM): The backup model.
    - parent_docs (list[Document]): The parent documents.
    - doc_ids (list[str]): The document IDs.
    - rag_chain (Runnable): The RAG chain.
    - retriever (MultiVectorRetriever): The retriever.
    - count (int): How many exchanges (back and forth messages) in history.
    - rag_mode (bool): If RAG mode is enabled.
    - exit (bool): The exit flag for the chat loop.
    - messages: The message history used in context for non-RAG chat.

    Methods:
    - chat: The main chat loop. It's as simple as `from chat import Chatbot; Chatbot().chat()`
    """

    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.config = config
        self.settings = self.get_chat_settings()
        self.rag_settings = self.get_rag_settings()
        # TODO: Disable this activation when RAG is enabled. This means forcing it when refreshing
        self.chat_model = None
        self.backup_model = self.settings["backup_model"]

        self.parent_docs = None
        self.doc_ids = []
        self.rag_chain = None
        self.retriever = None
        self.count = 0
        self.rag_mode = self.settings["rag_mode"]
        self.exit = False
        self.messages = []

        if self.rag_mode:
            self.ingest_documents()
        else:
            self.chat_model = self.activate_chat_model() # This activates the primary model
            self.initialize_messages()

    def activate_chat_model(self, backup=False) -> LLM:
        if backup:
            llm_fn = self.settings["backup_model"]
        else:
            llm_fn = self.settings["primary_model"]

        if isinstance(llm_fn, LLM):
            print('Model already initialized')
            return llm_fn
        assert isinstance(llm_fn, LLM_FN)
        return LLM(llm_fn)

    def get_rag_model(self) -> LLM:
        if isinstance(self.rag_settings["rag_llm"], LLM):
            print('RAG LLM already initialized')
            return self.rag_settings["rag_llm"]
        llm_fn = self.rag_settings["rag_llm"]
        assert isinstance(llm_fn, LLM_FN)
        return LLM(llm_fn)

    def initialize_messages(self):
        messages = []
        if self.rag_mode:
            messages = []
        else:
            if self.settings["enable_system_message"]:
                messages.append(
                    SystemMessage(
                        content=self.settings["system_message"]))
            else:
                print('System message is disabled')
        self.messages = messages
        self.count = 0

    def ingest_documents(self):
        """
        This function performs the initial RAG steps
        """
        if self.retriever is not None:
            print('Retriever already exists. Use "refresh" to clear it first')
            return

        # From this point forward, the rag_llm is of type LLM
        self.rag_settings["rag_llm"] = self.get_rag_model()
        assert isinstance(
            self.rag_settings["rag_llm"], LLM), "RAG LLM not initialized"
        self.rag_mode = True
        # get doc_ids
        if self.rag_settings["multivector_enabled"]:
            self.doc_ids = scan_manifest(self.rag_settings)
        self.retriever = self.get_retriever()
        self.rag_chain = get_rag_chain(
            self.retriever, self.rag_settings["rag_llm"].llm)

        self.initialize_messages()
        if self.rag_settings["multivector_enabled"]:
            if not self.doc_ids:
                raise ValueError("Doc IDs not initialized")
        update_manifest(self.rag_settings, self.doc_ids)

    def refresh_config(self, config: Config = None):
        if config is None:
            # Reload config from settings.json
            config = Config()
        self.config = config
        self.settings = self.get_chat_settings()
        self.rag_settings = self.get_rag_settings()

        self.rag_mode = self.settings["rag_mode"]
        self.backup_model = self.settings["backup_model"]
        self.retriever = None
        self.rag_chain = None
        self.doc_ids = []
        self.parent_docs = None

        if self.rag_mode:
            self.ingest_documents()
        else:
            self.chat_model = self.activate_chat_model()
            self.initialize_messages()
            self.count = 0

    def get_chat_settings(self):
        primary_model = self.config.chat_config["primary_model"]
        backup_model = self.config.chat_config["backup_model"]
        enable_system_message = self.config.chat_config["enable_system_message"]
        system_message = self.config.chat_config["system_message"]
        rag_mode = self.config.chat_config["rag_mode"]
        assert isinstance(primary_model, LLM_FN)
        assert isinstance(backup_model, LLM_FN)
        assert isinstance(enable_system_message, bool)
        assert isinstance(system_message, str)
        assert isinstance(rag_mode, bool)

        chat_settings = {
            "primary_model": primary_model,
            "backup_model": backup_model,
            "enable_system_message": enable_system_message,
            "system_message": system_message,
            "rag_mode": rag_mode}
        return chat_settings

    def get_rag_settings(self):
        collection_name = self.config.rag_config["collection_name"]
        embedding_model = self.config.rag_config["embedding_model"]
        assert isinstance(embedding_model, LLM_FN)
        # Initialize the embedder
        method = self.config.rag_config["method"]
        chunk_size = self.config.rag_config["chunk_size"]
        chunk_overlap = self.config.rag_config["chunk_overlap"]
        k_excerpts = self.config.rag_config["k_excerpts"]
        rag_llm = self.config.rag_config["rag_llm"]
        inputs = self.config.rag_config["inputs"]
        # Not sure if I want to keep these
        multivector_enabled = self.config.rag_config["multivector_enabled"]
        multivector_method = self.config.rag_config["multivector_method"]
        # TODO: Add doc_ids to rag_settings. This will be used for multivector.

        assert isinstance(collection_name, str)
        assert isinstance(method, str)
        assert isinstance(chunk_size, int)
        assert isinstance(chunk_overlap, int)
        assert isinstance(k_excerpts, int)
        assert isinstance(rag_llm, LLM_FN), "RAG LLM not initialized"
        assert isinstance(inputs, list)

        rag_settings = {
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "method": method,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k_excerpts": k_excerpts,
            "rag_llm": rag_llm,
            "inputs": inputs,
            "multivector_enabled": multivector_enabled,
            "multivector_method": multivector_method
        }
        return rag_settings

    def set_doc_ids(self):
        assert self.rag_settings["multivector_enabled"], "Multivector not enabled"
        assert self.parent_docs, "Parent docs not initialized"

        if self.doc_ids:
            print("Doc IDs already initialized")
        else:
            self.doc_ids = [str(uuid4()) for _ in self.parent_docs]
        assert self.doc_ids, "Doc IDs not created"

    def get_child_docs(self):
        # This function has limits for doc count/size set in constants.py
        assert self.rag_settings["multivector_enabled"], "Multivector not enabled"
        assert isinstance(
            self.rag_settings["rag_llm"], LLM), "RAG LLM not initialized"
        # When qa is supported, this will check
        # self.rag_settings["multivector_method"]
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
        summarize_chain = get_summary_chain(self.rag_settings["rag_llm"].llm)
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

    def get_vectorstore(self, processing_docs_fn: callable = PROCESSING_DOCS_FN):
        assert self.rag_mode, "RAG mode not enabled"
        collection_name = self.rag_settings["collection_name"]
        method = self.rag_settings["method"]
        embedding_model_fn = self.rag_settings["embedding_model"]
        assert isinstance(embedding_model_fn, LLM_FN)
        embedder = embedding_model_fn.get_llm()

        vectorstore = None
        inputs = self.rag_settings["inputs"]
        docs = []
        # If collection exists, load it
        if database_exists(collection_name, method):
            print(f"Collection {collection_name} exists, now loading")
            if method == "chroma":
                vectorstore = load_chroma_vectorstore(
                    collection_name, embedder)
            elif method == "faiss":
                vectorstore = load_faiss_vectorstore(
                    collection_name, embedder)
            assert vectorstore is not None, "Collection exists but not loaded properly"
            if self.rag_settings["multivector_enabled"]:
                for i in range(len(inputs)):
                    if not inputs[i]:
                        print(f'Input {i} is empty, skipping')
                        continue
                    docs.extend(input_to_docs(self.rag_settings["inputs"][i]))
                assert docs, "No documents to make parent docs"
                docs = split_documents(docs,
                                       self.rag_settings["chunk_size"],
                                       self.rag_settings["chunk_overlap"])
                # There should be an assertion to make sure parent docs are
                # correctly formed
                self.parent_docs = docs
                assert len(
                    self.parent_docs) == len(
                    self.doc_ids), "Parent docs and doc IDs do not match length"
            return vectorstore

        # Ingest documents
        for i in range(len(inputs)):
            # In the future this can be parallelized
            if not inputs[i]:
                print(f'Input {i} is empty, skipping')
                continue
            docs.extend(input_to_docs(inputs[i]))

        assert docs, "No documents to create collection"
        if processing_docs_fn:
                    docs = processing_docs_fn(docs)
        docs = split_documents(docs,
                               self.rag_settings["chunk_size"],
                               self.rag_settings["chunk_overlap"])
        if self.rag_settings["multivector_enabled"]:
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
            vectorstore = chroma_vectorstore_from_docs(
                collection_name, embedder, docs)
        elif method == "faiss":
            vectorstore = faiss_vectorstore_from_docs(
                collection_name, embedder, docs)
        assert vectorstore is not None, "Vectorstore not created properly"
        return vectorstore

    def get_retriever(self):
        # try:
        vectorstore = self.get_vectorstore()
        # except Exception as e:
        #     print(f'Error: {e}\n')
        #     print(f'Error creating vectorstore, check RAG settings in settings.json!')
        #     raise SystemExit
        search_kwargs = {}
        search_kwargs["k"] = self.rag_settings["k_excerpts"]
        # search_kwargs["filter"] = FILTERED_TAGS # Not yet implemented
        if self.rag_settings["multivector_enabled"]:
            assert self.parent_docs, "Parent docs not initialized"
            assert self.doc_ids, "Doc IDs not initialized"
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=InMemoryByteStore(),
                id_key=ID_KEY,
            )
            retriever.docstore.mset(list(zip(self.doc_ids, self.parent_docs)))
        else:
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        return retriever

    def messages_to_strings(self, messages):
        messages = [msg.type.upper() + ": " + msg.content for msg in messages]
        return messages

    def prompt_from_clipboard(self):
        try:
            from pyperclip import paste
        except ImportError:
            print('pyperclip not installed, try pip install pyperclip')
            return None
        prompt = paste().strip()
        return prompt

    def command_handler(self, prompt):
        assert prompt in COMMAND_LIST, "Invalid command"
        if prompt == "del":
            if len(self.messages) < 2:
                print('No messages to delete')
                return
            self.messages.pop()
            self.messages.pop()
            print('Deleted last exchange')
            self.count -= 1
            return
        elif prompt == "quit" or prompt == "exit":
            print('Exiting.')
            self.exit = True
            return
        elif prompt == "switch":
            if self.rag_mode:
                print("Cannot switch models in RAG mode")
                return
            self.backup_model = self.chat_model
            print(f'Switching to backup model {self.backup_model.model_name}')
            try:
                self.chat_model = self.activate_chat_model(backup=True)
                return
            except BaseException:
                print('Error switching to backup model')
                raise SystemExit
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
            return
        elif prompt == "info":
            print(f'RAG mode: {self.rag_mode}')
            if self.rag_mode is False:
                print(f'Exchanges: {self.count}')
                if self.settings["enable_system_message"]:
                    print(f'System message: {self.settings["system_message"]}')
                print(f'LLM: {self.chat_model.model_name}')
                return
            else:
                # TODO: Make sure important fields are consistent with manifest
                assert isinstance(
                    self.rag_settings["rag_llm"], LLM), "RAG LLM not initialized"
                print(f'RAG LLM: {self.rag_settings["rag_llm"].model_name}')
                print(
                    f'Using vectorstore: {self.rag_settings["collection_name"]}')
                print(
                    f'Embedding model: {self.rag_settings["embedding_model"].model_name}')
                print(f'Method: {self.rag_settings["method"]}')
                print(f'Chunk size: {self.rag_settings["chunk_size"]}')
                print(f'Chunk overlap: {self.rag_settings["chunk_overlap"]}')
                print(
                    f'Excerpts in context: {self.rag_settings["k_excerpts"]}')
                if self.rag_settings["multivector_enabled"]:
                    print(
                        f'Multivector enabled! Using method: {self.rag_settings["multivector_method"]}')
                return
        elif prompt == "rag":
            self.ingest_documents()
            return
        elif prompt == "reg":
            # Activate chat model (cast to LLM) if needed
            self.chat_model = self.activate_chat_model()
            self.rag_mode = False
            self.retriever = None
            self.rag_chain = None
            self.initialize_messages()
            return
        else:
            print('Invalid command: ', prompt)
            return

    def get_chat_response(self, prompt: str):
        assert self.chat_model is not None, "Chat model not initialized"
        self.messages.append(HumanMessage(content=prompt))
        print(f'Fetching response #{self.count + 1}!')
        try:
            response = self.chat_model.invoke(self.messages)
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

    def get_rag_response(self, prompt: str):
        self.messages.append(HumanMessage(content=prompt))
        print(f'RAG engine response #{self.count + 1}!')
        try:
            response = self.rag_chain.invoke(prompt)
        except KeyboardInterrupt:
            print('Keyboard interrupt, aborting generation.')
            return
        except Exception as e:
            print(f'Error!: {e}')
            return
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
                    prompt = self.prompt_from_clipboard()
                    if not prompt:
                        print('No text in clipboard! Try again.')
                        continue
                elif prompt == "read":
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
                self.command_handler(prompt)
                continue
            # Generate response
            if self.rag_mode:
                self.get_rag_response(prompt)
            else:
                self.get_chat_response(prompt)
            print()
        if save_response:
            save_response_to_markdown_file(self.messages[-1].content)
            print('Saved response to response.md')


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
    args = parser.parse_args()
    config = Config(rag_mode=args.rag_mode)
    prompt = args.prompt

    persistence_enabled = not args.not_persistent
    if prompt is None and persistence_enabled is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = read_sample()
            if EXPLAIN_EXCERPT:
                excerpt_as_prompt = SUMMARY_TEMPLATE.format(
                    excerpt=excerpt_as_prompt)
            prompt = excerpt_as_prompt

    if args.rag_mode:
        config.chat_config["rag_mode"] = True

    try:
        chatbot = Chatbot(config)
        chatbot.chat(prompt, persistence_enabled=persistence_enabled)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting.')
        raise SystemExit
