# task 1:
Implement workflow:
1. Add documents (.pdf or .txt files) to documents folder in repo
2. Set settings.json and adjust values for:
    - create_child_docs: bool = True - This will be used for MultiVectorRetriever
    - children_method: str = "summary" - This will be used to determine how to create child documents
3. Assert that vectorstore doesn't already exist
4. Create summaries (or Q&A pairs depending on children_method) for each page
    - LLM chain that takes doc_page and returns questions, can be batched for the full pdf
    - Generate list of summary strings
5. Create child_docs using the summary strings
    - These should be type `list[Document]` with appropriate metadata
    - Metadata should include an id_key like "doc_id" to hold uuid to match to parent docs
    - Must be the same length as parent docs
6. Create a vectorstore (faiss or chroma) using the child docs
7. Create a MultiVectorRetriever with:
    - vectorstore: the vectorstore created in step 4
    - docstore: the bytestore created with InMemoryStore with the parent documents
    - id_key: string for the uuid key in metadata of child docs (Usually "doc_id")
8. Use `retriever.docstore.mset(list(zip(doc_ids, parent_docs)))` to connect parent docs to the retriever
8. Create a pipeline with the MultiVectorRetriever and an LLM
    - LLM chain that takes retriever and query and returns response
    - This will vector search the summaries, but return references to the parent docs
    - Excerpts saved to output.md
9. Run chat.py with -rag just like with simple retriever
# Status: Complete

# Current task: Hit MVP on Flashcard demo

MVP:
- [X] Create a basic flashcard UI component
- [X] Given a dictionary or file flashcards.json: Display them as flashcards
- [ ] A flashapp.py that runs a flashcard app with features:
  - [X] UX allows for going through the flashcards
  - [X] An input to jump to a flashcard index
  - [ ] *A way to mark a flashcard as known (manage state?)

* = More advanced, but I'd want 3 types of flashcard attributes: Public, Hidden, Private

# TODO
- Use Unstructured for pdf partioning, cleaning, and chunking
- Implement FastAPI server
- Implement Web UI
- Implement a way to mark flashcards as known