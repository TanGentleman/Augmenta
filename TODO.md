# TODO

Create workflow:
Input: Document as text or PDF
-> Ingest document with appropriate chunking (256, overlap 40, with parent document as metadata (each page?))
-> Add Q&A or simply questions as metadata for each page
Input: Query/Prompt
Output: Relevant chunks saved to output.md + Response

Specific steps:
1. Add document to appropriate folder in repo
2. Set config.py with values for: 
    - document name
    - chunk size
    - overlap
    - questions per page
3. Assert that document is present and vectorstore doesn't already exist
4. Create vectorstore using config values
5. Create questions for each page
    - LLM chain that takes doc_page and returns questions, can be batched for the full pdf
    - Generate a question_list for each page
    - Output should be a list of question_lists with the same length as number of pages
6. Create a new list of question_docs with items using `Document(page_content=question_list, metadata={id_key: doc_ids[i]})`
    - Reminder that the id_key should match the the parent page
7. Create a MultiVectorRetriever with:
    - vectorstore: the vectorstore created in step 4
    - document_store: the bytestore created with InMemoryStore with the parent documents
    - id_key
8. Create a pipeline with the MultiVectorRetriever and an LLM
    - LLM chain that takes retriever and query and returns response
    - I also want it to save the retrieved documents to an output.md
9. Run the pipeline with the query for the given document
