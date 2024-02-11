The chunking process is crucial when working with large documents and long context embedding models, as these models have limitations on the input sequence length. The chunk_size determines how much of the text is processed in each step, while the chunk_overlap determines how much overlap there is between consecutive chunks to ensure context continuity.

For a 2K context length model, a chunk_size of 1800 with a chunk_overlap of 200 seems reasonable. Here's the reasoning:

1. **Chunk_size**: You want to keep the chunk size slightly smaller than the model's maximum context length (2K tokens) to avoid going over the limit. A chunk_size of 1800 allows the model to see almost the full context while processing each chunk.
2. **Chunk_overlap**: The overlap is important to ensure that the model can access relevant context from adjacent chunks when answering a query. A 200-token overlap should be enough to provide continuity between chunks, considering that the model can process up to 2K tokens in total.

For an 8K context length model, you would want to adjust the chunk_size accordingly:

1. **Chunk_size**: Since the model can handle a larger context, you could increase the chunk_size to, say, 7500 tokens, leaving a buffer of a few hundred tokens below the maximum limit (8K).
2. **Chunk_overlap**: The overlap can be reduced, as the larger context allows for more information to be captured between chunks. A 200-token overlap might still be sufficient, but you could experiment with lower values like 100 or 150 tokens to balance efficiency and context preservation.

Keep in mind that these values are not set in stone and might need to be adjusted based on your specific use case, the nature of your documents, and the performance of the model in retrieving relevant information. It's always a good idea to experiment with different chunk sizes and overlaps to find the optimal balance between efficiency and context preservation.