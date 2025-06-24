- Generate responses via garak test runs

- Look into approach #3 in addition to previously stated approaches:
1. Baseline (no guidelines)
2. Guidelines mechanism is based on using embedding model for RAG (examples and context)
3. Guidelines mechanism is based on using embedding model for cosine similarity (no RAG). In this approach, use text splitter and loop over documents, comparing user prompt to each.