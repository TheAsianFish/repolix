"""
embedder.py

Converts text chunks into dense vector embeddings using the OpenAI
embeddings API, with batching and retry logic for reliable throughput.
"""

# TODO: Implement batch embedding of Chunk objects via the OpenAI
# embeddings endpoint, returning a list of (Chunk, vector) pairs.
