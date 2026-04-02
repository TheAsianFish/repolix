"""
store.py

Persists chunk embeddings in a local ChromaDB collection so they can
be queried by similarity at retrieval time without network round-trips.
"""

# TODO: Implement upsert and collection management against a local
# ChromaDB instance, keyed by file path and chunk fingerprint.
