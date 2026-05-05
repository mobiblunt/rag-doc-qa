"""
Shared embedding model — loaded once at startup, reused everywhere.

Model: all-MiniLM-L6-v2
- 384-dimensional embeddings
- ~90MB download on first run (cached after that)
- Fast on CPU, no GPU needed
- Genuinely good quality for RAG retrieval tasks
"""
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model (first run may take a moment)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model ready.")
    return _model


def embed(text: str) -> list[float]:
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts at once — much faster than one-by-one."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()