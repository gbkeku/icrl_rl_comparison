# ─────────────────────────────────────────────
# BM25 Retriever + MockRetriever
# ─────────────────────────────────────────────
from config import TOP_K_DOCS


class BM25Retriever:
    def __init__(self, corpus_path: str):
        from flashrag.retriever import BM25Retriever as FlashBM25
        self.retriever = FlashBM25(corpus_path)

    def retrieve(self, query: str) -> str:
        docs   = self.retriever.search(query, topk=TOP_K_DOCS)
        result = ""
        for i, doc in enumerate(docs):
            result += f"Doc {i+1} (Title: \"{doc['title']}\")\n"
            result += f"{doc['text']}\n\n"
        return result.strip()


class MockRetriever:
    """Lightweight mock for testing without corpus."""
    def retrieve(self, query: str) -> str:
        return (
            f"Doc 1 (Title: \"Mock Result\")\n"
            f"This is a mock document for query: {query}\n"
            f"It contains placeholder information for testing."
        )
