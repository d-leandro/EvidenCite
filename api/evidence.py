"""Shared model resources for evidence retrieval."""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_retriever():
    """Load SciBERT once per API process."""
    from src.data.retrieval import SciBERTRetriever

    return SciBERTRetriever()
