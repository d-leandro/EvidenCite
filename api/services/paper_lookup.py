"""Resolve cited paper metadata using existing clients."""

import re

from fastapi import HTTPException


ARXIV_ID_PATTERN = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)



def search_arxiv_by_title(title: str) -> dict | None:
    """Search arXiv by title, accepting only exact matches."""
    import xml.etree.ElementTree as ET

    import httpx

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    clean_title = " ".join(title.split()).strip('"')
    expected = clean_title.casefold()

    try:
        response = httpx.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f'ti:"{clean_title}"',
                "max_results": 5,
            },
            timeout=15,
            follow_redirects=True,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (httpx.HTTPError, ET.ParseError):
        return None

    for entry in root.findall("atom:entry", namespace):
        found_title = " ".join(
            entry.findtext("atom:title", "", namespace).split()
        )

        if found_title.casefold() != expected:
            continue

        identifier = entry.findtext("atom:id", "", namespace)
        arxiv_id = identifier.rsplit("/", 1)[-1].split("v")[0]

        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        authors = [
            author.findtext("atom:name", "", namespace)
            for author in entry.findall("atom:author", namespace)
        ]

        return {
            "source": "arxiv",
            "title": found_title,
            "abstract": " ".join(
                entry.findtext("atom:summary", "", namespace).split()
            ),
            "authors": authors,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
        }

    return None


def lookup_paper(query: str) -> dict:
    query = query.strip()

    if query.lower().startswith("arxiv:"):
        query = query[6:].strip()

    if ARXIV_ID_PATTERN.fullmatch(query):
        from src.data.ingestion.arxiv_resolver import ArxivClient

        paper = ArxivClient().fetch_metadata(query)

        if paper is None:
            raise HTTPException(
                status_code=404,
                detail="Paper not found or arXiv unavailable.",
            )

        return {
            "source": "arxiv",
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "arxiv_id": paper.arxiv_id,
            "pdf_url": paper.pdf_url,
        }

    arxiv_match = search_arxiv_by_title(query)
    if arxiv_match is not None:
        return arxiv_match

    from src.data.ingestion.s2_client import (
        SemanticScholarClient,
        S2ClientConfig,
    )

    config = S2ClientConfig(
        timeout_seconds=10,
        max_retries=1,
    )

    paper = SemanticScholarClient(
        config=config
    ).search_paper_by_title(query)

    if paper is None:
        raise HTTPException(
            status_code=404,
            detail="Paper not found or Semantic Scholar unavailable.",
        )

    return {
        "source": "semantic_scholar",
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "arxiv_id": paper.arxiv_id,
        "doi": paper.doi,
        "year": paper.year,
    }
