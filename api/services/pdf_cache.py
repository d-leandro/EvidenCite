"""Bounded in-memory cache of parsed PDF documents."""

from collections import OrderedDict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import Lock


class CachedPDFParser:
    def __init__(self):
        self._papers = OrderedDict()
        self._lock = Lock()

    def parse_pdf_to_structured_paper(
        self,
        pdf_source,
        paper_id=None,
        title=None,
        abstract=None,
    ):
        path = Path(pdf_source)
        key = (
            sha256(path.read_bytes()).hexdigest(),
            paper_id,
            title,
            abstract,
        )

        with self._lock:
            if key in self._papers:
                self._papers.move_to_end(key)
                return self._papers[key]

            from src.data.ingestion.arxiv_resolver import DoclingPDFParser

            paper = DoclingPDFParser().parse_pdf_to_structured_paper(
                path,
                paper_id=paper_id,
                title=title,
                abstract=abstract,
            )

            if paper is not None:
                self._papers[key] = paper

                if len(self._papers) > 2:
                    self._papers.popitem(last=False)

            return paper


@lru_cache(maxsize=1)
def get_pdf_parser():
    return CachedPDFParser()
