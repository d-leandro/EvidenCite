"""Evidence retrieval from uploaded PDF documents."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.evidence import get_retriever
from api.pdf_cache import get_pdf_parser

router = APIRouter(prefix="/api/v1")
MAX_PDF_BYTES = 20 * 1024 * 1024


@router.post("/evidence/pdf")
def retrieve_pdf_evidence(
    file: UploadFile = File(...),
    citing_context: str = Form(...),
    target_citation: str | None = Form(None),
    cited_title: str | None = Form(None),
    cited_abstract: str | None = Form(None),
):
    if len(citing_context.strip()) < 20:
        raise HTTPException(422, "Citation context is too short.")

    try:
        contents = file.file.read(MAX_PDF_BYTES + 1)
    finally:
        file.file.close()

    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF exceeds 20 MB limit.")

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(422, "A valid PDF file is required.")

    from src.modeling.predict import (
        InferenceRequest,
        resolve_request_evidence,
    )

    with TemporaryDirectory(prefix="evidencite-") as directory:
        pdf_path = Path(directory) / f"{sha256(contents).hexdigest()[:16]}.pdf"
        pdf_path.write_bytes(contents)

        request = InferenceRequest(
            citing_context=citing_context,
            target_citation=target_citation,
            cited_pdf_path=pdf_path,
            cited_title=cited_title,
            cited_abstract=cited_abstract,
        )

        result = resolve_request_evidence(
            request,
            retriever=get_retriever(),
            docling_parser=get_pdf_parser(),
            top_k=3,
        )

    chunks = result.cited_chunks or []

    if not chunks:
        raise HTTPException(
            422,
            "Unable to extract evidence from this PDF.",
        )

    return {
        "source": "uploaded_pdf",
        "cited_title": result.cited_title,
        "cited_abstract": result.cited_abstract,
        "cited_chunks": chunks,
        "cited_chunk_sections": result.cited_chunk_sections or [],
        "count": len(chunks),
    }
