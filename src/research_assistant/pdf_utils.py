from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from .models import EvidenceChunk, UploadedDocument
from .text_utils import clean_text


def extract_pdf_document(filename: str, content: bytes, max_chars: int = 80_000) -> UploadedDocument:
    reader = PdfReader(BytesIO(content))
    all_text: list[str] = []
    chunks: list[EvidenceChunk] = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if not page_text:
            continue
        all_text.append(page_text)
        for chunk_index, chunk_text in enumerate(chunk_text_by_words(page_text), start=1):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"pdf-{filename}-{page_index}-{chunk_index}",
                    source_id=filename,
                    source_title=filename,
                    source_type="Uploaded PDF",
                    text=chunk_text,
                    page=page_index,
                )
            )

        if sum(len(text) for text in all_text) >= max_chars:
            break

    return UploadedDocument(
        filename=filename,
        text=clean_text(" ".join(all_text))[:max_chars],
        chunks=chunks,
    )


def chunk_text_by_words(text: str, chunk_words: int = 180, overlap_words: int = 35) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if len(chunk.split()) >= 25:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks

