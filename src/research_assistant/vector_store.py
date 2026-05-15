from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import EvidenceChunk, Paper


class LocalVectorStore:
    def __init__(self) -> None:
        self._chunks: list[EvidenceChunk] = []
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=8000,
        )
        self._matrix = None

    def build(self, papers: list[Paper]) -> None:
        chunks = [
            EvidenceChunk(
                chunk_id=f"paper-{index}",
                source_id=paper.paper_id,
                source_title=paper.title,
                source_type=paper.source,
                text=f"{paper.title}. {paper.abstract}",
                url=paper.url,
            )
            for index, paper in enumerate(papers)
        ]
        self.build_chunks(chunks)

    def build_chunks(self, chunks: list[EvidenceChunk]) -> None:
        self._chunks = [chunk for chunk in chunks if chunk.text.strip()]
        documents = [chunk.text for chunk in self._chunks]
        self._matrix = self._vectorizer.fit_transform(documents) if documents else None

    def search(self, query: str, top_k: int = 5) -> list[tuple[EvidenceChunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [(self._chunks[index], float(scores[index])) for index in ranked if scores[index] > 0]


def paper_chunks(papers: list[Paper]) -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id=f"paper-{index}",
            source_id=paper.paper_id,
            source_title=paper.title,
            source_type=paper.source,
            text=f"{paper.title}. {paper.abstract}",
            url=paper.url,
        )
        for index, paper in enumerate(papers)
        if paper.abstract.strip()
    ]
