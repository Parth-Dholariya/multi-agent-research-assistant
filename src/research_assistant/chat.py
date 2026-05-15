from __future__ import annotations

from .llm import LlmClient
from .models import EvidenceChunk
from .text_utils import extractive_summary
from .vector_store import LocalVectorStore


class ChatAgent:
    name = "Chat Agent"

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    def answer(self, question: str, chunks: list[EvidenceChunk], top_k: int = 5) -> tuple[str, list[EvidenceChunk]]:
        store = LocalVectorStore()
        store.build_chunks(chunks)
        matches = store.search(question, top_k=top_k)
        selected = [chunk for chunk, _ in matches]

        if not selected:
            return (
                "I could not find enough relevant evidence in the indexed papers or uploaded PDFs.",
                [],
            )

        if self.llm.enabled:
            answer = self._llm_answer(question, selected)
            if answer:
                return answer, selected

        return self._fallback_answer(question, selected), selected

    def _llm_answer(self, question: str, chunks: list[EvidenceChunk]) -> str | None:
        context = "\n\n".join(
            f"[{index}] {source_label(chunk)}\n{chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )
        prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            "Answer using only the evidence above. Cite sources with bracket numbers."
        )
        return self.llm.complete(
            "You are a grounded research assistant. If evidence is weak, say so.",
            prompt,
            max_tokens=650,
        )

    def _fallback_answer(self, question: str, chunks: list[EvidenceChunk]) -> str:
        lines = [f"Based on the indexed evidence for: **{question}**", ""]
        for index, chunk in enumerate(chunks, start=1):
            summary = " ".join(extractive_summary(chunk.text, max_sentences=2))
            lines.append(f"{index}. **{source_label(chunk)}**: {summary}")
        lines.extend(
            [
                "",
                "This answer is extractive because no LLM key is configured.",
            ]
        )
        return "\n".join(lines)


def source_label(chunk: EvidenceChunk) -> str:
    label = chunk.source_title
    if chunk.page:
        label += f", page {chunk.page}"
    return label

