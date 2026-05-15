from __future__ import annotations

from .agents import CriticAgent, SearchAgent, SummarizerAgent
from .config import Settings
from .llm import LlmClient
from .models import AgentTrace, LiteratureReviewResult, Paper, UploadedDocument
from .pdf_utils import extract_pdf_document
from .report import ReportAgent
from .vector_store import LocalVectorStore, paper_chunks


class LiteratureReviewPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = LlmClient(self.settings)
        self.search_agent = SearchAgent(self.settings)
        self.summarizer_agent = SummarizerAgent(self.llm)
        self.critic_agent = CriticAgent()
        self.report_agent = ReportAgent(self.llm)
        self.vector_store = LocalVectorStore()

    def run(
        self,
        topic: str,
        max_papers: int = 8,
        pdf_files: list[tuple[str, bytes]] | None = None,
    ) -> LiteratureReviewResult:
        trace: list[AgentTrace] = []
        papers = self.search_agent.run(topic, max_papers=max_papers, trace=trace)
        documents = self._extract_documents(pdf_files or [], trace)
        pdf_papers = self._documents_to_papers(documents)
        all_papers = papers + pdf_papers
        evidence_chunks = paper_chunks(papers) + [chunk for document in documents for chunk in document.chunks]
        self.vector_store.build_chunks(evidence_chunks)
        trace.append(
            AgentTrace(
                "Vector Store",
                f"Indexed {len(evidence_chunks)} evidence chunks from {len(papers)} papers and {len(documents)} PDFs.",
            )
        )
        summaries = self.summarizer_agent.run(all_papers, trace=trace)
        critiques = self.critic_agent.run(summaries, trace=trace)
        report_markdown = self.report_agent.run(topic, summaries, critiques, trace=trace)
        return LiteratureReviewResult(
            topic=topic,
            papers=all_papers,
            summaries=summaries,
            critiques=critiques,
            documents=documents,
            evidence_chunks=evidence_chunks,
            report_markdown=report_markdown,
            trace=trace,
        )

    def _extract_documents(
        self,
        pdf_files: list[tuple[str, bytes]],
        trace: list[AgentTrace],
    ) -> list[UploadedDocument]:
        documents: list[UploadedDocument] = []
        if not pdf_files:
            return documents

        trace.append(AgentTrace("PDF Agent", f"Extracting text from {len(pdf_files)} uploaded PDF(s)."))
        for filename, content in pdf_files:
            try:
                document = extract_pdf_document(filename, content)
                if document.text:
                    documents.append(document)
                    trace.append(
                        AgentTrace(
                            "PDF Agent",
                            f"{filename}: extracted {len(document.chunks)} searchable chunks.",
                        )
                    )
                else:
                    trace.append(AgentTrace("PDF Agent", f"{filename}: no readable text found."))
            except Exception as exc:
                trace.append(AgentTrace("PDF Agent", f"{filename}: extraction failed: {exc}"))
        return documents

    def _documents_to_papers(self, documents: list[UploadedDocument]) -> list[Paper]:
        return [
            Paper(
                paper_id=f"uploaded-{index}-{document.filename}",
                title=document.filename,
                abstract=document.text[:5000],
                authors=[],
                year=None,
                source="Uploaded PDF",
                url="",
                venue="Uploaded document",
            )
            for index, document in enumerate(documents, start=1)
        ]
