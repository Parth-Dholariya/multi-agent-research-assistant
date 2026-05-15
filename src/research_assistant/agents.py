from __future__ import annotations

from .config import Settings
from .llm import LlmClient
from .models import AgentTrace, Critique, Paper, PaperSummary
from .providers import deduplicate_papers, search_arxiv, search_semantic_scholar
from .text_utils import extractive_summary, keywords, split_sentences


class SearchAgent:
    name = "Search Agent"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, topic: str, max_papers: int, trace: list[AgentTrace]) -> list[Paper]:
        trace.append(AgentTrace(self.name, "Searching arXiv and Semantic Scholar."))
        papers: list[Paper] = []
        errors: list[str] = []

        for provider_name, provider in (
            ("arXiv", lambda: search_arxiv(topic, limit=max_papers)),
            ("Semantic Scholar", lambda: search_semantic_scholar(topic, self.settings, limit=max_papers)),
        ):
            try:
                provider_papers = provider()
                papers.extend(provider_papers)
                trace.append(AgentTrace(self.name, f"{provider_name} returned {len(provider_papers)} usable papers."))
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                trace.append(AgentTrace(self.name, f"{provider_name} search failed: {exc}"))

        unique = deduplicate_papers(papers)
        ranked = sorted(
            unique,
            key=lambda paper: ((paper.citation_count or 0), (paper.year or 0)),
            reverse=True,
        )[:max_papers]
        if errors and not ranked:
            raise RuntimeError("All search providers failed. " + " | ".join(errors))
        trace.append(AgentTrace(self.name, f"Deduplicated and selected {len(ranked)} papers."))
        return ranked


class SummarizerAgent:
    name = "Summarizer Agent"

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    def run(self, papers: list[Paper], trace: list[AgentTrace]) -> list[PaperSummary]:
        trace.append(AgentTrace(self.name, f"Summarizing {len(papers)} papers."))
        return [self._summarize(paper) for paper in papers]

    def _summarize(self, paper: Paper) -> PaperSummary:
        evidence = extractive_summary(paper.abstract, max_sentences=2)
        key_points = extractive_summary(paper.abstract, max_sentences=3)
        fallback = " ".join(evidence) if evidence else "No abstract was available for this paper."

        if self.llm.enabled and paper.abstract:
            prompt = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(paper.authors[:5])}\n"
                f"Year: {paper.year}\n"
                f"Abstract: {paper.abstract}\n\n"
                "Write a concise evidence-grounded summary in 3 bullets. Do not add facts missing from the abstract."
            )
            generated = self.llm.complete(
                "You are a careful academic summarizer. Stay grounded in the provided abstract.",
                prompt,
                max_tokens=350,
            )
            if generated:
                fallback = generated

        return PaperSummary(
            paper=paper,
            summary=fallback,
            key_points=key_points,
            evidence_quotes=evidence,
        )


class CriticAgent:
    name = "Critic Agent"

    def run(self, summaries: list[PaperSummary], trace: list[AgentTrace]) -> list[Critique]:
        trace.append(AgentTrace(self.name, "Checking evidence coverage and missing metadata."))
        return [self._critique(summary) for summary in summaries]

    def _critique(self, summary: PaperSummary) -> Critique:
        paper = summary.paper
        missing = []
        if not paper.authors:
            missing.append("authors")
        if not paper.year:
            missing.append("year")
        if not paper.abstract:
            missing.append("abstract")
        if not paper.url:
            missing.append("url")

        abstract_terms = set(keywords(paper.abstract, limit=20))
        summary_terms = set(keywords(summary.summary, limit=20))
        overlap = len(abstract_terms & summary_terms)
        evidence_score = overlap / max(len(summary_terms), 1)

        risks = []
        if evidence_score < 0.35:
            risks.append("Summary may contain claims weakly grounded in the abstract.")
        if len(split_sentences(paper.abstract)) < 3:
            risks.append("Abstract is short, so conclusions should be treated cautiously.")
        if paper.citation_count == 0:
            risks.append("Low citation count may indicate limited external validation.")
        if not risks:
            risks.append("No major evidence issues detected from available metadata.")

        return Critique(
            paper_id=paper.paper_id,
            risks=risks,
            missing_fields=missing,
            evidence_score=round(evidence_score, 2),
        )

