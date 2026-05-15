from __future__ import annotations

from collections import Counter

from .llm import LlmClient
from .models import AgentTrace, Critique, PaperSummary
from .text_utils import keywords


class ReportAgent:
    name = "Report Agent"

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    def run(
        self,
        topic: str,
        summaries: list[PaperSummary],
        critiques: list[Critique],
        trace: list[AgentTrace],
    ) -> str:
        trace.append(AgentTrace(self.name, "Creating final literature review report."))
        if self.llm.enabled:
            generated = self._llm_report(topic, summaries, critiques)
            if generated:
                return generated
        return self._fallback_report(topic, summaries, critiques)

    def _llm_report(
        self,
        topic: str,
        summaries: list[PaperSummary],
        critiques: list[Critique],
    ) -> str | None:
        evidence = []
        critique_by_id = {critique.paper_id: critique for critique in critiques}
        for index, summary in enumerate(summaries, start=1):
            paper = summary.paper
            critique = critique_by_id.get(paper.paper_id)
            evidence.append(
                f"[{index}] {paper.title} ({paper.citation_label})\n"
                f"Summary: {summary.summary}\n"
                f"Evidence: {' '.join(summary.evidence_quotes)}\n"
                f"Critique: {', '.join(critique.risks if critique else [])}"
            )
        prompt = (
            f"Topic: {topic}\n\n"
            + "\n\n".join(evidence)
            + "\n\nWrite a structured literature review with sections: Overview, Main Themes, "
            "Evidence Table, Gaps and Limitations, Future Work, References. Cite papers by bracket number."
        )
        return self.llm.complete(
            "You are a rigorous literature review assistant. Only use the supplied evidence.",
            prompt,
            max_tokens=1200,
        )

    def _fallback_report(
        self,
        topic: str,
        summaries: list[PaperSummary],
        critiques: list[Critique],
    ) -> str:
        critique_by_id = {critique.paper_id: critique for critique in critiques}
        theme_terms = Counter()
        for summary in summaries:
            theme_terms.update(keywords(f"{summary.paper.title} {summary.paper.abstract}", limit=8))
        themes = [term for term, _ in theme_terms.most_common(6)]

        lines = [
            f"# Literature Review: {topic}",
            "",
            "## Overview",
            f"This review synthesizes {len(summaries)} papers related to **{topic}**. "
            "The report was generated from paper metadata and abstracts, with a critic pass for evidence quality.",
            "",
            "## Main Themes",
        ]
        if themes:
            lines.extend([f"- {theme.title()}" for theme in themes])
        else:
            lines.append("- No stable themes could be extracted from the available abstracts.")

        lines.extend(["", "## Paper Summaries"])
        for index, summary in enumerate(summaries, start=1):
            paper = summary.paper
            lines.extend(
                [
                    f"### [{index}] {paper.title}",
                    f"**Citation:** {paper.citation_label}",
                    f"**Source:** {paper.source}",
                    f"**URL:** {paper.url or 'Not available'}",
                    "",
                    summary.summary,
                    "",
                    "**Key evidence from abstract:**",
                ]
            )
            lines.extend([f"- {quote}" for quote in summary.evidence_quotes] or ["- No abstract evidence available."])
            critique = critique_by_id.get(paper.paper_id)
            if critique:
                lines.extend(
                    [
                        "",
                        f"**Evidence score:** {critique.evidence_score}",
                        "**Critic notes:**",
                    ]
                )
                lines.extend([f"- {risk}" for risk in critique.risks])
                if critique.missing_fields:
                    lines.append(f"- Missing metadata: {', '.join(critique.missing_fields)}")
            lines.append("")

        lines.extend(
            [
                "## Gaps and Limitations",
                "- This review is abstract-driven; full-text claims should be verified before publication.",
                "- Search results may reflect API ranking biases and source coverage limits.",
                "- Papers with missing abstracts or metadata are underrepresented in the synthesis.",
                "",
                "## Future Work",
                "- Add full-text PDF ingestion for deeper evidence extraction.",
                "- Add citation graph expansion to find seminal and follow-up papers.",
                "- Add human-in-the-loop review controls for approving summaries before report generation.",
                "",
                "## References",
            ]
        )
        for index, summary in enumerate(summaries, start=1):
            paper = summary.paper
            authors = ", ".join(paper.authors[:6]) if paper.authors else "Unknown authors"
            lines.append(f"[{index}] {authors}. ({paper.year or 'n.d.'}). {paper.title}. {paper.url}")
        return "\n".join(lines)

