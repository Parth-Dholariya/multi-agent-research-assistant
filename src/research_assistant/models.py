from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    source: str
    url: str
    venue: str | None = None
    citation_count: int | None = None
    published_at: str | None = None

    @property
    def citation_label(self) -> str:
        lead_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        year = self.year or "n.d."
        return f"{lead_author}, {year}"


@dataclass
class PaperSummary:
    paper: Paper
    summary: str
    key_points: list[str]
    evidence_quotes: list[str]


@dataclass
class Critique:
    paper_id: str
    risks: list[str]
    missing_fields: list[str]
    evidence_score: float


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_id: str
    source_title: str
    source_type: str
    text: str
    url: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class UploadedDocument:
    filename: str
    text: str
    chunks: list[EvidenceChunk]


@dataclass
class AgentTrace:
    agent: str
    message: str
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class LiteratureReviewResult:
    topic: str
    papers: list[Paper]
    summaries: list[PaperSummary]
    critiques: list[Critique]
    documents: list[UploadedDocument]
    evidence_chunks: list[EvidenceChunk]
    report_markdown: str
    trace: list[AgentTrace]


@dataclass(frozen=True)
class GitHubRepo:
    name: str
    description: str | None
    language: str | None
    stars: int
    forks: int
    url: str
    updated_at: str | None
    topics: list[str]


@dataclass
class GitHubProfileReviewResult:
    username: str
    profile_url: str
    name: str | None
    bio: str | None
    public_repos: int
    followers: int
    following: int
    repos: list[GitHubRepo]
    report_markdown: str
    trace: list[AgentTrace]
