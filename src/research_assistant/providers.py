from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

from .config import Settings
from .models import Paper
from .text_utils import clean_text


ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


class SearchProviderError(RuntimeError):
    pass


def search_arxiv(query: str, limit: int = 8) -> list[Paper]:
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{quote_plus(query)}&start=0&max_results={limit}"
        "&sortBy=relevance&sortOrder=descending"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        paper_url = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
        abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS))
        published_at = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
        year = int(published_at[:4]) if published_at[:4].isdigit() else None
        authors = [
            clean_text(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
            for author in entry.findall("atom:author", ARXIV_NS)
        ]
        papers.append(
            Paper(
                paper_id=paper_url.rsplit("/", 1)[-1],
                title=title,
                abstract=abstract,
                authors=[author for author in authors if author],
                year=year,
                source="arXiv",
                url=paper_url,
                venue="arXiv",
                published_at=published_at or None,
            )
        )
    return papers


def search_semantic_scholar(query: str, settings: Settings, limit: int = 8) -> list[Paper]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,abstract,authors,year,url,venue,citationCount,externalIds"
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    response = requests.get(
        url,
        params={"query": query, "limit": limit, "fields": fields},
        headers=headers,
        timeout=20,
    )
    if response.status_code == 429:
        time.sleep(1.5)
        response = requests.get(
            url,
            params={"query": query, "limit": limit, "fields": fields},
            headers=headers,
            timeout=20,
        )
    response.raise_for_status()

    papers: list[Paper] = []
    for item in response.json().get("data", []):
        abstract = clean_text(item.get("abstract") or "")
        if not abstract:
            continue
        authors = [author.get("name", "") for author in item.get("authors", [])]
        papers.append(
            Paper(
                paper_id=item.get("paperId") or item.get("url") or item.get("title"),
                title=clean_text(item.get("title") or ""),
                abstract=abstract,
                authors=[author for author in authors if author],
                year=item.get("year"),
                source="Semantic Scholar",
                url=item.get("url") or "",
                venue=item.get("venue") or None,
                citation_count=item.get("citationCount"),
            )
        )
    return papers


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        key = clean_text(paper.title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique

