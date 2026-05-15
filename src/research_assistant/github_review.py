from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse

import requests

from .models import AgentTrace, GitHubProfileReviewResult, GitHubRepo


class GitHubProfileError(RuntimeError):
    pass


def looks_like_github_review_request(text: str) -> bool:
    lowered = text.lower()
    return "github" in lowered and any(word in lowered for word in ("review", "profile", "portfolio"))


def extract_github_username(text: str) -> str | None:
    url_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]+)", text)
    if url_match:
        return url_match.group(1).strip("/")

    at_match = re.search(r"@([A-Za-z0-9-]+)", text)
    if at_match:
        return at_match.group(1)

    lowered = text.lower()
    if "github.com/" in lowered:
        parsed = urlparse(text.strip())
        parts = [part for part in parsed.path.split("/") if part]
        return parts[0] if parts else None

    words = re.findall(r"[A-Za-z0-9-]+", text)
    noise = {
        "review",
        "my",
        "github",
        "profile",
        "portfolio",
        "please",
        "analyze",
        "check",
    }
    candidates = [word for word in words if word.lower() not in noise]
    return candidates[-1] if candidates else None


class GitHubProfileReviewer:
    def run(self, request_text: str) -> GitHubProfileReviewResult:
        trace: list[AgentTrace] = []
        username = extract_github_username(request_text)
        if not username:
            raise GitHubProfileError(
                "Add your GitHub username or profile URL, for example: review my GitHub profile github.com/octocat"
            )

        trace.append(AgentTrace("GitHub Search Agent", f"Fetching public profile for {username}."))
        profile = self._get_json(f"https://api.github.com/users/{username}")
        if "message" in profile and profile["message"] == "Not Found":
            raise GitHubProfileError(f"GitHub user '{username}' was not found.")

        trace.append(AgentTrace("Repo Agent", "Fetching public repositories and ranking recent portfolio signals."))
        repo_items = self._get_json(
            f"https://api.github.com/users/{username}/repos?sort=updated&per_page=100"
        )
        repos = [
            GitHubRepo(
                name=item.get("name", ""),
                description=item.get("description"),
                language=item.get("language"),
                stars=item.get("stargazers_count", 0),
                forks=item.get("forks_count", 0),
                url=item.get("html_url", ""),
                updated_at=item.get("updated_at"),
                topics=item.get("topics") or [],
            )
            for item in repo_items
            if not item.get("fork")
        ]
        repos = sorted(repos, key=lambda repo: (repo.stars, repo.forks, repo.updated_at or ""), reverse=True)
        trace.append(AgentTrace("Critic Agent", "Checking profile strengths, gaps, and recruiter-facing improvements."))
        report = self._build_report(profile, repos[:12])
        trace.append(AgentTrace("Report Agent", "Generated GitHub profile review report."))

        return GitHubProfileReviewResult(
            username=username,
            profile_url=profile.get("html_url", f"https://github.com/{username}"),
            name=profile.get("name"),
            bio=profile.get("bio"),
            public_repos=profile.get("public_repos", 0),
            followers=profile.get("followers", 0),
            following=profile.get("following", 0),
            repos=repos[:12],
            report_markdown=report,
            trace=trace,
        )

    def _get_json(self, url: str) -> dict | list:
        response = requests.get(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "multi-agent-research-assistant",
            },
            timeout=20,
        )
        if response.status_code == 403:
            raise GitHubProfileError("GitHub API rate limit reached. Try again later.")
        response.raise_for_status()
        return response.json()

    def _build_report(self, profile: dict, repos: list[GitHubRepo]) -> str:
        username = profile.get("login", "unknown")
        languages = Counter(repo.language for repo in repos if repo.language)
        topic_terms = Counter(topic for repo in repos for topic in repo.topics)
        described = sum(1 for repo in repos if repo.description)
        total_stars = sum(repo.stars for repo in repos)

        strengths = []
        if profile.get("bio"):
            strengths.append("Profile bio gives visitors quick context.")
        if repos:
            strengths.append(f"Shows {len(repos)} non-fork repositories in the reviewed set.")
        if languages:
            strengths.append("Clear technical stack signals: " + ", ".join(lang for lang, _ in languages.most_common(5)) + ".")
        if total_stars:
            strengths.append(f"Repositories have {total_stars} combined stars in the reviewed set.")
        if not strengths:
            strengths.append("The public profile exists and can be improved into a stronger portfolio surface.")

        gaps = []
        if not profile.get("bio"):
            gaps.append("Add a concise bio with role, main stack, and target domain.")
        if described < max(len(repos) // 2, 1):
            gaps.append("Several repositories need descriptions so recruiters can understand them quickly.")
        if not topic_terms:
            gaps.append("Add repository topics such as python, streamlit, ai-agents, rag, or machine-learning.")
        if not repos:
            gaps.append("Pin or publish project repositories that show practical engineering work.")

        lines = [
            f"# GitHub Profile Review: {username}",
            "",
            "## Snapshot",
            f"- Profile: {profile.get('html_url', f'https://github.com/{username}')}",
            f"- Name: {profile.get('name') or 'Not provided'}",
            f"- Bio: {profile.get('bio') or 'Not provided'}",
            f"- Public repositories: {profile.get('public_repos', 0)}",
            f"- Followers: {profile.get('followers', 0)}",
            "",
            "## Strengths",
        ]
        lines.extend([f"- {item}" for item in strengths])
        lines.extend(["", "## Improvement Areas"])
        lines.extend([f"- {item}" for item in gaps] or ["- No major gaps detected from public metadata."])

        lines.extend(["", "## Repository Highlights"])
        if repos:
            for repo in repos[:8]:
                lines.extend(
                    [
                        f"### {repo.name}",
                        f"- URL: {repo.url}",
                        f"- Language: {repo.language or 'Not specified'}",
                        f"- Stars/Forks: {repo.stars}/{repo.forks}",
                        f"- Description: {repo.description or 'Missing description'}",
                    ]
                )
        else:
            lines.append("- No non-fork public repositories found.")

        lines.extend(
            [
                "",
                "## Action Plan",
                "- Pin 4 to 6 strongest projects that match your target role.",
                "- Add a polished README to each pinned project with problem, tech stack, setup, screenshots, and results.",
                "- Use clear repo descriptions and topics so projects are searchable.",
                "- Put your best AI, RAG, agent, or full-stack projects first.",
                "- Add a profile README if you do not already have one.",
            ]
        )
        return "\n".join(lines)
