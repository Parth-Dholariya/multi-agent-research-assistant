from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_assistant.chat import ChatAgent
from research_assistant.config import Settings
from research_assistant.github_review import GitHubProfileReviewer, looks_like_github_review_request
from research_assistant.models import GitHubProfileReviewResult, LiteratureReviewResult
from research_assistant.pipeline import LiteratureReviewPipeline


st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="search",
    layout="wide",
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 24rem !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 24rem !important;
        }

        textarea[aria-label="Topic"] {
            min-height: 7rem !important;
            resize: vertical !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Multi-Agent Research Assistant")
st.caption("Review literature or GitHub profiles with coordinated AI agents.")

with st.sidebar:
    st.header("Research Setup")
    topic = st.text_area(
        "Topic",
        value="multi-agent systems for literature review",
        height=120,
        placeholder="Enter a research topic, or write: review my GitHub profile github.com/username",
    )
    max_papers = st.slider("Papers to review", min_value=3, max_value=15, value=8)
    uploaded_pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Optional: include your own papers or reports in the review and chat index.",
    )
    run_button = st.button("Run Agents", type="primary", use_container_width=True)

    st.divider()
    settings = Settings()
    if settings.has_azure_openai:
        st.success("Azure OpenAI enabled")
    elif settings.has_openai:
        st.success("OpenAI enabled")
    else:
        st.info("Running with offline extractive summaries")

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if run_button:
    if not topic.strip():
        st.warning("Enter a research topic or GitHub profile request first.")
    else:
        with st.status("Agents are working...", expanded=True) as status:
            try:
                if looks_like_github_review_request(topic):
                    st.write("GitHub Search Agent: querying the public GitHub API")
                    result = GitHubProfileReviewer().run(topic.strip())
                else:
                    st.write("Search Agent: querying academic APIs")
                    pipeline = LiteratureReviewPipeline(settings)
                    pdf_files = [(file.name, file.getvalue()) for file in uploaded_pdfs]
                    result = pipeline.run(topic.strip(), max_papers=max_papers, pdf_files=pdf_files)
                st.session_state.result = result
                st.session_state.chat_messages = []
                for trace in result.trace:
                    st.write(f"{trace.created_at} - {trace.agent}: {trace.message}")
                status.update(label="Agent run complete", state="complete")
            except Exception as exc:
                status.update(label="Run failed", state="error")
                st.error(str(exc))

result = st.session_state.result

if isinstance(result, GitHubProfileReviewResult):
    tab_report, tab_repos, tab_trace = st.tabs(["Review", "Repositories", "Agent Trace"])

    with tab_report:
        st.download_button(
            "Download GitHub Review",
            data=result.report_markdown,
            file_name=f"{result.username}_github_review.md",
            mime="text/markdown",
        )
        st.markdown(result.report_markdown)

    with tab_repos:
        repo_rows = [
            {
                "Name": repo.name,
                "Language": repo.language,
                "Stars": repo.stars,
                "Forks": repo.forks,
                "Description": repo.description,
                "Updated": repo.updated_at,
                "URL": repo.url,
            }
            for repo in result.repos
        ]
        st.dataframe(pd.DataFrame(repo_rows), use_container_width=True, hide_index=True)

    with tab_trace:
        for trace in result.trace:
            st.write(f"**{trace.created_at} - {trace.agent}:** {trace.message}")

elif isinstance(result, LiteratureReviewResult):
    tab_report, tab_chat, tab_papers, tab_docs, tab_critique, tab_trace = st.tabs(
        ["Report", "Chat", "Papers", "PDFs", "Critic", "Agent Trace"]
    )

    with tab_report:
        st.download_button(
            "Download Markdown Report",
            data=result.report_markdown,
            file_name="literature_review.md",
            mime="text/markdown",
        )
        st.markdown(result.report_markdown)

    with tab_chat:
        st.subheader("Chat With Results")
        st.caption("Answers are grounded in retrieved paper abstracts and uploaded PDF chunks.")

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask about the reviewed papers or uploaded PDFs")
        if question:
            st.session_state.chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            chat_agent = ChatAgent(LiteratureReviewPipeline(settings).llm)
            answer, sources = chat_agent.answer(question, result.evidence_chunks)
            source_lines = []
            if sources:
                source_lines.append("\n\n**Sources used:**")
                for index, source in enumerate(sources, start=1):
                    page = f", page {source.page}" if source.page else ""
                    source_lines.append(f"{index}. {source.source_title}{page}")
            content = answer + "\n".join(source_lines)
            st.session_state.chat_messages.append({"role": "assistant", "content": content})
            with st.chat_message("assistant"):
                st.markdown(content)

    with tab_papers:
        paper_rows = [
            {
                "Title": paper.title,
                "Year": paper.year,
                "Authors": ", ".join(paper.authors[:3]),
                "Source": paper.source,
                "Citations": paper.citation_count,
                "URL": paper.url,
            }
            for paper in result.papers
        ]
        st.dataframe(pd.DataFrame(paper_rows), use_container_width=True, hide_index=True)
        st.subheader("Summaries")
        for summary in result.summaries:
            with st.expander(summary.paper.title):
                st.write(summary.summary)
                st.markdown("**Evidence quotes**")
                for quote in summary.evidence_quotes:
                    st.markdown(f"- {quote}")

    with tab_docs:
        if result.documents:
            doc_rows = [
                {
                    "Filename": document.filename,
                    "Characters": len(document.text),
                    "Chunks": len(document.chunks),
                }
                for document in result.documents
            ]
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)
            for document in result.documents:
                with st.expander(document.filename):
                    st.write(document.text[:4000] + ("..." if len(document.text) > 4000 else ""))
        else:
            st.info("No PDFs were uploaded for this run.")


    with tab_critique:
        critique_rows = []
        title_by_id = {paper.paper_id: paper.title for paper in result.papers}
        for critique in result.critiques:
            critique_rows.append(
                {
                    "Paper": title_by_id.get(critique.paper_id, critique.paper_id),
                    "Evidence Score": critique.evidence_score,
                    "Risks": " | ".join(critique.risks),
                    "Missing Metadata": ", ".join(critique.missing_fields),
                }
            )
        st.dataframe(pd.DataFrame(critique_rows), use_container_width=True, hide_index=True)

    with tab_trace:
        for trace in result.trace:
            st.write(f"**{trace.created_at} - {trace.agent}:** {trace.message}")
else:
    st.info("Enter a research topic, or try: review my GitHub profile github.com/octocat")
