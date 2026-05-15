# Multi-Agent Research Assistant for Literature Review

A Streamlit app that coordinates specialized research agents to search academic sources, summarize papers, critique evidence quality, and generate a literature review report.

## Learning Purpose

This project is built for learning how multi-agent AI workflows can turn a manual research process into a coordinated system:

- **Search Agent** finds papers from arXiv and Semantic Scholar.
- **Summarizer Agent** creates grounded summaries from title, abstract, and metadata.
- **Critic Agent** checks citation coverage, missing evidence, weak claims, and possible hallucination risk.
- **Report Agent** builds a structured literature review with themes, evidence, limitations, and references.

The app runs without an LLM key using extractive NLP fallbacks. If Azure OpenAI or OpenAI credentials are provided, summaries and reports become more fluent.

## Features

- Search arXiv and Semantic Scholar from one UI
- Review a GitHub profile by entering a username or profile URL
- Upload PDFs and include them in the literature review
- Chat with reviewed papers and uploaded PDFs using grounded source snippets
- Deduplicate and rank papers
- Local vector search over retrieved abstracts
- Multi-agent run trace for explainability
- Evidence-first summaries with direct paper references
- Critic checks for unsupported claims and missing metadata
- Downloadable Markdown literature review
- Optional Azure OpenAI/OpenAI integration

## Tech Stack

- Python
- Streamlit
- arXiv API
- Semantic Scholar API
- Scikit-learn vector store
- Optional Azure OpenAI/OpenAI LLM calls

## Quick Start

```powershell
cd "D:\resume project\multi-agent-research-assistant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Example inputs:

```text
multi-agent systems for literature review
review my GitHub profile github.com/octocat
review my GitHub profile @octocat
```

To use PDF chat:

1. Enter a research topic.
2. Upload one or more PDF papers in the sidebar.
3. Click **Run Agents**.
4. Open the **Chat** tab and ask questions about the reviewed evidence.

## Optional Environment Variables

Copy `.env.example` to `.env` and fill in values if you want LLM-enhanced generation.

```powershell
Copy-Item .env.example .env
```

Supported providers:

- Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`
- Semantic Scholar key, optional: `SEMANTIC_SCHOLAR_API_KEY`

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- .env.example
`-- src
    `-- research_assistant
        |-- agents.py
        |-- config.py
        |-- llm.py
        |-- models.py
        |-- pipeline.py
        |-- providers.py
        |-- report.py
        |-- text_utils.py
        `-- vector_store.py
```
