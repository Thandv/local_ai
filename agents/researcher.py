"""
Researcher Agent
Analyses the user's instruction, identifies the tech stack, and gathers
real code patterns from the local reference repos.

When the prompt is in a domain the curated reference set doesn't cover
(e.g. video generation, audio processing), the Researcher can search
GitHub for relevant repos, peek at their READMEs, and shallow-clone the
best fit so its code patterns become available to grep for the rest of
the pipeline.
"""

import re
from pathlib import Path
from .shared.llm import run_agent_loop
from .shared.tools import (
    grep_repos, SEARCH_GITHUB, PEEK_README, CLONE_SHALLOW, HANDLERS,
)
from .shared.memory import Project

SYSTEM = """You are a senior software research agent.

Given a description of an app to build, your job is to:
1. Read the initial keyword-grep results from the curated reference set.
2. Decide whether they cover the prompt's domain well enough.
3. If not, use your tools to find additional reference material:
     • search_github(query, language, min_stars, n) — find candidate repos
     • peek_readme(repo_url)   — read a candidate's README without cloning
     • clone_shallow(repo_url) — shallow-clone the best 1-2 candidates so
                                 the downstream coder can grep their code
4. Identify the best technology stack for this project.
5. Note key patterns, libraries, and conventions to follow.
6. Surface any important architectural decisions upfront.

Heuristics:
- If the initial grep results name real frameworks relevant to the prompt
  (e.g. FastAPI for a REST API), trust them and don't search.
- If the initial grep is sparse or off-topic (e.g. React patterns for a
  video-generation prompt), search GitHub for 1-3 better matches, peek
  their READMEs, then clone the best one.
- Don't speculatively clone — only clone after you've read the README and
  confirmed relevance. Max 2 clones per build.
- Prefer popular, well-maintained repos (≥500 stars) for stability.

Be concrete and opinionated. Choose the most modern, production-ready defaults.
Keep your output structured — use headings, bullet points, and short code samples.
Do not write the actual app — only research and recommendations.
"""


def _extract_keywords(instruction: str) -> list[str]:
    stop = {"a", "an", "the", "and", "or", "with", "for", "that", "this",
            "to", "of", "in", "on", "is", "i", "want", "build", "create",
            "make", "app", "me", "my", "need", "can", "use", "using", "full"}
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", instruction.lower())
    return [w for w in words if w not in stop][:8]


def run(project: Project) -> Project:
    print("  [Researcher] Analysing instruction and gathering patterns …")

    keywords = _extract_keywords(project.instruction)
    repo_patterns = grep_repos(keywords)

    sparse_grep = (
        "No relevant patterns found" in repo_patterns
        or len(repo_patterns) < 400
    )
    if sparse_grep:
        sparse_note = ("\n\n⚠ The curated grep returned little. You should use "
                       "search_github / peek_readme / clone_shallow to find "
                       "domain-relevant repos before proceeding.")
    else:
        sparse_note = ""

    task = f"""Instruction: {project.instruction}

Reference code patterns found in the curated repos:
{repo_patterns}{sparse_note}

Write a research brief covering:
## Tech Stack
(languages, frameworks, libraries — be specific with versions if relevant)

## External services or models
(any third-party APIs, models, or services this app must call — list them)

## Project Structure
(recommended folder/file layout)

## Key Patterns
(authentication approach, state management, database, API style, etc.)

## Dependencies
(exact package names to install)

## Gotchas & Best Practices
(anything that commonly trips people up with this stack)
"""

    handlers = {
        "search_github": HANDLERS["search_github"],
        "peek_readme":   HANDLERS["peek_readme"],
        "clone_shallow": HANDLERS["clone_shallow"],
    }
    tools = [SEARCH_GITHUB, PEEK_README, CLONE_SHALLOW]

    def on_call(name, args):
        if name == "search_github":
            print(f"    [search] {args.get('query','')!r}")
        elif name == "peek_readme":
            print(f"    [peek]   {args.get('repo_url','')}")
        elif name == "clone_shallow":
            print(f"    [clone]  {args.get('repo_url','')}")
        else:
            print(f"    [{name}]")

    project.research = run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    print("  [Researcher] Done.\n")
    return project
