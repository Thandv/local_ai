#!/usr/bin/env python3
"""
build-app — CLI entry point for the multi-agent app builder.

Usage:
  build-app "Create a todo app with FastAPI and React"
  build-app --out ~/myproject "Build a REST API for a blog"
  build-app --interactive "SaaS dashboard with auth and billing"
  build-app --resume ~/builds/my-app-1430
  build-app --fix ~/builds/my-app-1430 --error "ModuleNotFoundError: No module named 'app'"
  build-app --no-review --no-debug "Quick Flask CRUD app"
  build-app --template saas "Subscription SaaS with Stripe"
"""

import sys
import argparse
from pathlib import Path

from local_ai import orchestrator
from local_ai.agents.shared import llm as _llm

# Canned preambles that guide the pipeline for common app shapes
TEMPLATES = {
    "saas": (
        "Build a production-ready SaaS application with: user registration and login "
        "(email/password + OAuth), subscription billing via Stripe, a dashboard for "
        "authenticated users, admin panel, and a landing page. "
    ),
    "api": (
        "Build a production-ready REST API with: JWT authentication, CRUD endpoints, "
        "request validation, rate limiting, OpenAPI docs, and a health-check endpoint. "
    ),
    "dashboard": (
        "Build a data dashboard with: charts and KPI cards, filterable tables, "
        "date-range picker, CSV export, and a sidebar navigation. "
    ),
    "cli": (
        "Build a polished command-line tool with: subcommands, rich terminal output "
        "(colours, progress bars), config file support, and comprehensive --help text. "
    ),
    "fullstack": (
        "Build a full-stack web application with a FastAPI backend, React + TypeScript "
        "frontend, PostgreSQL database, JWT auth, and Docker deployment. "
    ),
}


def main():
    parser = argparse.ArgumentParser(
        prog="build-app",
        description="Multi-agent AI app builder — describe an app, get working code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "instruction",
        nargs="*",
        help='What to build, e.g. "A todo app with FastAPI and React"',
    )
    parser.add_argument(
        "--out", "-o",
        metavar="DIR",
        help="Output directory (default: ~/builds/<project-name>/)",
    )
    parser.add_argument(
        "--template", "-t",
        choices=list(TEMPLATES.keys()),
        metavar="TYPE",
        help=f"Pre-populate instruction with a template ({', '.join(TEMPLATES)})",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="MODEL",
        help=(
            "LLM backend and model. Examples: "
            "qwen2.5-coder:32b (default Ollama), "
            "claude-opus-4-7 (best, needs ANTHROPIC_API_KEY), "
            "llama3.3:70b"
        ),
    )

    # ── Workflow modes ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Pause after Architect to confirm PLAN.md before coding starts",
    )
    parser.add_argument(
        "--resume",
        metavar="DIR",
        help="Resume a previous build from its last completed stage",
    )
    parser.add_argument(
        "--fix",
        metavar="DIR",
        help="Run Debugger on an existing build dir to fix a known error",
    )
    parser.add_argument(
        "--error",
        metavar="MSG",
        default="",
        help="Error message to pass when using --fix",
    )

    # ── Stage skips ────────────────────────────────────────────────────────────
    parser.add_argument("--no-review",   action="store_true", help="Skip Reviewer agent")
    parser.add_argument("--no-debug",    action="store_true", help="Skip Debugger agent")
    parser.add_argument("--no-devops",   action="store_true", help="Skip DevOps agent")
    parser.add_argument("--no-audit",    action="store_true", help="Skip Auditor agent")
    parser.add_argument("--no-package",  action="store_true", help="Skip Packager agent")
    parser.add_argument("--no-designer", action="store_true", help="Skip Designer agent")

    args = parser.parse_args()

    if args.model:
        _llm.set_model(args.model)

    # ── --fix mode ────────────────────────────────────────────────────────────
    if args.fix:
        orchestrator.fix(
            build_dir=Path(args.fix).expanduser(),
            error=args.error,
        )
        return

    # ── --resume mode ─────────────────────────────────────────────────────────
    if args.resume:
        instruction = " ".join(args.instruction) if args.instruction else ""
        orchestrator.run(
            instruction=instruction,
            resume_dir=Path(args.resume).expanduser(),
            skip_review=args.no_review,
            skip_debug=args.no_debug,
            skip_devops=args.no_devops,
            skip_audit=args.no_audit,
            skip_package=args.no_package,
            skip_designer=args.no_designer,
            interactive=args.interactive,
        )
        return

    # ── Normal build ──────────────────────────────────────────────────────────
    if not args.instruction and not args.template:
        parser.error("Provide an instruction or use --template.")

    instruction = " ".join(args.instruction) if args.instruction else ""

    if args.template:
        instruction = TEMPLATES[args.template] + instruction

    if not instruction.strip():
        parser.error("Instruction cannot be empty.")

    output_dir = Path(args.out).expanduser() if args.out else None

    orchestrator.run(
        instruction=instruction,
        output_dir=output_dir,
        skip_review=args.no_review,
        skip_debug=args.no_debug,
        skip_devops=args.no_devops,
        skip_audit=args.no_audit,
        skip_package=args.no_package,
        skip_designer=args.no_designer,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
