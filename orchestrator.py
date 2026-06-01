"""
Orchestrator — runs the full agent pipeline:
  Designer → Researcher → Architect → Migrator → Coder → Debugger
  → Tester → Auditor → DevOps → Reviewer → Packager

Supports:
  interactive  Pause after Architect for plan confirmation before coding
  resume_dir   Continue a previous build from its last completed stage
  fix()        Run Debugger on an existing build directory
"""

import re
import time
import json
from pathlib import Path
from datetime import datetime

from local_ai.agents.shared.memory import Project
from local_ai.agents import (
    researcher, architect, coder, tester, reviewer,
    designer, debugger, devops, migrator, auditor, packager,
)

BUILDS_DIR = Path.home() / "builds"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    slug = slug.strip("-")[:40]
    ts   = datetime.now().strftime("%H%M")
    return f"{slug}-{ts}"


def _header(title: str):
    width = 58
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def _confirm_plan(project: Project) -> bool:
    """Print PLAN.md and ask the user to confirm before coding starts."""
    plan_path = project.output_dir / "PLAN.md"
    if plan_path.exists():
        print("\n" + "="*60)
        print(plan_path.read_text(encoding="utf-8")[:4000])
        print("="*60)
    print("\n  [Interactive] Review the plan above.")
    while True:
        choice = input("  Continue with this plan? [y/n/edit]: ").strip().lower()
        if choice == "y":
            return True
        if choice == "n":
            print("  Build cancelled.")
            return False
        if choice == "edit":
            print(f"  Edit {plan_path} then press Enter to continue …")
            input()
            return True


def _load_checkpoint(output_dir: Path) -> dict:
    log_path = output_dir / "BUILD_LOG.json"
    if log_path.exists():
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_checkpoint(project: Project, stage: str):
    log_path = project.output_dir / "BUILD_LOG.json"
    try:
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    except Exception:
        log = {}
    completed = log.get("completed_stages", [])
    if stage not in completed:
        completed.append(stage)
    log.update({
        "instruction":      project.instruction,
        "started_at":       project.started_at,
        "files":            project.files_written,
        "success":          project.success,
        "packaged":         project.packaged,
        "completed_stages": completed,
    })
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def run(
    instruction: str,
    output_dir: Path = None,
    skip_review: bool = False,
    interactive: bool = False,
    skip_designer: bool = False,
    skip_devops: bool = False,
    skip_debug: bool = False,
    skip_audit: bool = False,
    skip_package: bool = False,
    resume_dir: Path = None,
) -> Project:
    """
    Run the full pipeline for a given instruction.

    Args:
        instruction:    Plain-English description of the app to build
        output_dir:     Where to write files (default: ~/builds/<slug>/)
        skip_review:    Skip the Reviewer agent
        interactive:    Pause after Architect for plan confirmation
        skip_designer:  Skip the Designer agent
        skip_devops:    Skip the DevOps agent
        skip_debug:     Skip the Debugger agent
        skip_audit:     Skip the Auditor agent
        skip_package:   Skip the Packager agent
        resume_dir:     Resume a previous build from this directory
    Returns:
        Completed Project object
    """
    if resume_dir:
        resume_dir = Path(resume_dir).expanduser()
        checkpoint = _load_checkpoint(resume_dir)
        completed_stages = set(checkpoint.get("completed_stages", []))
        instruction = instruction or checkpoint.get("instruction", instruction)
        output_dir = resume_dir
        print(f"\n  Resuming build in: {output_dir}")
        print(f"  Completed stages : {', '.join(completed_stages) or 'none'}")
    else:
        completed_stages = set()
        if not output_dir:
            output_dir = BUILDS_DIR / _slugify(instruction)
        output_dir.mkdir(parents=True, exist_ok=True)

    project = Project(instruction=instruction, output_dir=output_dir)

    if resume_dir:
        for f in checkpoint.get("files", []):
            if f not in project.files_written:
                project.files_written.append(f)

    print(f"\n{'='*60}")
    print(f"  Building: {instruction}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}")

    all_stages = [
        ("designer",   designer.run,   skip_designer),
        ("researcher", researcher.run, False),
        ("architect",  architect.run,  False),
        ("migrator",   migrator.run,   False),
        ("coder",      coder.run,      False),
        ("debugger",   debugger.run,   skip_debug),
        ("tester",     tester.run,     False),
        ("auditor",    auditor.run,    skip_audit),
        ("devops",     devops.run,     skip_devops),
        ("reviewer",   reviewer.run,   skip_review),
        ("packager",   packager.run,   skip_package),
    ]

    active_stages = [
        (name, fn) for (name, fn, skip) in all_stages
        if not skip and name not in completed_stages
    ]

    total = len(active_stages) + len(completed_stages)
    step  = len(completed_stages)

    for name, agent_fn in active_stages:
        step += 1
        label = name.capitalize()
        _header(f"Stage {step}/{total}: {label}")

        if interactive and name == "coder":
            if not _confirm_plan(project):
                project.save_log()
                return project

        t0 = time.time()
        try:
            project = agent_fn(project)
            _save_checkpoint(project, name)
        except Exception as e:
            print(f"\n  [ERROR in {label}]: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing to next stage …\n")
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

    project.save_log()
    print(project.summary())
    return project


def fix(build_dir: Path, error: str) -> Project:
    """Run only the Debugger on an existing build directory with a known error."""
    build_dir = Path(build_dir).expanduser()
    checkpoint = _load_checkpoint(build_dir)
    instruction = checkpoint.get("instruction", "existing project")

    project = Project(instruction=instruction, output_dir=build_dir)
    for f in checkpoint.get("files", []):
        project.files_written.append(f)

    project.debug_log = f"Known error reported by user:\n{error}"

    print(f"\n  Fixing: {build_dir}")
    _header("Debugger (fix mode)")
    t0 = time.time()
    project = debugger.run(project)
    print(f"  Completed in {time.time() - t0:.1f}s")

    project.save_log()
    return project
