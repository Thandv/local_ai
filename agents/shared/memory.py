"""
Project memory — the shared state object passed through the agent pipeline.
Each agent reads from and writes to this object.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json


# Source-file extensions that count toward the "real output" threshold for
# the success criterion. Markdown docs and config files do NOT count — a
# build that produced only PLAN.md and a SECURITY.md is not a build.
_SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
    ".swift", ".rb", ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".scala",
    ".sh", ".bash", ".sql", ".html", ".css", ".vue", ".svelte",
}

# Minimum bars for declaring a build successful. Tuned so a real-but-tiny
# CLI passes (≥2 source files, ≥30 LOC) while empty stubs do not.
MIN_SOURCE_FILES = 2
MIN_SOURCE_LOC   = 30


@dataclass
class Project:
    instruction: str              # Original user instruction
    output_dir: Path              # Where all files are written

    # Populated by each agent in sequence
    design:        str = ""       # Designer agent output (UI/UX spec)
    research:      str = ""       # Researcher agent output
    plan:          str = ""       # Architect agent plan (also written to PLAN.md)
    files_written: list = field(default_factory=list)  # Paths written by all agents
    debug_log:     str = ""       # Debugger agent output
    test_results:  str = ""       # Tester output
    audit:         str = ""       # Auditor agent output
    review:        str = ""       # Reviewer output
    packaged:      bool = False   # Whether Packager ran successfully
    success:       bool = False   # Did the test/build gates pass?

    # Why the final SUCCESS / INCOMPLETE verdict was reached, populated by
    # compute_final_success() at the end of the pipeline.
    success_reason: str = ""

    # Metadata
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def _dedupe_files(self):
        seen, deduped = set(), []
        for f in self.files_written:
            if f not in seen:
                seen.add(f); deduped.append(f)
        self.files_written = deduped

    def compute_final_success(self) -> tuple[bool, str]:
        """Decide whether the build actually produced something usable.

        Requirements (all must hold):
          - PLAN.md exists in the output dir
          - At least MIN_SOURCE_FILES source files exist (markdown/docs don't count)
          - At least MIN_SOURCE_LOC lines of source across those files
          - self.success is True (i.e. the test gate didn't outright fail)

        Returns (passed, reason). Reason is empty on pass; explains the
        failing condition on fail.
        """
        out = Path(self.output_dir)

        if not (out / "PLAN.md").exists():
            return False, "PLAN.md was not produced — Architect stage failed or wrote no plan"

        source_files = []
        for p in out.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _SOURCE_EXTENSIONS:
                continue
            rel = p.relative_to(out)
            # Skip generated / tooling dirs
            if any(part in {".git", "venv", ".venv", "node_modules",
                            "__pycache__", ".pytest_cache", "dist", "build"}
                   for part in rel.parts):
                continue
            source_files.append(p)

        if len(source_files) < MIN_SOURCE_FILES:
            return False, (f"only {len(source_files)} source file(s) — "
                           f"need at least {MIN_SOURCE_FILES}")

        total_loc = 0
        for p in source_files:
            try:
                total_loc += sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
        if total_loc < MIN_SOURCE_LOC:
            return False, (f"only {total_loc} lines across source files — "
                           f"need at least {MIN_SOURCE_LOC}")

        if not self.success:
            return False, "test gate did not pass"

        return True, ""

    def save_log(self):
        """Persist pipeline log to output_dir/BUILD_LOG.json.
        Also computes the final SUCCESS verdict using compute_final_success()."""
        self._dedupe_files()
        passed, reason = self.compute_final_success()
        self.success = passed
        self.success_reason = reason

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log = {
            "instruction":    self.instruction,
            "started_at":     self.started_at,
            "files":          self.files_written,
            "success":        self.success,
            "success_reason": reason or "all gates passed",
            "packaged":       self.packaged,
        }
        (self.output_dir / "BUILD_LOG.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )

    def summary(self) -> str:
        status   = "✓ SUCCESS" if self.success else "✗ INCOMPLETE"
        reason   = f"  Reason  : {self.success_reason}\n" if (not self.success and self.success_reason) else ""
        packaged = "  Packaged: yes (README + git init)\n" if self.packaged else ""
        files    = "\n".join(f"  • {f}" for f in self.files_written) or "  (none)"
        return (
            f"\n{'='*60}\n"
            f"  {status}\n"
            f"  Project : {self.output_dir}\n"
            f"{reason}"
            f"{packaged}"
            f"  Files   :\n{files}\n"
            f"{'='*60}\n"
        )
