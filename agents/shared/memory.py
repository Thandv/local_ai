"""
Project memory — the shared state object passed through the agent pipeline.
Each agent reads from and writes to this object.
"""

from dataclasses import dataclass, field
from enum import Enum
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

BUILD_LOG = "BUILD_LOG.json"


class Verdict(str, Enum):
    """The build's final verdict; Project.compute_verdict() is its one owner."""
    SUCCESS    = "SUCCESS"     # every gate in compute_final_success() passed
    INCOMPLETE = "INCOMPLETE"  # the pipeline ran, but a gate did not pass
    FAILED     = "FAILED"      # a required stage raised, or the model was unavailable


def read_build_log(output_dir: Path) -> dict:
    """Return output_dir/BUILD_LOG.json as a dict ({} if missing or corrupt)."""
    log_path = Path(output_dir) / BUILD_LOG
    if not log_path.exists():
        return {}
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return log if isinstance(log, dict) else {}


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

    # Exceptions raised by stages: {"stage", "error", "fatal"}. A fatal one
    # (required stage, or model unavailable) makes the verdict FAILED.
    stage_errors:  list = field(default_factory=list)

    # The final verdict and why it was reached, set by save_log() from
    # compute_verdict() at the end of the pipeline.
    verdict:        Verdict = Verdict.INCOMPLETE
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

    def record_stage_error(self, stage: str, error: Exception, fatal: bool) -> None:
        self.stage_errors.append({
            "stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "fatal": fatal,
        })

    def compute_verdict(self) -> tuple[Verdict, str]:
        """The single verdict function: FAILED if a fatal stage error was
        recorded, else SUCCESS or INCOMPLETE from compute_final_success().
        Returns (verdict, reason); reason is empty on SUCCESS."""
        fatal = next((e for e in self.stage_errors if e["fatal"]), None)
        if fatal:
            return Verdict.FAILED, f"{fatal['stage']} stage raised {fatal['error']}"
        passed, reason = self.compute_final_success()
        return (Verdict.SUCCESS if passed else Verdict.INCOMPLETE), reason

    def save_log(self):
        """Persist the pipeline log to output_dir/BUILD_LOG.json, with the
        final verdict from compute_verdict(). Checkpoint keys already in the
        file (completed_stages) are kept, so a finished build resumes as done."""
        self._dedupe_files()
        self.verdict, reason = self.compute_verdict()
        self.success = self.verdict is Verdict.SUCCESS
        self.success_reason = reason

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log = read_build_log(self.output_dir)
        log.update({
            "instruction":    self.instruction,
            "started_at":     self.started_at,
            "files":          self.files_written,
            "verdict":        self.verdict.value,
            "success":        self.success,
            "success_reason": reason or "all gates passed",
            "stage_errors":   self.stage_errors,
            "packaged":       self.packaged,
        })
        (self.output_dir / BUILD_LOG).write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )

    def summary(self) -> str:
        mark     = "✓" if self.verdict is Verdict.SUCCESS else "✗"
        status   = f"{mark} {self.verdict.value}"
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
