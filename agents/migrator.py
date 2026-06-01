"""
Migrator Agent
Reads PLAN.md's data model section and generates database schema/migration
files: Alembic + SQLAlchemy for Python, Prisma for TypeScript.
Runs after the Architect and before the Coder so migrations are in place.
"""

from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, HANDLERS
from .shared.memory import Project

SYSTEM = """You are a database engineer specialising in schema design and migrations.

Given a project plan with data models, generate all database-related files:

For **Python / SQLAlchemy** projects:
1. `db/models.py` — SQLAlchemy ORM models for every entity
2. `db/database.py` — engine, SessionLocal, get_db dependency
3. `alembic.ini` + `alembic/env.py` — Alembic setup pointing at models
4. `alembic/versions/0001_initial.py` — initial migration (create all tables)
5. `db/seed.py` — realistic sample data using the ORM

For **TypeScript / Prisma** projects:
1. `prisma/schema.prisma` — all models, relations, indexes, enums
2. `prisma/seed.ts` — realistic seed data
3. `lib/db.ts` or `src/db/client.ts` — PrismaClient singleton

Schema rules (apply to all):
- UUID primary keys (uuid_generate_v4() for Postgres, CUID for Prisma)
- created_at / updated_at timestamps on every table
- deleted_at soft-delete on main entities
- Proper indexes on all foreign keys and high-cardinality filter fields
- Prefer Postgres for production; SQLite is acceptable for dev-only projects
- Relations: define both sides, add cascade delete where appropriate

Infer the database technology from the research brief and plan.
"""


def run(project: Project) -> Project:
    print("  [Migrator] Generating database schema and migrations …")

    out = str(project.output_dir)

    task = f"""Project: {project.instruction}

Research brief (excerpt):
{project.research[:2000]}

Architecture plan (excerpt):
{project.plan[:2000]}

Output directory: {out}

1. Read {out}/PLAN.md to understand the full data model
2. Generate all database files (models, migrations, seed, connection layer)
   — write them into the correct paths as defined in PLAN.md
3. For Python: initialise Alembic if the plan uses SQLAlchemy
4. For TypeScript: write the Prisma schema and seed

Write every file completely — no placeholders.
"""

    written = []

    def tracked_write(path: str, content: str) -> str:
        result = HANDLERS["write_file"](path, content)
        written.append(path)
        return result

    handlers = {
        "read_file":   HANDLERS["read_file"],
        "write_file":  tracked_write,
        "list_files":  HANDLERS["list_files"],
        "run_command": HANDLERS["run_command"],
    }
    tools = [READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND]

    def on_call(name, args):
        if name == "write_file":
            print(f"    [write] {args.get('path', '')}")
        elif name == "run_command":
            print(f"    [run]   {args.get('command', '')[:80]}")
        else:
            print(f"    [{name}]")

    run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    project.files_written.extend(written)
    print("  [Migrator] Done.\n")
    return project
