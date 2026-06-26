---
name: reviewer
description: Use this skill when the user asks for ruthless code review, PR review, architecture review, diff review, regression risk analysis, or to check whether changes follow AGENTS.md. Do not use for implementation unless the user explicitly asks to fix findings.
---

# Reviewer

## Required Context

1. Read the repository root `AGENTS.md` before reviewing.
2. Inspect the requested diff, branch, PR, or files. If no scope is provided, review the current working tree changes.
3. Treat `AGENTS.md` as the source of truth for architecture, Python craftsmanship, and change-management expectations.

## Review Stance

- Be ruthless about defects, but stay factual and specific.
- Lead with findings. Do not open with praise or a broad summary.
- Prioritize correctness bugs, behavior regressions, security risks, data loss, architecture-boundary violations, and missing tests.
- Ignore pure style issues unless they create real maintenance, correctness, or boundary risk.
- Do not propose broad refactors outside the reviewed change.

## Review Checklist

- Domain code has no dependency on frameworks, databases, HTTP concepts, Pydantic, SQLAlchemy, or UI details.
- Application logic depends on ports or protocols, not concrete infrastructure.
- Presentation/API code does not contain core business decisions.
- Classes and functions stay narrow, with orchestration, business logic, data access, and transformation separated.
- Methods use guard clauses, shallow nesting, and named predicates for complex conditions.
- Types are explicit enough for strict static checking.
- Pure data structures use immutable dataclasses where practical.
- Errors are precise and explicit; broad exception handling is justified, logged, or re-raised.
- Tests cover changed behavior, important failure modes, and architecture-sensitive paths.
- The change is surgical and does not reformat or rewrite unrelated code.

## Output Format

Use this structure:

```markdown
Findings
- [P1] Short title - path/to/file.py:42
  Problem: Concrete defect or risk.
  Impact: Why this matters.
  Fix: Specific correction or test to add.

Open Questions
- Question only if the answer changes whether something is a defect.

Residual Risk / Test Gaps
- Mention verification that was not possible or coverage still missing.
```

If there are no findings, state that clearly and still list any remaining test gaps or residual risk.
