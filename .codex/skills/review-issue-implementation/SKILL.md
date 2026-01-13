---
name: review-issue-implementation
description: Review and rate code implementations against issue cards and ExecPlans using git diffs, AGENTS.md, and PLANS.md. Use when a user asks for an implementation review, compliance check, regression risk assessment, or to validate changes against one or more issue cards or ExecPlans.
---

# Review Issue Implementation

## Overview

Use this skill to verify that code changes satisfy issue cards and ExecPlans, assess regression risk, and provide a clear rating with prioritized findings. Look for implementation mistakes, things that can lead to errors, or bad practices. We do not want over engineering, nor cluttering too much. We must follow the DRY principle and the best practices and guidelines.

## Inputs (required)

- ISSUE_PATH: Path to the issue card file (required).
- EXECPLAN_PATH: Path to the ExecPlan file (required).
- DIFF_RANGE: Diff range to review (optional). Default to `STAGED` (staged changes) when not provided. Use `WORKING_TREE` for working tree vs `HEAD`, or provide an explicit range (for example, `main..feature-branch`, `<base>..<head>`, or a commit hash).
- Require ISSUE_PATH and EXECPLAN_PATH; ask for them if missing or ambiguous.

## Inputs (optional)

- SCOPE_NOTES: Focus area or high-risk surfaces to emphasize (for example, auth, data migrations, UI flows).
- TEST_EVIDENCE: Commands run and key results, if already executed by the user.

## Workflow (default)

1. Intake and scope
   - Collect issue card path(s), ExecPlan path(s), and the diff range (working tree vs HEAD, commit range, or branch).
   - If any are missing, ask for the minimum needed to review (prefer file paths and a diff range).
2. Load governing docs
   - Read `AGENTS.md` and `PLANS.md` from the repo root if they are not already in context.
   - Read each issue card and ExecPlan; extract explicit requirements, acceptance criteria, and any approved scope expansions.
3. Diff-first review
   - Use `git status`, `git diff --stat`, and `git diff` (plus `git diff --cached` if staged) to identify touched files.
   - Open files only to validate behavior or get precise line references.
4. Requirements alignment
   - Map each requirement to the actual change; mark gaps, partial coverage, or behavior mismatches.
   - If a refactor or file move appears, verify explicit approval in the issue/plan; otherwise flag as scope risk.
5. Risk and regression analysis
   - Evaluate correctness, safety, and unintended side-effects (security, data integrity, background jobs, UI flows).
   - Look for masked invalid states; changes should fail explicitly with clear errors.
6. Tests and verification
   - Note new/updated tests, and gaps. Highlight required checks from `AGENTS.md` if not evident.
7. Output
   - Produce findings first (ordered by severity, with file/line references).
   - Provide rating and risk summary using the rubric in `references/review-checklist.md`.
   - Follow with open questions/assumptions and a brief change-summary (only after findings).

## Output order (use this sequence)

1. Findings (severity order; include file path + line)
2. Open questions / assumptions
3. Rating (per category + overall status + risk level)
4. Change summary (brief)

## Resources

Use `references/review-checklist.md` for the rating rubric, risk prompts, and repo-specific checks.