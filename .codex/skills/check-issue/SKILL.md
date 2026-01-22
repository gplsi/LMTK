---
name: check-issue
description: Validate an issue card and ExecPlan against AGENTS.md and PLANS.md, and produce junior-proof, unambiguous edits. Use when Codex needs to QA or align issue cards/ExecPlans for this repo.  Requires ISSUE_PATH and EXECPLAN_PATH.
---

# Purpose
Ensure an issue card and its ExecPlan comply with AGENTS.md and PLANS.md so the work is safe to hand to a junior developer, with no ambiguity or underspecified steps. Look for implementation mistakes, things that can lead to errors, or bad practices. We do not want over engineering, nor cluttering too much. We must follow the DRY principle and the best practices and guidelines.

## Inputs (required)

- ISSUE_PATH: Path to the issue card file (required).
- EXECPLAN_PATH: Path to the ExecPlan file (required).
- Require ISSUE_PATH and EXECPLAN_PATH; ask for them if missing or ambiguous.

## Procedure
1. Read AGENTS.md and PLANS.md first (refresh fully if needed).
2. Read ISSUE_PATH and EXECPLAN_PATH fully.
3. Produce:
   - A compliance checklist mapped to the exact AGENTS.md / PLANS.md requirements.
   - A gap list (missing/unclear/conflicting) with severity, including a junior-proofness audit (undefined terms, vague steps, missing file paths, missing validation, or unclear acceptance). Include this under Summary.
   - Exact proposed edits (patch-style or "replace this section with ...").
   - A junior-proof acceptance criteria + validation steps (tests, logs, expected outputs) with explicit inputs and observable outputs.
4. Resolve ambiguity by inspecting the repo (do not ask the user unless blocked).
5. Use assets/align_issue_card_rubric.md as the output template when helpful.

## Output format
- Summary (include junior-proof gaps)
- Compliance matrix (Requirement -> Evidence -> Status -> Fix)
- Proposed edits
- Final "ready to implement" checklist