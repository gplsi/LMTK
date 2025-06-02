---
trigger: always_on
---

# AI Self-Correction and Repository-Specific Knowledge Base (.knowledge.md)

## Core Principle: Continuous Learning and Error Avoidance

Your primary objective within any repository is to avoid repeating past mistakes and to continuously learn from user feedback. This will be facilitated by maintaining and consulting a project-specific knowledge file named `.knowledge.md`.

## Phase 1: Pre-Computation Analysis (Before Any Code Modification or Generation)

1.  **Mandatory `.knowledge.md` Check:** Before generating or modifying any code, or providing solutions, you **MUST** first check for the existence of a file named `.knowledge.md` located in the root directory of the current repository.
2.  **Thorough Review of Contents:**
    *   If `.knowledge.md` exists, you **MUST** carefully read and internalize all information contained within it. This file lists short statements of previous errors, incorrect assumptions, or specific guidance relevant to this repository, as identified by the user.
    *   Treat each entry as a critical piece of information to guide your current task.
3.  **Proactive Error Prevention:** You **MUST** use the lessons from `.knowledge.md` to inform your responses and to actively prevent the repetition of documented mistakes. If a planned action or piece of information you are about to provide seems to contradict a lesson learned in `.knowledge.md`, you **MUST** explicitly state this, reference the relevant lesson, and ask for clarification or further instruction from me (the user) before proceeding.

## Phase 2: Learning from User-Identified Errors

If I (the user) identify an error, a misunderstanding, or an undesirable output that you have produced within the current repository:

1.  **Acknowledge and Prepare to Update Knowledge:** Politely acknowledge my feedback. State your intention to record this learning opportunity in the `.knowledge.md` file to improve future interactions in this repository.
2.  **Check/Manage `.knowledge.md` File:**
    *   Verify if `.knowledge.md` already exists in the repository root.
    *   **If `.knowledge.md` does not exist:** Inform me that the file is not present and propose to create it by saying, for example: "I see. To avoid this in the future, I will create a `.knowledge.md` file in this repository to log this learning. Is that okay?"
    *   **If `.knowledge.md` exists:** Proceed to formulate the new lesson.
3.  **Formulate the Lesson Entry:**
    *   Based *directly* on my correction and the context of the error, formulate a concise, single-line statement summarizing the mistake and the correct approach.
    *   The format should be clear and easy to understand, for example:
        *   `Mistake: [Brief description of the error/incorrect assumption]. Correction: [Brief description of the correct method/information/guideline].`
        *   _Example:_ `Mistake: Used library X for task Y. Correction: Library Z is preferred for task Y in this project due to [reason].`
        *   _Example:_ `Mistake: Assumed default configuration for module A. Correction: Module A requires `some_specific_setting = true` in this repository.`
4.  **Propose and Confirm Update to `.knowledge.md`:**
    *   Present the formulated lesson statement to me.
    *   Explicitly ask for my confirmation before writing to the file. For example: "I've noted the following lesson: '[The formulated lesson statement]'. Shall I add this to `.knowledge.md`?"
5.  **Update `.knowledge.md` (Upon User Confirmation):**
    *   If I confirm, and if you had to create the file (and I confirmed its creation), add the new lesson statement as a new line in `.knowledge.md`.
    *   If the file existed, append the new lesson statement as a new line.
    *   Inform me once the update is complete (e.g., "Okay, I've updated `.knowledge.md`.").
6.  **Apply Corrected Understanding:** Immediately apply this new understanding to revise your previous incorrect output or to continue the task with the corrected information.

## Ongoing Consideration:

*   The `.knowledge.md` file is a living document for each repository. Its consistent use is paramount for your effectiveness.
*   Always ensure entries are concise and directly reflect the errors and corrections discussed.
*   This rule prioritizes learning from direct user feedback within the context of individual repositories.
