---
name: readable-module-code
description: Write or refactor code so it is easy to understand, beginner-friendly, modular, and maintainable. Use when the user asks for clear code, readable code, meaningful variable names, comments for each major part, step-by-step explanations, notebook-to-module cleanup, or code organized into functions/classes/modules.
---

# Readable Module Code

## Core Goal

Produce code that another developer can read quickly, modify safely, and explain without guessing. Prefer clarity over cleverness.

## Workflow

1. Read the surrounding code or data shape before proposing structure.
2. Identify the main responsibilities in the task.
3. Split responsibilities into small functions, classes, or modules.
4. Use names that describe intent, not implementation trivia.
5. Add short comments before non-obvious blocks.
6. Keep public functions easy to call and easy to test.
7. Show a small usage example when it helps the user apply the code.

## Structure Rules

- Use one function for one clear job.
- Keep orchestration code separate from helper code.
- Put configuration values near the top of the file or cell.
- Prefer pure helper functions when possible.
- Avoid hidden global state unless the existing project already relies on it.
- Avoid deeply nested logic; return early when it improves readability.
- Keep notebook cells runnable in order if working in a notebook.

## Data Pipeline Rules

- Explain the overall flow before or near the code: input -> transform -> output.
- Add preview/debug helpers for every important intermediate artifact.
- Print compact summaries before showing long examples.
- Keep preview code separate from production functions.
- Prefer small checkable steps over one large end-to-end block.
- For retrieval work, show the metadata that affects matching, filtering, and tracing.

## Naming Rules

- Use concrete nouns for data: `qa_records`, `topic_aliases`, `normalized_question`.
- Use verbs for actions: `load_dataset`, `resolve_topic`, `build_qa_chunk`.
- Avoid vague names: `data2`, `temp`, `result`, `x`, `info`.
- Name booleans as questions or states: `has_valid_answer`, `is_generic_question`.
- Name collections in plural form: `documents`, `questions`, `dedup_keys`.

## Comment Rules

Use comments to explain why a block exists or how a tricky decision is made. Do not comment obvious assignments.

Good:

```python
# Prefer the identification answer because dataset keywords can be aliases or noisy labels.
topic = resolve_topic_from_questions(sample)
```

Avoid:

```python
# Assign topic
topic = resolve_topic_from_questions(sample)
```

## Module Pattern

When writing new code, prefer this layout:

```python
# 1. Imports

# 2. Configuration

# 3. Small generic helpers

# 4. Domain-specific helpers

# 5. Main build/run function

# 6. Debug or example usage
```

For notebook code, use section headers that match this layout so the user can later move code into `.py` files.

## Refactor Checklist

Before finishing, check:

- Can each function be summarized in one sentence?
- Are important assumptions visible near the code that uses them?
- Are variable names meaningful without reading five lines above?
- Are comments helpful but not noisy?
- Is the main flow visible without scrolling through helper internals?
- Is the code easy to extract into modules later?

## Response Style

When proposing code, briefly explain each section before or after the snippet. When editing existing code, keep changes scoped and preserve the user's current architecture unless the user asks for a larger rewrite.
