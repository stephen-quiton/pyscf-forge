# Repository Guidelines

## Project Structure & Module Organization
This is a private version of the `pyscf-forge` repository.

For this local installation, the corresponding `pyscf` installation is located at:
`/home/sjquiton/sjq_dev/pyscf-sjq`

## Build, Test, and Development Commands

For anything that isn't changing just a few lines of code, or for plans that
come out of Plan mode, you (the current model) will be an orchestrator that feeds the implementation
details to Luna-MAX. Specifically, after planning, tell yourself the following prompt:

"""
TASK

Your job is to orchestrate and review the Luna max-thinking agent.

Focus especially on:

- Code quality
- Simple and understandable implementations
- Useful comments and documentation
- Idiomatic framework-specific best practices
- Meaningful tests

Tests should be brief, meaningful and well-documented. Edge cases that are not physically, scienfiically, or 
numerically meaningful should be limited. For example, extensive tests about API edge cases should not
be prioritized. For each major implementation, try to triage at most 10-15 tests that you think are the most meaningful,
and the fewer the better.

After reviewing Luna’s work, decide whether to:

1. Call Luna max-thinking again with the full context required to resolve the identified issues, or
2. Fix the issues yourself when doing so would require substantially fewer tokens.

START THE LUNA AGENT WITH:

codex exec \
  -m GPT-6-Luna \
  -c 'model_reasoning_effort="max"' \
  --ephemeral \
  -s workspace-write \
  -a never \
  'PLAN'
"""

Only do this Luna-MAX orchestration for implementing Plans specifically. For everything else, just use
whatever model is enabled.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, readable line lengths,
`snake_case` for functions, methods, and variables, and `PascalCase` for
classes. Keep numerical array shapes, units, and physical sign conventions
explicit in code and docstrings. No formatter or linter is configured, so
preserve surrounding formatting and use standard-library-compatible imports.

As much as possible, follow PySCF code-style.

## Testing Guidelines

Tests use both `unittest.TestCase` and pytest assertions/fixtures; follow the
style of the target test file. Name files `test_<feature>.py` and test methods
with descriptive `test_` names. Add regression coverage for numerical changes,
including tolerances appropriate to the calculation, and run focused tests
before the full suite. Avoid enabling slow tests by default in routine edits.

Keep tests brief and meaningful, with a docstring for readability. For instance, 
tests that probe API edge cases are almost certainly not needed.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, sentence-case summaries (for example,
`Add ability to release constraint 1` and `Bug fix`). Keep commits focused and
explain scientific or numerical behavior changes in the body when needed.
Pull requests should describe the motivation, affected methods, validation
commands and results, and any performance or precision tradeoffs. Include
updated documentation or examples when public behavior changes.

## Security & Configuration Tips

Do not commit generated caches, local data, or secrets. PySCF calculations can
be compute- and memory-intensive; use small cells/meshes for development and
document any required environment variables or external data in the PR.