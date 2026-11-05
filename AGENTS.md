# Agent Instructions & ECC Skill Compliance

> **Project**: Research on OCR combined with Large Language Models for Information Extraction from Vietnamese Retail Receipts  
> **Project Root**: `D:\NCKH\OCR_LLM_Vietnamese_Receipt_Extraction`  
> **ECC Skills Library**: `D:\NCKH\ECC\skills` (262 modular skills)  
> **Virtual Environment**: `.\.venv\Scripts\python.exe` / `.\.venv\Scripts\pytest.exe`  
> **Test Baseline**: 30/30 passing tests (verified via pytest 9.1.1 on Python 3.10.11)

---

## 1. Mandatory Pre-Task Skill Discovery & Compliance

Every agent working in this repository **must** read and follow these instructions before planning or executing any task. These instructions persist across tasks, context compactions, and new sessions.

### A. Efficient Skill Discovery (Do NOT read all 262 skills)
Before writing code or designing a plan:
1. **Analyze Requirements**: Extract domain, components, and task keywords (e.g., regex, LLM prompt, evaluation, image preprocessing, API).
2. **Search ECC Library**: Search `D:\NCKH\ECC\skills` using relevant keywords. Use targeted queries (e.g., directory names or frontmatter search).
3. **Core Domain Mapping**:
   - **Receipt / Structured Text Extraction**: `regex-vs-llm-structured-text`, `nutrient-document-processing`
   - **LLM Routing, Cost & Token Budgeting**: `cost-aware-llm-pipeline`, `prompt-optimizer`
   - **Evaluation, Benchmarks & Metrics**: `eval-harness`, `benchmark`, `benchmark-optimization-loop`
   - **Python Standards & Typing**: `python-patterns`
   - **Testing & TDD**: `python-testing`, `tdd-workflow`
   - **Verification & Quality Gates**: `verification-loop`
   - **Security & Hygiene**: `security-scan`
   - **Scientific Methodology & Literature**: `scientific-thinking-literature-review`, `research-ops`
   - **Compliance & Discovery Tools**: `skill-comply`, `skill-scout`
4. **Mandatory Complete Reading**: Read the **complete** `SKILL.md` file (and any required references) for all selected skills. Do not rely on summaries, partial line reads, or assumptions.
5. **Cross-Cutting Skills**: Always apply quality gating (`verification-loop`), testing standards (`python-testing`), and Python idioms (`python-patterns`) on every code change.
6. **Dynamic Scope Changes**: If the scope expands or a new technical domain is encountered during work, pause and discover/read the new relevant skills before proceeding.

---

## 2. Standard Task Lifecycle

Every future task must adhere to the 8-stage lifecycle:

$$\text{Understand} \longrightarrow \text{Discover Skills} \longrightarrow \text{Read Instructions} \longrightarrow \text{Plan} \longrightarrow \text{Implement} \longrightarrow \text{Verify} \longrightarrow \text{Review} \longrightarrow \text{Report}$$

### Before Implementation (Planning Gate)
Explicitly identify:
- **Task Objective & Acceptance Criteria**
- **Selected ECC Skills** and the explicit constraints/patterns they mandate
- **Files & Components** to be inspected or edited
- **Verification Strategy** (targeted unit tests, regression tests, edge-case checks)
- **Risks & Constraints** (backward compatibility, performance, cost)

### During Implementation (Execution Discipline)
- **Minimal, Targeted Edits**: Prefer the smallest correct change. Avoid gratuitous refactoring.
- **Preserve Existing Behavior**: Never break or modify working code or existing tests without explicit justification.
- **No Unsolicited Dependencies**: Do not install packages or alter `pyproject.toml` / `requirements.txt` unless requested and justified.
- **Never Fake Implementations**: Do not write mocks or dummy logic solely to bypass tests without genuine implementation.
- **Preserve User Changes**: Never overwrite user work.

---

## 3. Verification & Final Compliance Gate

Before declaring any task complete:
1. **Run Actual Commands**: Execute tests using the project virtual environment:
   ```powershell
   Set-Location "D:\NCKH\OCR_LLM_Vietnamese_Receipt_Extraction"
   .\.venv\Scripts\pytest.exe
   ```
2. **Follow the 4-Phase Verification Loop**:
   - **Phase 1: Syntax & Build**: Verify syntax and imports compile cleanly.
   - **Phase 2: Types & Contracts**: Verify Pydantic v2 schemas and type annotations.
   - **Phase 3: Lint & Hygiene**: Ensure no lingering temporary debug statements or secrets.
   - **Phase 4: Test Suite**: All tests must pass (maintain baseline of ≥30 passing tests).
3. **Honest Reporting**:
   - Explicitly list which skills were read and applied.
   - Show exact command output and test counts.
   - Disclose any unverified assumptions, skipped checks, or remaining limitations.
   - Never fabricate or assume success.

---

## 4. Persistent Compliance Record

- **Verified ECC Skill Root**: `D:\NCKH\ECC\skills` (262 skills verified)
- **Active Instruction Paths**:
  - `D:\NCKH\OCR_LLM_Vietnamese_Receipt_Extraction\AGENTS.md` (Primary project rule)
  - `D:\NCKH\OCR_LLM_Vietnamese_Receipt_Extraction\GEMINI.md` (Antigravity alias)
- **Python Environment**: `D:\NCKH\OCR_LLM_Vietnamese_Receipt_Extraction\.venv` (Python 3.10.11, pytest 9.1.1)
- **Core Baseline**: 30 passing unit and integration tests across schemas, validator, OCR factory, preprocessor, and regex baseline.
