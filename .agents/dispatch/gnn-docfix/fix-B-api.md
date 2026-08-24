# B-api — GNN documentation accuracy fixes (SAFE DOC-ONLY EDITS)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Mission
Apply the documentation corrections listed below. These are SAFE, DOCUMENTATION-ONLY
edits: change markdown prose, code examples, import references, file paths, counts,
and doc metadata. Do NOT change any source code (.py), test logic, config, or
dependency. The fixes were verified against the current tree (imports resolve; paths
are tracked); your job is to apply them to the .md files exactly.

## Rules
- Edit ONLY the .md files named below. Do NOT touch src/**, scripts/**, pyproject.toml,
  tests/**, or any .py.
- Preserve surrounding formatting/markdown. Make the minimal change (one line / one
  token) per fix.
- For imports: replace the broken module path with the verified-correct one below.
- For counts: set the number to the value stated below (verified by git ls-files/wc).
- For prose claims that are wrong/unverifiable and no exact replacement exists: reword
  minimally to be accurate (e.g. mark as illustrative, or remove the fabricated claim).
- HARD RULE: do NOT commit, stage, or push. Leave changes uncommitted.

## Specific fixes to apply
FIXES:
1. doc/api/README.md:153 — block importing `from gnn.render import PyMDPRenderer, RxInferRenderer` — replace with real API: `from render import PyMDPRenderer` and `from render import render_gnn_to_pymdp, render_gnn_to_rxinfer` (or mark the block illustrative; prefer using real names: PyMDPRenderer exists, RxInferRenderer does not; use render_gnn_to_rxinfer).
2. doc/llm/README.md:222 — comment "From pipeline step 11 (11_llm.py)" -> "step 13 (13_llm.py)" (LLM is script 13_llm.py)
3. doc/llm/README.md:223,227 — `from src.llm import get_processor` / `get_processor()` -> `from src.llm import get_global_processor` / `get_global_processor()`
4. doc/llm/README.md:264,265,268 — env vars ENABLE_FALLBACK / ENABLE_STREAMING / DEFAULT_TEMPERATURE are NOT consumed in src/ — annotate them as not-currently-consumed or remove (prefer a short note that these are reserved/not enforced).
5. doc/llm/README.md:269 — DEFAULT_MAX_TOKENS is a module constant, not read from env — adjust wording (it is a default, not an env var).
6. doc/security/README.md — illustrative code blocks with non-existent paths: correct or annotate each. Specifically:
   - ~34 header `# src/gnn/security/validator.py` -> mark as illustrative OR correct to real security module (src/security/processor.py). Prefer annotating as illustrative if no exact match.
   - ~88 `# src/llm/security/prompt_sanitizer.py` -> illustrative (src/llm has no security/ subdir) — annotate.
   - ~128 `# src/mcp/security/secure_server.py` -> illustrative — annotate.
   - ~268 `# src/security/audit.py` -> does not exist — annotate or remove.
   - ~238 `# tests/security/test_security.py` -> real path is src/tests/security/test_security_functional.py — correct the path.
7. doc/security/security_framework.md — `from gnn.security import ...` and `from gnn.auth import ...` do not exist — annotate these example imports as illustrative/aspirational (do not claim they are importable).
8. doc/security/security_framework.md ~604/~648 — `--secure-mode` flag and `src.main:app` gunicorn target don't exist — annotate as illustrative.
9. File-count metadata:
   - doc/mcp/AGENTS.md:20 "Files: 3" -> 5 (AGENTS, README, SPEC, fastmcp, gnn_mcp_model_context_protocol)
   - doc/llm/AGENTS.md:20 "Files: 1" -> 4 (AGENTS, README, SPEC, security_guidelines)
   - doc/security/AGENTS.md:20 "Files: 2" -> 10 (AGENTS, README, SPEC, codex_security_remediation, compliance_guide, incident_response, monitoring, security_assessment, security_framework, vulnerability_assessment)


## Verification
After editing, run:
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write   (must stay green)
- uv run --extra dev python scripts/check_repo_terminology.py --strict   (must stay clean — do NOT use banned words: legacy/stub/placeholder/deprecated)
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

## Report
Write a concise report to /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docfix/REPORT-fix-B-api.md listing each file you changed and the specific
edit(s) applied. Reply with only the absolute path to your report.
