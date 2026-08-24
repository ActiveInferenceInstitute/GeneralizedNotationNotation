# REPORT — api-mcp (GNN documentation-vs-code audit, REPORT-ONLY)

Repo: `/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`
Region: `doc/api/`, `doc/mcp/`, `doc/lsp/`, `doc/llm/`, `doc/security/`
Source authority: `src/api/`, `src/mcp/`, `src/lsp/`, `src/llm/`, `src/security/`
Date: 2026-08-24

## Scope notes / method
- Commands/imports/paths verified with `git ls-files`, `grep`, and `.venv/bin/python -c` with `PYTHONPATH=src`.
- `doc/api/comprehensive_api_reference.md` explicitly disclaims sections *below* its accuracy box as
  "illustrative sketches — do not import unless confirmed in src/", so those are not reported (they are
  self-flagged). Only its "Quick start (current gnn exports)" box and `doc/api/README.md` (no disclaimer)
  are treated as authoritative claims.
- `doc/mcp/fastmcp.md` and `doc/mcp/gnn_mcp_model_context_protocol.md` are external/educational analyses
  (FastMCP upstream guide; a Perplexity export) with illustrative third-party/vision examples, not GNN
  API claims — not reported except where they name GNN source paths.

## Clean regions (no findings)
- **doc/mcp/README.md** — Verified accurate. Every documented MCP tool name exists in the registries:
  `parse_gnn_content`, `validate_gnn_content`, `process_gnn_directory`, `get_gnn_documentation`
  (`src/gnn/mcp.py`); `process_export`, `export_single_gnn_file`, `list_export_formats`,
  `validate_export_format` (`src/export/mcp.py`); `process_visualization`, `get_visualization_options`,
  `list_visualization_artifacts`, `get_visualization_module_info` (`src/visualization/mcp.py`);
  `process_ontology`, `validate_ontology_terms`, `extract_ontology_annotations`,
  `list_standard_ontology_terms` (`src/ontology/mcp.py`); `process_llm`, `analyze_gnn_with_llm`,
  `generate_llm_documentation`, `get_llm_providers` (`src/llm/mcp.py`); `process_gui`,
  `list_available_guis`, `oxdraw.convert_to_mermaid`, `oxdraw.convert_from_mermaid`,
  `oxdraw.check_installation` (`src/gui/mcp.py` lines 234-236). `get_pipeline_status` registered at
  `src/pipeline/mcp.py:365`; `process_render` at `src/render/mcp.py:183`. CLI subcommands
  `add_parser("list"/"execute"/"resource"/"status"/"info"/"diagnostics"/"server")`, transports
  `choices=["stdio","http"]`, `--host`/`--port` all present in `src/mcp/cli.py`. `GNN_MCP_TOKEN` consumed
  (`src/mcp/server_http.py:55`). `src/mcp/__init__.py` exports `initialize`, `mcp_instance`,
  `MCP`, `MCPTool`, `MCPResource`; `MCP` has `register_tool`/`execute_tool`/`list_available_tools`/
  `get_tool_info`/`.tools`. `mcp.capabilities` handled in `server_http.py:263` and `server_stdio.py:308`.
- **doc/lsp/** — No `doc/lsp/` directory exists in the repo (region is served by `src/lsp/README.md`).
  LSP capabilities are accurate: `create_server`, `start_server`, `_publish_diagnostics`, `_get_hover`
  all exist in `src/lsp/__init__.py`; `textDocument/didOpen`, `didSave`, `hover` map correctly. The
  `python -m src.lsp` entry imports OK. Real finding is only the INFO count below.
- **doc/security/** narrative docs aside from the flagged items below.
- **doc/llm/README.md** overall env-key handling: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
  `PERPLEXITY_API_KEY`, `DEFAULT_PROVIDER`, `OPENAI_ORG_ID`, `OPENAI_BASE_URL`,
  `OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME`, `OLLAMA_MODEL`, `OLLAMA_DISABLED`, `OLLAMA_TEST_MODEL`
  are each consumed in `src/llm/`. `src/llm/.env.example` exists (697 bytes). `initialize_global_processor`,
  `AnalysisType`, `ProviderType`, `LLMConfig`, `LLMMessage`, `LLMProcessor`, `get_global_processor`,
  `analyze_gnn_file_with_llm` are exported from `src/llm/__init__.py`. `AnalysisType` members
  (`SUMMARY..SEARCH_ENHANCED`) and `ProviderType` (`OPENAI/OPENROUTER/PERPLEXITY/OLLAMA`) verified in
  `src/llm/llm_processor.py:37-43` and `src/llm/providers/base_provider.py:23-26`.

---

## Findings

### doc/llm/README.md
| Line | Severity | Finding |
|------|----------|---------|
| doc/llm/README.md:222 | ERROR | Comment claims "From pipeline step 11 (11_llm.py)". LLM is step **13**; `src/11_llm.py` does not exist — `src/11_render.py` is render, `src/13_llm.py` is LLM (`grep '"13_llm"' src/main.py:23`). Fix: change to "step 13 (13_llm.py)". |
| doc/llm/README.md:223, 227 | ERROR | `from src.llm import get_processor` then `processor = get_processor()`. `get_processor` is NOT exported from `src.llm`; the exported symbol is `get_global_processor` (`src/llm/__init__.py:69` aliases `get_processor as get_global_processor`; `'get_processor' in m.__all__` → False, `'get_global_processor'` → True). Per charter this would raise ImportError. Fix: `from src.llm import get_global_processor`. |

### doc/api/README.md
| Line | Finding | Severity |
|------|---------|----------|
| doc/api/README.md:153 | `from gnn.render import PyMDPRenderer, RxInferRenderer` — module `gnn.render` does NOT exist (`ModuleNotFoundError: No module named 'gnn.render'`; `ls src/gnn/render` → none). Correct location is `src/render/` (`from render import ...`), and top-level `render` exports `PyMDPRenderer` but NOT `RxInferRenderer` (that symbol is an internal class in `src/render/rxinfer/rxinfer_renderer.py`, not in `render.__init__` — `ImportError: cannot import name 'RxInferRenderer' from 'render'`). Fix: correct to `from render.pymdp import PyMDPRenderer`-style real paths and name real helpers (`render_gnn_to_pymdp`, `render_gnn_to_rxinfer`), or mark the block illustrative. | ERROR |

### doc/security/*.md — illustrative code blocks reference non-existent source modules/files
| Location | Finding | Severity |
|----------|---------|----------|
| doc/security/README.md (~line 34) | Code block header `# src/gnn/security/validator.py` and body imports nothing (defines `validate_gnn_file_security`) — `src/gnn/security/` does not exist (`git ls-files` → none; `ls src/gnn/security` → doesn't exist). | WARNING |
| doc/security/README.md (~line 88) | `# src/llm/security/prompt_sanitizer.py` — `src/llm/security/` does not exist (`git ls-files` → none). | WARNING |
| doc/security/README.md (~line 128) | `# src/mcp/security/secure_server.py` — `src/mcp/security/` does not exist. | WARNING |
| doc/security/README.md (~line 268) | `# src/security/audit.py` — file not tracked (`git ls-files` → none). `# src/security/` has no `audit.py`. | WARNING |
| doc/security/README.md (~line 238) | `# tests/security/test_security.py` — no such tracked file. | WARNING |
| doc/security/security_framework.md (~lines 36, 199) | `from gnn.security import SecureGNNParser, SecurityConfig` and `from gnn.auth import AuthenticationManager, Role, Permission` — `src/gnn/security` and `src/gnn/auth` do not exist. | WARNING |
| doc/security/security_framework.md (Dockerfile ~line 604, CMD 648) | Recommends `python src/main.py --secure-mode` and `gunicorn -- src.main:app`. `--secure-mode` flag is not handled in `src/main.py` (grep → none) and `src/main.py` is a CLI pipeline orchestration entry, not a FastAPI `app` module. | WARNING |

### doc/*/AGENTS.md — stale module AGENTS.md "Contents" file-count stubs
| Location | Finding | Severity |
|----------|---------|----------|
| doc/api/AGENTS.md:20 | "Files: README.md, AGENTS.md, comprehensive_api_reference.md" — correct (matches those 3 + SPEC). No deviation. | (OK) |
| doc/mcp/AGENTS.md:20 | "Files: 3" — directory actually has 5 `.md` (AGENTS, README, SPEC, fastmcp, gnn_mcp_model_context_protocol). Stale count. | INFO |
| doc/llm/AGENTS.md:20 | "Files: 1" — directory actually has 4 `.md` (AGENTS, README, SPEC, security_guidelines). Stale count. | INFO |
| doc/security/AGENTS.md:20 | "Files: 2" — directory actually has 10 `.md` (AGENTS, README, SPEC, codex_security_remediation, compliance_guide, incident_response, monitoring, security_assessment, security_framework, vulnerability_assessment). Stale count. | INFO |

### doc/llm/README.md — env vars documented but not consumed anywhere in code
| Lines | Finding | Severity |
|-------|---------|----------|
| doc/llm/README.md:264, 265, 268 | `ENABLE_FALLBACK`, `ENABLE_STREAMING`, `DEFAULT_TEMPERATURE` are listed as configuration environment variables, but no `os.getenv("ENABLE_FALLBACK")` / `ENABLE_STREAMING` / `DEFAULT_TEMPERATURE` exists anywhere in `src/` (grep → none; only present in `.env.example`). | WARNING |
| doc/llm/README.md:269 | `DEFAULT_MAX_TOKENS` documented as an env var, but in code it is a module constant (`DEFAULT_MAX_TOKENS = 8000` in `src/llm/llm_operations.py:31`), not read from the environment. | WARNING |

### doc/mcp/gnn_mcp_model_context_protocol.md — illustrative, not flagged
Lines 383-589 contains hypothetical MCP tool/method names (`gnn/findModelsByOntologyTerm`,
`gnn/validateModel`, `gnn/translateToPyMDP`, `@app.get_resource`, `from mcp.server import Server`, etc.)
that do not exist in the GNN codebase's MCP registries. These read as an external "GNN +
MCP vision" analysis (Perplexity export) rather than a claim about the current API, so they are not
reported as errors; if maintainers want this file to be buildable/reference-accurate, it deserves an
"illustrative, not part of the GNN codebase" banner.

---

## Summary
- Regions clean: `doc/mcp/README.md` (full tool registry cross-checked), `doc/api/comprehensive_api_reference.md`
  (accuracy box + quick-start exports all match `src/gnn/__init__.py`), `doc/lsp/` surface (via src/lsp).
- 6 genuine ERRORs: `11_llm.py`/step-11 (doc/llm), `get_processor` not exported (doc/llm), `gnn.render` /
  `RxInferRenderer` broken import (doc/api), plus the 3 grouped as warnings above and the un-consumed env vars.
- Recommended fixes are one-line doc corrections paired to each finding above. No source files were
  modified, staged, or committed (REPORT-ONLY honored).