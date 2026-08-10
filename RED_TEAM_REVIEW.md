# Red-Team Review: GNN Pipeline (GeneralizedNotationNotation v3.0.0)

**Review date**: 2026-07-31
**Reviewer**: Adversarial red-team review per `critical_review.red_team_review` (CogSecSkills)
**Target**: `GeneralizedNotationNotation` repository root (local checkout)
**Mode**: Read-only, defensive-only, no exploitation guidance

---

## BLUF (Bottom Line Up Front)

The GNN pipeline has a **compounding attack chain** from untrusted input to arbitrary code
execution on the host. The pipeline's core purpose — parsing GNN specification files and
rendering them into executable Python/Julia simulation scripts — means the design
**inherently executes attacker-influenced code**. The security module (Step 18) runs *after*
the execution step (Step 12), making it a post-incident forensics tool rather than a gate.
Additional critical findings: unauthenticated FastAPI endpoints with path-boundary validation
that can be bypassed via symlinks, `pickle.load` on untrusted input files, and
`ast.literal_eval` on parsed GNN parameter strings. The system is safe for trusted local
research use but **should not be exposed to network input or untrusted GNN files** without
remediation.

**Recommendation: GO-WITH-CONDITIONS** — see §5.

---

## 1. Artifact Map (Target Set)

| ID | Component | File | Trust Boundary |
|----|-----------|------|-----------------|
| AM-1 | Pipeline orchestrator | `src/main.py` | Runs 25 steps in sequence; Step 12 executes rendered code, Step 18 scans for security |
| AM-2 | Code executor | `src/execute/executor.py:1069` | Runs rendered `.py` via `subprocess.run([sys.executable, script])` |
| AM-3 | FastAPI server (v1) | `src/api/server.py` | No authentication; `--host 0.0.0.0` documented in docstring:11 |
| AM-4 | FastAPI server (v2) | `src/api/app.py` | No authentication; path validation via `resolve_repo_path` |
| AM-5 | Path validation | `src/api/path_utils.py:17-55` | `resolve_repo_path` checks `relative_to(repo_root)` after `.resolve()` |
| AM-6 | GNN parser | `src/gnn/parsers/markdown_parser_parameter.py:97` | `ast.literal_eval(value_str)` on parsed GNN file content |
| AM-7 | Binary parser | `src/gnn/parsers/binary_parser.py:46,67` | `pickle.load` / `pickle.loads` on untrusted file input |
| AM-8 | Security processor | `src/security/processor.py:53` | Scans `*.md` files only (line 53: `glob("*.md")`); AST scans `.py` (line 229) |
| AM-9 | MCP HTTP server | `src/mcp/server_http.py:53-93` | Bearer token auth optional; `GNN_MCP_ALLOW_INSECURE_LOCAL` bypass |
| AM-10 | Render→Execute flow | `src/execute/processor.py:1028` | `rglob("*.py")` discovers rendered scripts; executes all |
| AM-11 | LLM providers | `src/llm/providers/openrouter_provider.py:79` | API keys from env vars; outbound HTTP to OpenRouter/OpenAI/Perplexity |
| AM-12 | Distributed execution | `src/execute/distributed.py:42` | Ray/Dask cluster connection; `ray.init(address=self.address)` |

---

## 2. Adversary Profile

**Archetype**: External attacker with untrusted GNN file submission capability (e.g., via API
endpoint, shared model repository, or supply-chain model file).

| Field | Value |
|-------|-------|
| **Intent** | Achieve arbitrary code execution on the host running the GNN pipeline |
| **Capability** | Can craft GNN `.md` or `.pickle` input files; can reach API/MCP endpoints if exposed |
| **Access** | Network access to FastAPI/MCP endpoints (if bound to 0.0.0.0) OR file-system write access to `input/` directory |
| **Risk Tolerance** | High — code execution is the objective, not stealth |

---

## 3. Vulnerability Catalog

| ID | Category | Description | Evidence | Exploitability | Impact | Priority | Inherent/Remediable |
|----|----------|-------------|----------|----------------|--------|----------|---------------------|
| V-01 | Technical | **Rendered code execution without pre-scan**: Step 12 executes rendered `.py` scripts via `subprocess.run` before Step 18 security scans. Rendered code is derived from GNN input — attacker controls the template content. | `src/execute/executor.py:1069`; `src/security/processor.py:53` (scans `*.md` only, not render output) | 4 (feasible with moderate effort — craft GNN input that renders to malicious Python) | 5 (full RCE — `subprocess.run([sys.executable, script])` with no sandboxing) | **20 CRITICAL** | Remediable (run security scan on render output before execution) |
| V-02 | Technical | **`pickle.load` on untrusted input files**: Binary parser deserializes pickle files from disk without verification. `pickle.load` is arbitrary code execution by design. | `src/gnn/parsers/binary_parser.py:46` (`pickle.load(f)  # nosec B301`); `src/gnn/parsers/binary_parser.py:67` (`pickle.loads(binary_data)`); `src/gnn/schema_validator.py:210` | 4 (attacker supplies `.pkl`/`.pickle` file to pipeline) | 5 (arbitrary code execution via crafted pickle) | **20 CRITICAL** | Remediable (validate/verify pickle origin or replace with safe format) |
| V-03 | Technical | **`ast.literal_eval` on parsed GNN input**: GNN parameter values are evaluated via `ast.literal_eval` which, while safer than `eval`, can still cause DoS via deeply nested structures and will crash on malformed input. | `src/gnn/parsers/markdown_parser_parameter.py:97`; `src/gnn/schema.py:263`; `src/gnn/schema_validator.py:664`; `src/gnn/pomdp_extractor.py:731`; `src/export/format_exporters.py:92,183`; `src/render/generators.py:183` | 3 (requires crafted GNN `.md` file with malicious parameter values) | 3 (DoS via resource exhaustion; not RCE since `literal_eval` blocks calls) | **9 HIGH** | Remediable (add depth/size limits before `literal_eval`) |
| V-04 | Technical | **FastAPI server has no authentication**: Both `server.py` and `app.py` explicitly state "No authentication — designed for local research use." `server.py` docstring:11 documents `--host 0.0.0.0` as a valid invocation. Any network-reachable deployment allows unauthenticated pipeline job submission. | `src/api/server.py:6` ("No authentication"); `src/api/server.py:11` (`--host 0.0.0.0`); `src/api/app.py:100-107` (CORS); no auth middleware anywhere | 5 (trivial if endpoint is network-reachable) | 4 (unauthenticated pipeline execution = indirect code execution via V-01) | **20 CRITICAL** | Remediable (add API key auth; bind to 127.0.0.1 by default with warning) |
| V-05 | Technical | **Path validation bypass via symlinks**: `resolve_repo_path` calls `.resolve()` which follows symlinks, then checks `relative_to(repo_root)`. An attacker who can create a symlink inside the repo root pointing outside can bypass the boundary check. | `src/api/path_utils.py:32-35` (`resolved = candidate.resolve(strict=False)` then `resolved.relative_to(repo_root)`) | 3 (requires ability to create a symlink inside repo, e.g., via GNN output dir) | 4 (read/write outside repo root; path traversal) | **12 HIGH** | Remediable (reject symlinks or use `os.path.realpath` with symlink detection) |
| V-06 | Operational | **Security step runs after execution step**: Pipeline order is Step 11 (Render) → Step 12 (Execute) → … → Step 18 (Security). Security scan is forensic, not preventive. Rendered code runs before any security check. | `src/main.py` (AGENTS.md confirms order: "Step 11: Render → Step 12: Execute → Step 18: Security"); `src/security/processor.py:53` only scans `*.md` | 5 (inherent in design — no action needed by attacker) | 5 (security scan cannot prevent execution of malicious rendered code) | **25 CRITICAL** | Inherent (pipeline order would need redesign to scan render output before execution) |
| V-07 | Technical | **MCP HTTP auth bypass via env var**: `GNN_MCP_ALLOW_INSECURE_LOCAL=1` disables bearer auth for loopback clients. `is_loopback_client` only checks IP, not network path (no proxy header validation). | `src/mcp/server_http.py:63-67` (`allow_insecure_local_http`); `src/mcp/server_http.py:70-79` (`is_loopback_client` checks `ipaddress.is_loopback` only) | 3 (requires loopback access or SSRF to localhost) | 3 (unauthenticated MCP tool execution) | **9 HIGH** | Remediable (document warning; require explicit opt-in; add X-Forwarded-For check) |
| V-08 | Technical | **`safe_pickle_dump` writes pickle to output dir**: Pipeline writes pickle files to `output/` as simulation results. If `output/` is shared or served, these can be loaded by another pipeline run. | `src/execute/pymdp/pymdp_utils.py:74-91` (`pickle.dump(data, f)`) | 2 (requires output dir access) | 3 (poisoned pickle → RCE on next load via V-02) | **6 MEDIUM** | Remediable (sign pickle files; prefer JSON for output) |
| V-09 | Informational | **Error messages leak internal paths and stderr**: API job failure returns last 500 chars of stderr. Error responses expose internal file paths and exception details. | `src/api/processor.py:194-197` (`stderr_text[-500:]`); `src/api/path_utils.py:38` (error includes path) | 3 (submit failing job, read error response) | 2 (information disclosure: paths, library versions, stack traces) | **6 MEDIUM** | Remediable (sanitize error responses) |
| V-10 | Operational | **Unbounded glob discovery for render output**: `execute/processor.py:1028` uses `rglob("*.py")` to discover scripts, executing every `.py` file found in render output. A planted script in the render directory gets executed. | `src/execute/processor.py:1028` (`scripts = list(render_output_dir.rglob(pattern))`); `src/execute/executor.py:1069` (`subprocess.run([sys.executable, script])`) | 3 (plant `.py` file in render output dir) | 4 (code execution) | **12 HIGH** | Remediable (allowlist expected script names; verify render manifest) |
| V-11 | Technical | **No rate limiting on API endpoints by default**: FastAPI has no rate limiting middleware. Job submission is unbounded. | `src/api/server.py` (no rate limiting middleware); `src/api/app.py` (none) | 4 (trivial — just send requests) | 2 (DoS via resource exhaustion; unbounded subprocess spawning) | **8 MEDIUM** | Remediable (add slowapi or similar) |
| V-12 | Reputational | **`nosec` annotations suppress bandit warnings**: Multiple `# nosec B403`, `# nosec B603`, `# nosec B301` annotations suppress security linter findings for subprocess and pickle usage. These are legitimate suppressions *if* the input is trusted, but the pipeline accepts untrusted GNN files. | `src/execute/executor.py:10` (`import subprocess  # nosec B404`); `src/gnn/parsers/binary_parser.py:46` (`# nosec B301`); `src/execute/pymdp/pymdp_utils.py:7` (`import pickle  # nosec B403`) | N/A (governance finding) | 2 (false sense of security — linter is silenced) | **4 MEDIUM** | Remediable (review each nosec; document trust assumptions) |

---

## 4. Adversarial Narrative (Compounding Chain)

**The strongest attack chain routes through V-06 → V-01 → V-04:**

1. **V-06 (Inherent)**: The pipeline executes rendered code at Step 12 *before* security
   scanning at Step 18. This is inherent to the design — the security step is forensic, not
   preventive.

2. **V-04 (Critical)**: An attacker reaches the unauthenticated FastAPI endpoint (if
   deployed with `--host 0.0.0.0` as documented in `server.py:11`). No API key, no auth
   middleware. They submit a `POST /api/v1/process` with `target_dir` pointing to a
   directory containing a crafted GNN `.md` file.

3. **V-01 (Critical)**: The GNN file is parsed (Step 3), rendered into Python (Step 11),
   and executed (Step 12) via `subprocess.run([sys.executable, script])` at
   `executor.py:1069`. The crafted GNN content renders into a Python script containing
   arbitrary code — `os.system`, `subprocess`, `socket` — which executes with the
   pipeline's full privileges.

4. **V-02 (Critical, alternate entry)**: If the attacker can supply a `.pickle` or `.pkl`
   file (via the input directory or API), `binary_parser.py:46` calls `pickle.load`
   directly, achieving RCE without even needing the render→execute chain.

**Secondary chain**: V-05 → V-10 — An attacker creates a symlink inside the output
directory (via API `output_dir` parameter with `create=True`) pointing outside the repo.
Combined with V-10's unbounded `rglob("*.py")`, a planted script in a symlinked directory
gets discovered and executed.

---

## 5. Mitigations

| Finding | Mitigation | Adversary Re-test |
|---------|------------|-------------------|
| **V-01/V-06** | Insert a security gate between Step 11 (Render) and Step 12 (Execute): scan rendered `.py` files via `_check_python_ast` and block execution if high-severity findings are present. This converts Step 18 from forensic to preventive. | Adversary cannot bypass without first defeating the AST scanner — which checks the actual code that will run, not the source GNN. |
| **V-02** | Replace `pickle.load` with a safe deserializer (e.g., `json.load` for JSON files, or `pickletools.disassemble` + validate before `pickle.load`). Alternatively, require a signed manifest for pickle inputs. | Adversary's crafted pickle is rejected at the deserialization boundary. |
| **V-04** | Add API key authentication (e.g., `X-API-Key` header) to all FastAPI endpoints. Change `server.py:11` docstring to use `--host 127.0.0.1`. Add startup warning if bound to non-loopback. | Adversary without the API key cannot submit jobs. |
| **V-05** | After `resolve()`, check `os.path.islink()` on the original path and reject symlinks. Or use `Path.stat()` to verify the resolved path is within the repo root *without* following symlinks. | Adversary cannot create an escaping symlink. |
| **V-07** | Add `X-Forwarded-For` / `X-Real-IP` header rejection when `GNN_MCP_ALLOW_INSECURE_LOCAL` is set. Document the risk in startup logs. | SSRF via proxy headers is detected and rejected. |
| **V-10** | Instead of `rglob("*.py")`, maintain a manifest of expected rendered scripts (written by Step 11) and only execute scripts listed in the manifest. | Planted scripts not in the manifest are not executed. |
| **V-03** | Add depth/length limits before `ast.literal_eval` calls (e.g., reject strings > 10KB or nesting > 10 levels). | DoS payloads are rejected before evaluation. |

---

## 6. Go/No-Go Recommendation

### **GO-WITH-CONDITIONS**

The GNN pipeline is safe for **trusted, local, single-user research use** where:
- Input GNN files are authored by or trusted by the operator
- The API/MCP servers are not exposed to networks (bind to `127.0.0.1` only)
- The `input/` directory is not writable by untrusted parties

**Preconditions for broader use** (all must be met):
1. **V-01/V-06**: Implement pre-execution security scanning of rendered code (security gate between Steps 11 and 12)
2. **V-02**: Replace or sandbox `pickle.load` on untrusted input
3. **V-04**: Add authentication to API endpoints before any network deployment
4. **V-05**: Harden path validation against symlink traversal

**Residual risk**: The pipeline's core design — rendering untrusted text specifications
into executable code and running them — is inherently risky. Even with pre-execution
scanning, an attacker who can influence the GNN input can attempt to craft content that
passes the AST scanner but still achieves code execution through obfuscation or framework-
specific exploits (e.g., JAX/pymdp APIs). The strongest mitigation is sandboxing
(`subprocess.run` inside a container/namespace with no network and restricted filesystem).

**Confidence**: Medium — all findings are tied to specific file:line evidence. The
compounding chain (V-06→V-01→V-04) is stable across evidence sources. The weakest link is
V-05 (symlink bypass), which depends on the attacker having write access to the output
directory — this may not be exploitable in all deployment configurations. Alternative
adversary model: a malicious insider with filesystem access would bypass V-04 (no auth
needed) but still benefit from V-01 and V-02.
