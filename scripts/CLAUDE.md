# === COGNILAYER (auto-generated, do not delete) ===

## CogniLayer v3 Active
Persistent memory is ON.
ON FIRST USER MESSAGE in this session, briefly tell the user:
  'CogniLayer v3 active — persistent memory is on. Type /cognihelp for available commands.'
Say it ONCE, keep it short, then continue with their request.

## Memory Tools
You have access to the `cognilayer` MCP server:
- memory_search(query) — search memory semantically
- memory_write(content) — save important information
- file_search(query) — search project files (PRD, docs...)
- decision_log(query) — find past decisions

When unsure about context or project history,
ALWAYS search memory first via memory_search.
When you need info from PRD or docs, use file_search
INSTEAD of reading the entire file.

## VERIFY-BEFORE-ACT — MANDATORY
When memory_search returns a fact marked with ⚠ STALE:
1. ALWAYS read the source file and verify the fact still holds
2. If the fact changed -> update it via memory_write
3. NEVER make changes based on STALE facts without verification

## PROACTIVE MEMORY — IMPORTANT
When you discover something important during work, SAVE IT IMMEDIATELY:
- Bug and fix -> memory_write(type="error_fix")
- Pitfall/danger -> memory_write(type="gotcha")
- Exact procedure -> memory_write(type="procedure")
- How components communicate -> memory_write(type="api_contract")
- Performance issue -> memory_write(type="performance")
- Important command -> memory_write(type="command")
DO NOT wait for /harvest — session may crash.

## RUNNING BRIDGE — CRITICAL
After completing each task AUTOMATICALLY update session bridge:
  session_bridge(action="save", content="Progress: ...; Open: ...")
This is Tier 1 — do it yourself, don't announce, it's part of the job.

## Safety Rules — MANDATORY
- Before ANY deploy, push, ssh, pm2, docker, db migration:
  1. ALWAYS call verify_identity(action_type="...") first
  2. If it returns BLOCKED — STOP and ask the user
  3. If it returns VERIFIED — READ the target server to the user and request confirmation

## Git Rules
- Commit often, small atomic changes. Format: "[type] what and why"
- commit = Tier 1 (do it yourself). push = Tier 3 (verify_identity).

## Project DNA: scripts
Stack: unknown
Style: [unknown]
Structure: ?
Deploy: [NOT SET]
Active: [new session]
Last: [first session]

## Last Session Bridge
[Emergency bridge — running bridge was not updated]
No changes or facts in this session.

# === END COGNILAYER ===
