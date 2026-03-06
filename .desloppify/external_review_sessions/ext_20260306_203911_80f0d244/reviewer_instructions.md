# External Blind Review Session

Session id: ext_20260306_203911_80f0d244
Session token: 7f1f04d3ceba0fc7de1aa95cc0f846e9
Blind packet: /Users/4d/Documents/GitHub/generalizednotationnotation/.desloppify/review_packet_blind.json
Template output: /Users/4d/Documents/GitHub/generalizednotationnotation/.desloppify/external_review_sessions/ext_20260306_203911_80f0d244/review_result.template.json
Claude launch prompt: /Users/4d/Documents/GitHub/generalizednotationnotation/.desloppify/external_review_sessions/ext_20260306_203911_80f0d244/claude_launch_prompt.md
Expected reviewer output: /Users/4d/Documents/GitHub/generalizednotationnotation/.desloppify/external_review_sessions/ext_20260306_203911_80f0d244/review_result.json

Happy path:
1. Open the Claude launch prompt file and paste it into a context-isolated subagent task.
2. Reviewer writes JSON output to the expected reviewer output path.
3. Submit with the printed --external-submit command.

Reviewer output requirements:
1. Return JSON with top-level keys: session, assessments, findings.
2. session.id must be `ext_20260306_203911_80f0d244`.
3. session.token must be `7f1f04d3ceba0fc7de1aa95cc0f846e9`.
4. Include findings with required schema fields (dimension/identifier/summary/related_files/evidence/suggestion/confidence).
5. Use the blind packet only (no score targets or prior context).
