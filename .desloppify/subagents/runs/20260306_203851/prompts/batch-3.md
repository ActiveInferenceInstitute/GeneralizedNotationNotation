You are a focused subagent reviewer for a single holistic investigation batch.

Repository root: /Users/4d/Documents/GitHub/generalizednotationnotation
Blind packet: /Users/4d/Documents/GitHub/generalizednotationnotation/.desloppify/review_packet_blind.json
Batch index: 3
Batch name: Abstractions & Dependencies
Batch dimensions: abstraction_fitness, dependency_health, mid_level_elegance, low_level_elegance
Batch rationale: abstraction hotspots (wrappers/interfaces/param bags), dep cycles

Files assigned:
- src/gnn/parsers/common.py
- doc/cognitive_phenomena/meta-awareness/utils.py
- src/tests/infrastructure/utils.py
- src/gui/oxdraw/utils.py
- src/setup/utils.py
- src/ontology/utils.py
- src/audio/sapf/utils.py
- src/export/utils.py
- src/gnn/parsers/utils.py
- src/template/utils.py
- src/mcp/meta_mcp.py
- src/utils/mcp.py
- src/pipeline/mcp.py
- src/gnn/testing/test_round_trip.py
- src/render/pymdp/pymdp_renderer.py
- src/execute/pymdp/pymdp_simulation.py
- src/gnn/__init__.py
- src/gui/gui_2/ui_simple.py
- src/render/pymdp_template.py
- doc/cognitive_phenomena/meta-aware-2/test_complete_implementation.py
- src/llm/providers/openai_provider.py
- output/11_render_output/actinf_pomdp_agent/pymdp/Active Inference POMDP Agent_pymdp.py
- output/11_render_output/deep_planning_horizon/pymdp/Deep Planning Horizon POMDP_pymdp.py
- output/11_render_output/hmm_baseline/pymdp/Hidden Markov Model Baseline_pymdp.py
- output/11_render_output/markov_chain/pymdp/Simple Markov Chain_pymdp.py
- output/11_render_output/multi_armed_bandit/pymdp/Multi Armed Bandit Agent_pymdp.py
- output/11_render_output/simple_mdp/pymdp/Simple MDP Agent_pymdp.py
- output/11_render_output/tmaze_epistemic/pymdp/T-Maze Epistemic Foraging Agent_pymdp.py
- output/11_render_output/two_state_bistable/pymdp/Two State Bistable POMDP_pymdp.py
- doc/cognitive_phenomena/meta-aware-2/core/meta_awareness_model.py
- doc/axiom/axiom_implementation/axiom.py
- src/gnn/schema_validator.py
- src/mcp/mcp.py
- src/gnn/parsers/schema_parser.py
- doc/cognitive_phenomena/meta-aware-2/config/gnn_parser.py
- src/tests/test_pipeline_scripts.py
- src/type_checker/estimation_strategies.py
- src/gui/gui_3/ui_designer.py
- src/tests/runner.py
- src/utils/pipeline_monitor.py
- src/render/processor.py
- src/gnn/processors.py
- doc/axiom/axiom_implementation/utils/visualization_utils.py
- doc/pymdp/pymdp_pomdp/pymdp_gridworld_visualizer.py
- src/advanced_visualization/network_viz.py
- src/render/rxinfer/toml_generator.py
- doc/cognitive_phenomena/meta-aware-2/execution/simulation_runner.py
- src/intelligent_analysis/processor.py
- src/render/numpyro/numpyro_renderer.py
- src/render/pytorch/pytorch_renderer.py
- src/utils/error_recovery.py
- src/pipeline/context.py
- src/utils/test_utils.py
- src/render/discopy/translator.py

Task requirements:
1. Read the blind packet and follow `system_prompt` constraints exactly.
1a. If previously flagged issues are listed above, use them as context for your review.
    Verify whether each still applies to the current code. Do not re-report fixed or
    wontfix issues. Use them as starting points to look deeper — inspect adjacent code
    and related modules for defects the prior review may have missed.
1c. Think structurally: when you spot multiple individual issues that share a common
    root cause (missing abstraction, duplicated pattern, inconsistent convention),
    explain the deeper structural issue in the finding, not just the surface symptom.
    If the pattern is significant enough, report the structural issue as its own finding
    with appropriate fix_scope ('multi_file_refactor' or 'architectural_change') and
    use `root_cause_cluster` to connect related symptom findings together.
2. Evaluate ONLY listed files and ONLY listed dimensions for this batch.
3. Return 0-10 high-quality findings for this batch (empty array allowed).
3a. Do not suppress real defects to keep scores high; report every material issue you can support with evidence.
3b. Do not default to 100. Reserve 100 for genuinely exemplary evidence in this batch.
4. Score/finding consistency is required: broader or more severe findings MUST lower dimension scores.
4a. Any dimension scored below 85.0 MUST include explicit feedback: add at least one finding with the same `dimension` and a non-empty actionable `suggestion`.
5. Every finding must include `related_files` with at least 2 files when possible.
6. Every finding must include `dimension`, `identifier`, `summary`, `evidence`, `suggestion`, and `confidence`.
7. Every finding must include `impact_scope` and `fix_scope`.
8. Every scored dimension MUST include dimension_notes with concrete evidence.
9. If a dimension score is >85.0, include `issues_preventing_higher_score` in dimension_notes.
10. Use exactly one decimal place for every assessment and abstraction sub-axis score.
11. Ignore prior chat context and any target-threshold assumptions.
12. Do not edit repository files.
13. Return ONLY valid JSON, no markdown fences.

Scope enums:
- impact_scope: "local" | "module" | "subsystem" | "codebase"
- fix_scope: "single_edit" | "multi_file_refactor" | "architectural_change"

Output schema:
{
  "batch": "Abstractions & Dependencies",
  "batch_index": 3,
  "assessments": {"<dimension>": <0-100 with one decimal place>},
  "dimension_notes": {
    "<dimension>": {
      "evidence": ["specific code observations"],
      "impact_scope": "local|module|subsystem|codebase",
      "fix_scope": "single_edit|multi_file_refactor|architectural_change",
      "confidence": "high|medium|low",
      "issues_preventing_higher_score": "required when score >85.0",
      "sub_axes": {"abstraction_leverage": 0-100 with one decimal place, "indirection_cost": 0-100 with one decimal place, "interface_honesty": 0-100 with one decimal place}  // required for abstraction_fitness when evidence supports it
    }
  },
  "findings": [{
    "dimension": "<dimension>",
    "identifier": "short_id",
    "summary": "one-line defect summary",
    "related_files": ["relative/path.py"],
    "evidence": ["specific code observation"],
    "suggestion": "concrete fix recommendation",
    "confidence": "high|medium|low",
    "impact_scope": "local|module|subsystem|codebase",
    "fix_scope": "single_edit|multi_file_refactor|architectural_change",
    "root_cause_cluster": "optional_cluster_name_when_supported_by_history"
  }],
  "retrospective": {
    "root_causes": ["optional: concise root-cause hypotheses"],
    "likely_symptoms": ["optional: identifiers that look symptom-level"],
    "possible_false_positives": ["optional: prior concept keys likely mis-scoped"]
  }
}
