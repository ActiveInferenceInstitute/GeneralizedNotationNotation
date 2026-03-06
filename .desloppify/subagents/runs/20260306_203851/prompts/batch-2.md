You are a focused subagent reviewer for a single holistic investigation batch.

Repository root: /Users/4d/Documents/GitHub/generalizednotationnotation
Blind packet: /Users/4d/Documents/GitHub/generalizednotationnotation/.desloppify/review_packet_blind.json
Batch index: 2
Batch name: Conventions & Errors
Batch dimensions: convention_outlier, error_consistency, mid_level_elegance
Batch rationale: naming drift, behavioral outliers, mixed error strategies

Files assigned:
- doc/activeinference_jl/test_activeinference_renderer.py
- doc/axiom/axiom_implementation/modules/__init__.py
- doc/cognitive_phenomena/meta-aware-2/config/__init__.py
- doc/cognitive_phenomena/meta-aware-2/core/__init__.py
- doc/cognitive_phenomena/meta-aware-2/execution/__init__.py
- doc/cognitive_phenomena/meta-aware-2/run_meta_awareness.py
- doc/cognitive_phenomena/meta-aware-2/simulation_logging/__init__.py
- doc/cognitive_phenomena/meta-aware-2/tests/__init__.py
- doc/cognitive_phenomena/meta-aware-2/visualization/__init__.py
- doc/cognitive_phenomena/meta-awareness/__init__.py
- doc/kit/gnn_kit/kit_setup.py
- doc/petri_nets/__init__.py
- doc/pkl/pkl_gnn_demo.py
- src/advanced_visualization/__init__.py
- src/advanced_visualization/html_generator.py
- src/analysis/__init__.py
- src/analysis/activeinference_jl/__init__.py
- src/analysis/discopy/__init__.py
- src/analysis/jax/__init__.py
- src/analysis/math_utils.py
- src/analysis/numpyro/__init__.py
- src/analysis/pymdp/__init__.py
- src/analysis/pytorch/__init__.py
- src/analysis/rxinfer/__init__.py
- src/analysis/trace_analysis.py
- src/api/models.py
- src/audio/__init__.py
- src/audio/generator.py
- src/audio/sapf/__init__.py
- src/audio/sapf/utils.py
- src/execute/activeinference_jl/__init__.py
- src/execute/discopy/__init__.py
- src/execute/discopy_translator_module/__init__.py
- src/execute/fallback.py
- src/execute/install_dependencies.py
- src/execute/jax/__init__.py
- src/execute/numpyro/__init__.py
- src/execute/pymdp/package_detector.py
- src/execute/pymdp/validator.py
- src/execute/pytorch/__init__.py
- src/execute/rxinfer/__init__.py
- src/export/__init__.py
- src/export/utils.py
- src/gnn/__init__.py
- src/gnn/contracts.py
- src/gnn/cross_format.py
- src/gnn/dep_graph.py
- src/gnn/documentation/__init__.py
- src/gnn/formal_specs/__init__.py
- src/gnn/frontmatter.py
- src/gnn/grammars/__init__.py
- src/gnn/multimodel.py
- src/gnn/parsers/__init__.py
- src/gnn/parsers/alloy_serializer.py
- src/gnn/parsers/asn1_serializer.py
- src/gnn/parsers/base_serializer.py
- src/gnn/parsers/binary_parser.py
- src/gnn/parsers/binary_serializer.py
- src/gnn/parsers/common.py
- src/gnn/parsers/converters.py
- src/gnn/parsers/coq_parser.py
- src/gnn/parsers/coq_serializer.py
- src/gnn/parsers/functional_parser.py
- src/gnn/parsers/functional_serializer.py
- src/gnn/parsers/grammar_parser.py
- src/gnn/parsers/grammar_serializer.py
- src/gnn/parsers/isabelle_parser.py
- src/gnn/parsers/isabelle_serializer.py
- src/gnn/parsers/json_parser.py
- src/gnn/parsers/json_serializer.py
- src/gnn/parsers/lean_parser.py
- src/gnn/parsers/lean_serializer.py
- src/gnn/parsers/markdown_serializer.py
- src/gnn/parsers/maxima_parser.py
- src/gnn/parsers/maxima_serializer.py
- src/gnn/parsers/pkl_serializer.py
- src/gnn/parsers/protobuf_serializer.py
- src/gnn/parsers/python_parser.py
- src/gnn/parsers/python_serializer.py
- src/gnn/parsers/scala_serializer.py

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
  "batch": "Conventions & Errors",
  "batch_index": 2,
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
