# Specification: .Agent Rules Ecosystem

## Design Requirements
The `.agent_rules` module operates outside the standard 25-step execution pipeline. Rather than mapping structural runtime logic, it establishes the **behavioral boundaries** and **compliance frameworks** for automated systems editing the repository.

## Components
While it exports no executable classes, it provides static guarantees:
1. All AI actions must check the Zero-Mock policy defined in `testing.md`.
2. Pipeline architectural changes must conform to the Thin Orchestrator pattern outlined in `architecture.md`.
3. Error handling must strictly follow the Safe-to-Fail principles in `error_handling.md`.

## Interfaces
These specifications are consumed natively during context injection phases by the connected Large Language Models (LLMs).
