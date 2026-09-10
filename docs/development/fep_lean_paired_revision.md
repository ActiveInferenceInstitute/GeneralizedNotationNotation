# fep_lean paired-revision protocol (GNN-04)

The GNN side hosts paired-revision CI for the fep_lean bridge pair, mirroring
the mechanism GEO-INFER already runs for this repository: its
`.github/workflows/gnn-interchange.yml` workflow with pin file
`.github/gnn-pair.json`.

## What is pinned

`.github/fep-lean-pair.json` records the reviewed companion revision:

```json
{
  "repository": "ActiveInferenceInstitute/fep_lean",
  "revision": "3f3100edf358d1eec5c8c56b4750c4c122cb3c4c"
}
```

The revision is a full 40-hex SHA of the fep_lean main HEAD that the GNN-side
contract was checked against. The current pin (`3f3100e`) is the fep_lean main
commit "docs: close the residual check-list parity gaps the review pool caught".

## What runs

`.github/workflows/fep-lean-paired-revision.yml`:

1. Reads and validates the pin (known repository slug, 40-hex revision).
2. Checks out fep_lean at exactly the pinned SHA.
3. Records both checkout revisions plus the pin file as receipts, retained as
   workflow artifacts (`if: always()`, mirroring GEO's receipt semantics).
4. Runs fep_lean's **read-only** bridge surface against this GNN checkout:

   - `uv run fep-lean bridge status --gnn-root <gnn checkout>`
   - `uv run fep-lean bridge emit --check --model finite --gnn-root <gnn>`
   - `uv run fep-lean bridge emit --check --model continuous --gnn-root <gnn>`

   `status` and `emit --check` never write: they compare source bytes and
   emitted-document digests only.

Strictness mirrors GEO: the job is blocking. A red run is the drift signal —
the pair must be re-pinned together, not the check suppressed.

## Who bumps the pin

Pin bumps are paired with fep_lean-side re-pins:

- **GNN side**: changing GNN sources that the bridge's source binding covers
  requires a paired commit on fep_lean (per its FEP-EVIDENCE-CURRENT policy)
  that re-records digests against the new GNN state; the GNN-side PR then
  bumps `.github/fep-lean-pair.json` to the new fep_lean revision in the same
  change set.
- **fep_lean side**: when fep_lean main moves, this repository's pin is
  updated to the new reviewed fep_lean SHA together with any GNN-side source
  changes the new pin requires.

A green run therefore certifies agreement between two specific, named
revisions — never a floating main.

## What the workflow proves and does NOT prove

The workflow is a **drift-detection** surface. It proves that the pinned
fep_lean checkout and this GNN checkout still agree on the bridge's source
binding (owner rosters and content digests), finite/continuous freshness, the
syntax surface, the contract mirror, and the formal projection with no drift.

It does **not** prove any Lean theorem, and no notation mapping claim may
promote it into one: the bridge compares symbols and constructs across the two
repositories; correspondence of notation is a documentation-level statement.
Evidence planes stay distinct — a green paired run says "the two checkouts are
in the state fep_lean reviewed", nothing about the truth of formal claims.

## Note on the current pin

The pin at `3f3100e` predates fep_lean's docs restructure; the bridge surface
reads source bytes, not docs, so a docs-only fep_lean change does not affect
it. If both checkouts have genuinely drifted since the pin, the job reports
`source_binding` failures listing the changed owner files — that output is the
paired re-pin worklist.

## Related

- [GEO-INFER paired interchange (GNN-side docs)](geo_infer_2026_09.md)
- fep_lean bridge contract (GNN-side docs): `docs/other/fep_lean/bridge-contract.md`
