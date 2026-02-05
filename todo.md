# Microatomics Project Plan

## Vision
Build a pure-geometry runtime that turns SVG paths and π-based transformations into a consistent, testable inference engine with a minimal REST API and a lightweight dashboard.

## Guiding Principles
- **Pure math first**: Keep dependencies minimal and avoid ML frameworks.
- **Deterministic geometry**: Favor repeatable operations and clearly defined invariants.
- **Clear APIs**: Make the geometric runtime usable from both Python and HTTP.
- **Observable outputs**: Every operation should be inspectable and visualizable (e.g., SVG output).

## Roadmap

### Phase 1: Foundation (Now)
- [ ] **Establish repository structure**
  - [ ] Add top-level package layout (`geometric/`, `server/`, `public/`, `tests/`).
  - [ ] Add `pyproject.toml` and dependency pinning.
- [ ] **Define core geometry primitives**
  - [ ] `PiTensor`: base tensor type with π-scale metadata.
  - [ ] `SvgTensor`: path-backed tensor representation.
  - [ ] `Plane`: relationship model for clusters.
- [ ] **Write invariants + reference math**
  - [ ] Document rotation, scaling, and shear rules.
  - [ ] Define input/output shapes for operations.
- [ ] **Bootstrap tests**
  - [ ] Unit tests for tensor creation and transformations.
  - [ ] Property tests for invariants (e.g., round-trip SVG conversions).

### Phase 2: Engine + API (Next)
- [ ] **Geometric operation engine**
  - [ ] Implement glyph operations `(⤍)(⤎)(↻)(↔)(⟲)(⟿)(⤂)(⤦)`.
  - [ ] Add consistent error handling and validation.
- [ ] **Cluster system**
  - [ ] Build cluster formation and plane relationship metrics.
  - [ ] Implement similarity scoring and inference outputs.
- [ ] **REST API + SDK**
  - [ ] Implement REST endpoints for compute/cluster/model.
  - [ ] Provide a small Python SDK for local use.
- [ ] **Docs + examples**
  - [ ] Add usage examples for each endpoint.
  - [ ] Provide a minimal tutorial in README.

### Phase 3: Visualization + UX (Later)
- [ ] **Dashboard**
  - [ ] Interactive SVG cluster explorer.
  - [ ] Operation playground with sample inputs.
- [ ] **Export formats**
  - [ ] SVG/JSON export for clusters and inference.
  - [ ] CLI tool for batch export.
- [ ] **Performance tuning**
  - [ ] Benchmarks for tensor ops and clustering.
  - [ ] Optional C-accelerated math routines.

## Immediate Next Steps (Suggested)
1. Define the `PiTensor` and `SvgTensor` data structures and add tests.
2. Implement the `(↻)` rotation operation with SVG output.
3. Add a `/api/geometric/compute` endpoint that returns SVG + metadata.
4. Create a small demo script showing tensor creation → transform → visualization.

## Questions to Resolve
- What exact SVG path format constraints do we support?
- How should inference outputs be expressed (geometry-only vs. metrics + geometry)?
- What is the minimal viable cluster similarity metric?
- Do we want deterministic seeds for randomized geometric operations?
