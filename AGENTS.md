# Cubesim Agent Brief

## Scope
- `cubesim` is a reusable ETC framework intended for public release.
- Prefer stable contracts and incremental changes over broad rewrites.
- Preserve science-visible behavior unless a change is explicitly intended and documented.

## Architecture
- Keep clear boundaries between:
  - instrument config loading
  - target/source model construction
  - background, throughput, QE, detector, and noise models
  - ETC computation and result assembly
  - reusable analysis helpers
- Treat the package root as a deliberate public API boundary.
- Re-export only supported user-facing entrypoints there.
- Keep project-specific behavior out of `cubesim`.

## Contracts
- Treat instrument configs, artifact references, and ETC inputs as explicit contracts.
- Validate early with actionable errors.
- Avoid silent coercions and hidden fallback behavior.
- Keep one obvious owner per contract.
- Define stable keys and field names as named constants in the narrowest module that owns the contract.

## Code Style
- Optimize for readability, homogeneity, and symmetry in naming and ordering.
- Prefer explicit ownership over convenience abstractions.
- Remove stale indirection rather than preserving weak abstraction layers.
- Keep helpers in the narrowest module that owns the behavior.
- Keep comments concise and technical.
- Update docstrings when behavior or ownership changes.
- Prefer explicit names that match actual behavior.
- Inline helpers that do too little to justify abstraction.
- Collapse duplicated validators or wrappers when they enforce the same invariant.
- Push back before implementing changes that reduce clarity or consistency.

## Module Organization
- Prefer a consistent module order:
  - constants first
  - data structures next (`dataclass`, types, enums)
  - properties or simple accessors near the top
  - helper primitives next
  - composed helpers next
  - public entrypoints last
- If a module follows a strong lifecycle, order methods by lifecycle instead.
- Separate logical blocks with clear section comments.

## Public API
- Keep public APIs clean, explicit, and minimally surprising.
- CLI or scripts should remain thin wrappers over the Python API.
- Do not expand the public API casually.

## Testing Expectations
- Add or adjust tests with every behavior change.
- Prioritize numerical regression protection for ETC behavior.
- Prefer tests of externally visible behavior over tests coupled to internal structure.
- When refactoring, confirm representative ETC workflows still match within defined tolerances.

## Quality Bar
- Keep docs and examples aligned with code.
- If an abstraction does not clearly improve the reusable package boundary, do not add it.
- Clean up stale comments, duplicated contract checks, and outdated docstrings in touched code.
