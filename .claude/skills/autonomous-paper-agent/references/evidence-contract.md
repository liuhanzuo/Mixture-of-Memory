# Evidence Contract

## Claim Statuses

- `verified`: directly supported by registered evidence or a verified citation.
- `partially_supported`: evidence supports a narrower or qualified version.
- `planned`: intended but not yet executed or verified.
- `unsupported`: no adequate evidence exists.
- `rejected`: contradicted or removed after analysis.

Only `verified` and carefully qualified `partially_supported` claims may appear as positive headline claims.

## Numeric Provenance

Every reported number should map to:

- evidence ID;
- source file or database location;
- run ID;
- exact extraction/aggregation procedure;
- unit and metric definition;
- uncertainty or seed information when relevant;
- manuscript locations.

## Method Provenance

Every material method statement should map to:

- source file and symbol;
- configuration key and value;
- executed code version or commit;
- runtime environment when behavior is version-dependent;
- manuscript locations.

## Citation Provenance

Every citation should map to:

- stable identifier when available;
- verified title, authors, year, and venue;
- sentence or claim it supports;
- support strength: direct, partial, background, or contrast;
- verification date and source.

## Reviewer Evidence Rules

A reviewer criticism is actionable only when it identifies:

- the affected claim or location;
- observed evidence or absence;
- why the issue changes a rubric dimension;
- a possible verification test.

Reviewer authority alone is not evidence. Unsupported criticisms should be preserved as opinions but not treated as verified defects.
