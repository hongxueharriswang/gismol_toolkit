# Pull Request

## Summary
Describe the change and why it’s needed.

## Type of change
- [ ] fix (bug fix)
- [ ] feat (new feature)
- [ ] perf (performance)
- [ ] refactor
- [ ] docs
- [ ] test
- [ ] chore (build/ci/tooling)

## Related issues
Closes #

## Details
- Implementation notes
- Design decisions, trade-offs

## Breaking changes
- [ ] No breaking changes
- [ ] Breaking changes (describe migration/mitigation):

## Module impact (check all that apply)
- [ ] `czoi.core`
- [ ] `czoi.permission` (effective permission calculus / gamma mappings)
- [ ] `czoi.constraint` (targets, priorities, `safe_eval`)
- [ ] `czoi.neural`
- [ ] `czoi.embedding`
- [ ] `czoi.daemon`
- [ ] `czoi.simulation`
- [ ] `czoi.integrations` (Django/Flask/FastAPI)
- [ ] `czoi.cli`
- [ ] `czoi.storage` (models/migrations)
- [ ] `czoi.utils`

## Security considerations
If this PR touches security-sensitive areas, summarize the threat model and add negative tests. See ../SECURITY.md.

## Performance considerations
Any expected performance/memory impact? Include benchmarks if relevant.

## Checklist
- [ ] Lint & format pass (`ruff`, `black`)
- [ ] Types pass (`mypy`)
- [ ] Tests added/updated
- [ ] Docs/examples updated
- [ ] DB migration included (if applicable)
- [ ] Changelog updated (if applicable)
- ../CODE_OF_CONDUCT.md and ../CONTRIBUTING.md