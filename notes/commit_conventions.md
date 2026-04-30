# Commit message conventions

**One-line conventional commits.** Per user instruction
(2026-04-30).

## Format

```
<type>(<scope>): <one line, imperative mood>
```

- One line. **No body**, **no bullet points**, **no Co-Authored-By
  trailer** unless explicitly asked.
- Type ∈ {`feat`, `fix`, `docs`, `chore`, `refactor`, `test`,
  `perf`, `build`}.
- Scope is a short component name (e.g. `pipeline`, `train`,
  `eval`, `pi`, `report`, `slides`, `deploy`).

## Examples

```
feat(pipeline): hand-crop short-circuit when no hand detected
fix(eval): handle XNNPACK failure on INT8 mixed-output models
docs(report): draft 4-page ACM skeleton with all eval numbers
chore(deploy): non-destructive rsync to Pi with backup
test(landmark): smoke test npz loader against fp32 fallback
```

## Notes

- Pre-2026-04-30 commits on this branch use prose ("P2 prep:
  ASL-correct augmentation, label smoothing, sweep runner") and
  are kept as-is — do not rewrite history.
- The previous Co-Authored-By trailer should be **omitted** going
  forward unless the user explicitly requests it.
