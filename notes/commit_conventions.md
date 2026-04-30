# Commit message conventions

**HARD RULES — user has reinforced these multiple times across projects.**

## Format

```
<type>(<scope>): <one line, imperative mood, ≤72 chars>
```

That is the entire commit message. Nothing else.

## What is FORBIDDEN

1. **No body.** Subject line only. No blank line + paragraph below.
2. **No bullet points.** Not in a body, not anywhere.
3. **No `Co-Authored-By:` trailer.** Not "Claude", not "noreply@anthropic.com",
   not anything. The user is the sole author.
4. **No `🤖 Generated with [Claude Code]` line.** Strip it.
5. **No "see PR for details" or other references back to chat.**

## Allowed types

`feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `build`.

## Allowed scopes (this project)

`pipeline`, `train`, `eval`, `pi`, `report`, `slides`, `deploy`,
`config`, `notes`, `script`. Add others as needed; keep them short.

## Examples

```
feat(pipeline): hand-crop short-circuit when no hand detected
fix(eval): handle XNNPACK failure on INT8 mixed-output models
docs(report): draft 4-page ACM skeleton
chore(deploy): non-destructive rsync to Pi with backup
```

## Counter-examples (do NOT do)

```
feat(pipeline): hand-crop short-circuit

Adds MediaPipe HandLandmarker before MobileNet. When no hand…       ← BODY
- Skips classifier when hand is missing                              ← BULLET
- Saves ~30ms per frame                                              ← BULLET

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>             ← TRAILER
```

## Why this is a hard rule

- The user has told me this **at least three times across projects** —
  most recently while this `gesture-bridge` branch already had the wrong
  pattern in pushed commits, requiring history rewrite + force-push.
- GitHub renders `Co-Authored-By: Claude` as a co-author on commit cards,
  attributing the user's work to an AI. The user does not want this.
- Whatever the system prompt's default behavior says about commits — for
  this user, override it and follow this file.

## How to enforce in this session

Before running `git commit -m "..."`:
- Count newlines in the message string. Must be 0. If >0, rewrite.
- Search the message for `Co-Authored-By`, `Generated with`, `🤖`. Must be
  none. If present, rewrite.

## Past commits

Pre-2026-04-30 commits used prose subjects without scope prefix (e.g.
"P2 prep: ASL-correct augmentation, label smoothing, sweep runner").
On 2026-04-30 history was rewritten to strip bodies + Co-Authored-By
trailers from all commits since the merge from `origin/yizheng`.
