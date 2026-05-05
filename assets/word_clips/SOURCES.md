# Word clip sources (Phase 2)

These reference clips power the speech-to-sign visual output. They are
downloaded, **not committed** (license uncertain — academic /
educational use only). Re-fetch via `scripts/fetch_word_clips.sh` (the
original 5) or follow the steps below for the full set.

## Provenance

| Group | Glosses | Source |
|---|---|---|
| 5 base words (IT-1) | hello, help, no, yes, thank_you | aslbricks.org direct MP4 + signbsl.com (Start ASL clip via signasl.org) |
| 19 mined locally (IT-7) | chair, table, bed, shirt, orange, dance, work, finish, enjoy, wrong, many, family, mother, school, like, drink, dog, deaf, walk | Picked the largest clip per gloss from the WLASL-100 download already on disk under `data/wlasl100/videos/<gloss>/`. Per-clip source URLs recorded in `data/wlasl100/manifest.csv`. |
| 18 added via aslbricks (IT-7) | water, food, please, more, father, time, day, name, eat, play, good, bad, happy, sad, old, book, you, my | `http://aslbricks.org/New/ASL-Videos/<word>.mp4` — needs a browser User-Agent header to bypass 406. |
| 2 via signbsl mirror (IT-7) | love, friend | `https://media.signbsl.com/videos/asl/startasl/mp4/<word>.mp4` with referer `https://www.signasl.org/`. |

**Total: 44 word clips covering ~55 spoken-word tokens with aliases**
(e.g. `mom → mother.mp4`, `dad → father.mp4`, `done → finish.mp4`,
`mine → my.mp4`).

## License notes

- WLASL videos: C-UDA, computational use only. Do not redistribute the
  raw clips. Trained model weights and per-clip landmark npz arrays
  are derivatives we keep.
- aslbricks.org / signbsl.com / signasl.org: each site has its own
  terms. Use only for educational / academic demonstration. Do not
  redistribute.

## Reproduce

`scripts/fetch_word_clips.sh` covers the original 5. For the IT-7
expansion (39 more), the same approach was applied to a wider list:

1. Run `scripts/prepare_wlasl100.py` (downloads the local WLASL pool).
2. For glosses present locally:
   `cp $(ls -S data/wlasl100/videos/<gloss>/*.mp4 | head -1) assets/word_clips/<gloss>.mp4`
3. For glosses missing from the local pool, fetch via aslbricks /
   signbsl with a browser UA + Referer header (see above).
