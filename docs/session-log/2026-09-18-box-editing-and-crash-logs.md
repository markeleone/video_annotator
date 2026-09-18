# 2026-09-18 — Box visibility, box editing, crash diagnostics

**Trigger:** fish-ID student reported (1) still getting crashes, (2) boxes
"sometimes genuinely not there" when going back to check, (3) hard to change
details of an already-placed box.

## Findings
1. **Crashes** — a 30-action random stress test (`tests/fuzz_annotator.py`,
   4 seeds × 3000 steps) found **no Python exceptions** in the 2026-09-08
   code. Likely causes: student still on a pre-09-08 zip (the box-placement
   NameError), or a native crash. There was no log to tell. Also found two
   "looks like a crash" paths: a startup open failure silently quitting
   (`sys.exit(msg)` in a windowed app), and full-disk frame extraction
   silently dropping frames (temp frames are GBs/clip and leaked on crash).
2. **Vanishing boxes** — boxes live on exactly one frame; one frame off and
   they aren't drawn. Delete/Backspace could also remove a selected box that
   wasn't on screen.
3. **Editing** — required select → edit → *Update*; the list rebuild after an
   update cleared the selection, so further edits silently did nothing.

## Changes (all in `run_annotator.py`)
- Dashed "nearby" boxes (±1 s) with offset label; click → jump + select.
  List row click also jumps to the box's frame.
- Edit mode: yellow banner, live apply (`_live_edit`), single undo per
  session, Esc / *Done editing* exits, next-box settings stashed/restored.
  New boxes are not auto-selected. Selection survives list rebuilds.
- Delete key only for a box visible on the current frame; undo hint.
- `APP_VERSION = "2026.09.18"`; `~/CTAG_Annotator/logs/errors.log` and
  `hard_crash.log` (faulthandler) with a next-launch notice; startup-open
  failure dialog; imwrite failure → clear error; stale temp-frame sweep via
  `.owner_pid`; preload dict iteration race fixed.

## Tests
`tests/test_annotator.py` 33/33 pass (11 new). Fuzz: 0 errors.

## Next
- Ask the student which zip they have (Help now shows the version) and to
  send `~/CTAG_Annotator/logs/` if anything crashes again.
- If they want a box to follow an animal across frames, that's a data-model
  change (multi-frame bboxes / interpolation) — discuss before building.
