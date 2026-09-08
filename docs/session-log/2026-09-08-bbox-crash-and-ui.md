# 2026-09-08 — fish-box crash fix + UI round

## What was wrong

The app hard-crashed whenever a bounding box was placed. Root cause was a
one-line rename miss, not anything structural:

`_on_bbox` used to read `species = self.species_combo.currentText()`. When the
taxonomy picker replaced that combo, the local became `tax` (a dict) and the
`TrackedFeature(...)` call was updated — but two status-bar lines further down
still referenced `species`, raising `NameError`. Because that runs inside a Qt
slot, PyQt aborts the process, so it looked like a crash rather than an error.

Worse: `store.add_feature()` had already run, and the 2 s autosave debounce had
not yet fired, so the annotation died with the process.

## What changed

- `_on_bbox` now builds the status message from `feat.id_label()`.
- `_install_crash_guard()` added: any exception escaping a slot flushes work to
  disk and shows a dialog instead of aborting.
- Visibility back to 5 points ("Very bad" restored).
- "None" option on each label bar; erase-by-rescrub.
- Fresh videos start with blank labels (no accidental painting).
- Focus mode (`F`), settings persistence, "Fish labeler" rename.
- Family common names for all 55 families, searchable.

## Files touched

- `run_annotator.py` — all of the above.
- `tests/test_annotator.py` — **new**, 22 headless tests, all passing.
- `CLAUDE.md`, `NOTES.md` — **new**, project + running record.

## Numbers

- 22/22 tests pass.
- Focus mode: video pane 961×404 → 1268×707 at a 1280×800 window.
- Taxonomy: 301 rows, 103 genera all mapped, 55 families all named.

## Environment traps re-confirmed

- iCloud-synced Desktop breaks `codesign` **and** Qt plugin loading — build and
  stage plugins outside it.
- Qt rejects platform plugins copied with `shutil.copytree`; use `cp -R`.
- A stale plugin copy from an older PyQt6 fails identically to a missing one.

## Next

Fish reference photos; EN/ES toggle; Intel-Mac and Windows builds. Ask Mark
whether to open the PR to `jadenvc/video_annotator`.
