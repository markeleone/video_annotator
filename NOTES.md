# Running notes

Newest entries at the bottom.

## 2026-09-08 — crash on placing a fish box + UI round

- **Fixed the crash.** Placing any bounding box raised
  `NameError: name 'species' is not defined` in `_on_bbox`. When the flat
  species combo was replaced by the taxonomy picker, the local `species` was
  renamed to `tax` (a dict) but two status-bar lines still used the old name.
  PyQt aborts the process when an exception escapes a slot, so it presented as
  a hard crash — and because it fired *after* `add_feature` but *before* the
  2 s autosave debounce, the box was lost too.
- Swept the file with pyflakes: no other undefined names. (First pyflakes run
  was a false negative — the module wasn't installed and the grep found
  nothing. Verify the tool exists before trusting a clean lint.)
- **Added `_install_crash_guard()`**: a `sys.excepthook` that flushes work to
  disk, then shows a dialog instead of letting Qt abort. Has a re-entrancy flag
  so repeated errors don't stack dialogs.
- Visibility: **"Very bad" restored** (5-point scale again) at Mark's request.
  `VISIBILITY_LEGACY_MAP` now only maps the old Poor/Moderate values.
- **"None" button on every label bar.** Sets that axis's pen to blank; select
  None and re-scrub a stretch to erase labels applied by accident.
- **Pen semantics tightened**: a fresh video starts with all four axes blank,
  so scrubbing before choosing a label records nothing. Seeking onto a labeled
  frame adopts it; seeking over blank frames keeps your current selection.
- **Focus mode** (button + `F`): hides the panels, bars and timeline. Redraw is
  deferred a tick so the video actually grows into the freed space
  (961×404 → 1268×707 at a 1280×800 window).
- **Settings persist** in `~/CTAG_Annotator/settings.json`: window size, focus
  mode, labeling on/off, per-axis checkboxes, speed, category.
- Control card renamed **"Controls" → "Fish labeler"**.
- **Family common names** (`FAMILY_COMMON`, all 55 families) shown in the Family
  dropdown and matched by the Find box — "grouper" now finds Serranidae.
- **Test suite moved into the repo** at `tests/test_annotator.py` (22 tests).
  The previous ad-hoc tests lived in `/tmp` and were deleted between sessions.
- Found while doing that: Qt won't load platform plugins copied by
  `shutil.copytree` (macOS provenance), and won't load them from the
  iCloud-synced Desktop at all. Suite now stages them with `cp -R` into
  `~/.cache/ctag-annotator/qtplugins`, refreshed each run.

Open: fish reference photos; EN/ES toggle; Intel-Mac + Windows builds.
