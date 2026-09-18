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

## 2026-09-18 — student report: crashes, "vanishing" boxes, hard to edit boxes

- **Crash hunt.** New `tests/fuzz_annotator.py` drives ~30 random actions
  (draw/select/edit/delete/undo/seek/label/switch video/zoom/focus/autosave)
  and records every exception + invariant break. 4 seeds × 3000 steps on the
  2026-09-08 code: **0 Python errors.** So the student is most likely on a
  pre-09-08 zip, or hitting a native crash we can't see. Can't tell which —
  there was no log. Fixed that:
  - `APP_VERSION` constant (shown in Help, stamped into logs).
  - Excepthook appends to `~/CTAG_Annotator/logs/errors.log`; the dialog
    points at the file. Guarded `sys.stderr` (may be unusable in the bundle).
  - `faulthandler` → `logs/hard_crash.log` for segfaults/aborts. Clean exit
    deletes it; if one exists at launch it is archived as
    `hard_crash_<stamp>.log` and the user is told to send the logs folder.
  - Startup "Could not open video" used `sys.exit(message)` — invisible in the
    .app, the window just never appeared (looks like a crash). Now a dialog.
  - Frame extraction ignored `cv2.imwrite` failures (full disk → silently
    missing frames). Now raises with free-space figure. Each 3-min clip is
    GBs of JPEGs in $TMPDIR and a crash leaked them; temp dirs now carry an
    `.owner_pid` marker and `_sweep_stale_frame_dirs()` removes orphans at
    startup (marker-less dirs only if >24 h old).
  - `_cleanup_preloads` iterated a dict the preload thread can insert into.
- **"Box is sometimes not there when I go back."** Boxes are single-frame
  (bbox stored only at the frame drawn). Coming back one frame off showed
  nothing. Now boxes within ±1 s are drawn **dashed** with "name at +0.4s";
  clicking one jumps to its frame and selects it. Picking a row in the Boxes
  list also seeks to the box's frame. (Did NOT make boxes span frames — that
  would change the data model / exports; revisit if they want tracks.)
- Also: Backspace deleted the selected box even when it wasn't on screen.
  Now Delete/Backspace only acts on a box visible on the current frame, and
  the status bar says "⌘Z to undo".
- **"Hard to change info on a placed box."** Old flow: select row → edit →
  press *Update* (easy to forget; and `_refresh_features` cleared the
  selection so a second edit silently did nothing). New:
  - Click a box (video or list) → **edit mode**: yellow banner, fields load
    the box, every change applies live (`_live_edit`), one undo step per
    editing session. *Update* replaced by *Done editing*; Esc also exits.
  - Next-box settings are stashed on entering edit mode and restored on exit,
    so editing never clobbers what you set up for the next box.
  - A freshly drawn box is deliberately NOT selected — otherwise changing the
    ID for the *next* fish would rewrite the one just drawn.
  - Arming Draw exits edit mode; clicking empty video exits edit mode.
  - `_refresh_features` keeps the selection (signals blocked).
- Tests: 22 → 33 (`tests/test_annotator.py`), all pass; fuzz clean.
