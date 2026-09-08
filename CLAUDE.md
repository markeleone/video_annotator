# CTAG Annotator — project guide

Desktop app (PyQt6) for annotating shark camera-tag ("CTAG") footage from the
Costa Rica IRES fieldwork. Fork of Jaden's `video_annotator`, extended heavily
on the `annotator-only` branch.

## Run it

```bash
.venv/bin/python run_annotator.py                 # startup chooser
.venv/bin/python run_annotator.py --frames-dir DIR
.venv/bin/python tests/test_annotator.py          # 22-test headless suite
```

Build the shareable Mac app (see "Build gotchas" below):

```bash
.venv/bin/pyinstaller annotator.spec --noconfirm --clean \
  --distpath <TMP>/appdist --workpath <TMP>/appbuild
```

## Layout

| Path | What |
|---|---|
| `run_annotator.py` | The entire app (single file, ~6.5k lines). |
| `tests/test_annotator.py` | Headless regression suite; Qt runs offscreen. |
| `annotator.spec` | PyInstaller spec (hiddenimports incl. openpyxl). |
| `dist/CTAG Annotator.zip` | Built app for collaborators (git-ignored). |
| `~/CTAG_Annotator/` | **User config**, not in the repo: `taxonomy.csv`, `behaviors.csv`, `habitats.csv`, `settings.json`, `recent.json`, and `annotations/`. |

## Data model

Four independent per-frame label axes, each "persist until changed":
**Movement**, **Social**, **Habitat**, **Visibility**. Plus per-frame
bounding-box features, timestamped comments, a whole-video note, highlight
clips, and a conspecific shark log.

Saved as JSON (run-length-encoded segments). Loading migrates older formats —
never assume a field exists.

## Decisions worth knowing (and things already rejected)

- **Behavior is two axes, not one.** Movement + Social are independent so
  "foraging while parallel swimming" is expressible.
- **Conspecific sharks live ONLY on the Social bar**, via the ♂/♀/? + Log
  control, and each entry applies to the **whole social segment** it sits in
  (derived live from `social_segment_at`, so extending the interaction extends
  the entry). Do **not** re-add sex fields to the fish-ID card — that was
  explicitly removed as redundant double-entry.
- **`"With 1+ sharks"` is retired** from the Social bar (superseded by the
  shark log) but is deliberately **NOT stripped from saved annotations** — the
  5/25 deployment uses it for ~11% of the time. See `RETIRED_SOCIAL`.
- **Labels are user-editable CSV** (`taxonomy/behaviors/habitats`), because the
  ethogram is still being discovered in the field. Defaults seed on first run;
  Labels ▸ Manage labels can add/remove/import/restore.
- **Taxonomy is hierarchical.** `GENUS_FAMILY` maps all 103 genera to families
  and `FAMILY_COMMON` gives each family a plain-English name. An ID may stop at
  family or genus — `feat.id_level()` records how far it got. Never force the
  annotator through family → genus → species.
- **The label bars are a "pen", not a readout.** They show what will be written
  next. A fresh video starts blank (so an accidental scrub records nothing);
  seeking onto a labeled frame adopts that label; seeking over blank frames
  keeps your selection. "None" sets the pen to erase.
- **Saves are always local.** Autosave writes to `~/CTAG_Annotator/annotations/`,
  never next to the video — writes into a Google Drive File Provider mount fail
  silently. `is_cloud_path()` also redirects the Save-JSON default.
- **Qt stylesheet:** do not restyle `QComboBox::drop-down`. Without a valid
  arrow image Qt drops the native caret and the combos look like plain text
  fields. Qt also does not support the CSS border-triangle trick (paints a
  solid block) — tried and reverted.
- **Rejected:** a scroll area around the label bars (looked broken); removing
  "Very bad" from Visibility (added back by request).

## Build gotchas (macOS)

1. **This checkout is on an iCloud-synced Desktop.** File-provider xattrs break
   `codesign` ("resource fork … not allowed"). Build with `--distpath`/
   `--workpath` pointing **outside** the synced folder, then copy the zip back.
2. `~/miniforge3/bin/codesign` shadows Apple's — always use `/usr/bin/codesign`.
3. Output is **arm64 only**, ad-hoc signed (recipients: right-click ▸ Open).
4. Qt plugins staged for headless tests must be copied with **`cp -R`**, not
   `shutil.copytree` — a copytree'd dylib is rejected by Qt on macOS. The test
   suite stages them in `~/.cache/ctag-annotator/qtplugins` and refreshes every
   run (a stale copy from an older PyQt6 fails the same way).

## Still open

- Reference photos of fish in the app.
- English / Español toggle.
- No Intel-Mac or Windows build.
- Work lives on the fork `markeleone/video_annotator` (branch
  `annotator-only`); no push access to `jadenvc/video_annotator`.
