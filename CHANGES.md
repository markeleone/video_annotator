# CTAG Annotator — what's new

Changes to `run_annotator.py` on the `annotator-only` branch, building on
Jaden's original annotator. ~3,000 lines added across 7 commits.

Config files live in `~/CTAG_Annotator/`; annotations autosave to
`~/CTAG_Annotator/annotations/`.

---

## Labeling

- **Behavior split into two independent axes** — every frame carries a
  **Movement** label *and* a **Social** label, so "foraging while parallel
  swimming" is now expressible. Both persist until changed.
  - Movement: Burst swimming, Low-speed swimming, Stationary, Foraging, **Turn** *(new — for turn-rate vs. magnetometer)*
  - Social: Solo, Parallel swimming, Following a shark, Shark following tagged, Brief interaction, With 1+ sharks
- **Visibility axis** — 5-point scale: Very bad / Bad / Fine / Good / Very good.
- **Per-axis labeling checkboxes** (Mov / Soc / Hab / Vis) — untick axes to
  protect them while re-scrubbing, e.g. fix Habitat Sand → Gravel over a
  section without touching behavior labels.
- **Family-level fish ID** — family names are in the species dropdown; pick one
  when species-level ID isn't possible.
- **Confidence** (High / Medium / Low) per identified animal.

## Conspecific sharks (on the Social bar)

- Other sharks are logged as a **social observation**, not a fish-ID box:
  set ♂ / ♀ / ? counts on the Social bar and press **Log**.
- Each entry applies to the **whole social interaction** it sits in (e.g. all
  of a "Brief interaction"), and follows that interaction if you extend or
  relabel it. Logging again inside the same interaction replaces its entry.
- Shown as a band + ◆ + count on the Social timeline row, and in a jump-list.

## Customisable label sets (no code edits)

- **Movement, Social, Habitats and Species are all user-editable.**
  Defaults are the built-in lists.
- **Labels ▸ Manage labels…** — add, remove, **import a CSV**, or
  **restore any category to default**.
- Quick **＋** buttons on each bar / the species field for one-off additions.
- Everything is stored as plain CSV in `~/CTAG_Annotator/` for bulk editing.

## Highlight clips

- Mark **In / Out**, **Save Clip** (named, stored with the annotations), and
  **Export Clip** to MP4 — with a per-export choice of **clean footage or
  annotations burned in**. No second app needed for social-media cuts.

## Bounding boxes

- **Two-click drawing** (click two corners) instead of click-and-drag.
- **Armed state** is obvious — the button changes colour and prompts.
- **Click a box on the video to select it**, then **drag its corners** to edit.
- **Delete key** removes the selected animal.
- **Zoom** into part of the frame (Ctrl/⌘ + wheel, or ＋ / − / Fit).

## Comments & notes

- **Timestamped comments** at any frame ("ID this fish later"), with markers on
  the timeline; double-click to jump back.
- **Whole-video note** field for observations about the entire video.

## Saving & workflow

- **Autosave every 60 s** plus save-on-close, with a **resume prompt** on reopen.
- **Cloud-safe saving** — annotations always write to a local folder, fixing
  JSONs silently failing to save when videos are opened from the Google Drive
  (File Provider) mount. Failed saves fall back to a local copy with a clear
  message.
- **Multiple videos / folder loading** — open a folder of videos and switch via
  a playlist dropdown; the **next video preloads in the background** so
  switching is near-instant.
- **Undo** (⌘Z / Ctrl+Z) across boxes, labels, comments and clips.

## Keyboard shortcuts + Help

`F1` lists everything. Space play/pause · ←/→ step frame · Ctrl+←/→ jump 10 ·
B draw box · C comment · I/O clip in/out · Delete remove animal · ⌘Z undo ·
Ctrl +/− zoom · Ctrl+S save · Ctrl+E export clip.
(Letter keys are ignored while typing in a text field.)

## Exports

- **JSON** — full annotation format, backward-compatible with older files.
- **Excel** — Sightings, Species / Movement / Social / Habitat / Visibility
  summaries, per-axis timelines, **Shark Log** (with interaction start/end and
  duration), and Notes.
- **Plots** — species by habitat, Movement / Social / Habitat / Visibility
  composition, sightings over time, per-species timelines.
- **Annotated video** export.

## Interface

- Rebuilt for space and legibility: far more room for the video, tighter
  timeline tracks, and a scrollable control panel.
- **Fixed dark-mode rendering** — the app forces a consistent light theme, so
  button text no longer disappears on hover on dark-mode Macs.

---

## Known limits

- The prebuilt app (`dist/CTAG Annotator.zip`) is **Apple Silicon only** —
  Intel Macs and Windows need separate builds.
- It's ad-hoc signed: **right-click ▸ Open** on first launch.

## Not yet built

- Reference photos of fish in the app.
- English / Español language toggle.
