"""
Headless regression suite for the CTAG Annotator.

Run:  .venv/bin/python tests/test_annotator.py

Covers the behaviour that has actually broken in the past — the bbox slot
crash, taxonomy ID levels, label erasing, autosave triggers, legacy file
migration, and the exports. Qt runs offscreen; no window appears.
"""
import os, sys, json, shutil, tempfile, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_HOME = os.path.expanduser("~")            # capture before overriding HOME
WORK = tempfile.mkdtemp(prefix="ctag_tests_")
os.environ["HOME"] = WORK                      # isolate ~/CTAG_Annotator

# Qt's plugin cache is keyed by path; copying the plugins to a fresh dir makes
# the offscreen platform load reliably.
import glob as _glob
_plat_src = _glob.glob(os.path.join(REPO, ".venv/lib/python*/site-packages/PyQt6/Qt6/plugins/platforms"))
# NOT inside the repo: this checkout lives on an iCloud-synced Desktop, and
# Qt refuses to load plugins from there (same file-provider xattrs that break
# code-signing). ~/.cache is not synced.
_plat_dst = os.path.join(REAL_HOME, ".cache", "ctag-annotator", "qtplugins")
os.makedirs(os.path.dirname(_plat_dst), exist_ok=True)
if _plat_src:
    # Refresh every run: a copy left from an older PyQt6 silently fails to load
    # ("could not find the platform plugin") once it no longer matches QtCore.
    # Use `cp -R`, NOT shutil.copytree — a copytree'd dylib is rejected by Qt
    # on macOS (provenance/signing metadata), while byte-identical `cp -R`
    # output loads fine. Verified both ways.
    shutil.rmtree(_plat_dst, ignore_errors=True)
    subprocess.run(["cp", "-R", _plat_src[0], _plat_dst], check=True)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _plat_dst
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, REPO)
import numpy as np, cv2
import run_annotator as ra
ra.load_all_label_config()

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QPoint
QMessageBox.critical = staticmethod(lambda *a, **k: None)   # never block
QMessageBox.question = staticmethod(lambda *a, **k: None)

app = QApplication(sys.argv)
app.setStyle("Fusion")
ra._apply_light_palette(app)
app.setStyleSheet(ra.APP_QSS)

FRAMES = os.path.join(WORK, "frames")
os.makedirs(FRAMES, exist_ok=True)
for i in range(10):
    cv2.imwrite(os.path.join(FRAMES, f"{i:05d}.jpg"),
                np.full((720, 1280, 3), (140, 110, 60), np.uint8))
PREP = ra.prepare_source(FRAMES, True)

PASS, FAIL = [], []

def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name); print(f"  PASS  {name}")
        except Exception as e:
            FAIL.append((name, e)); print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    return deco

def win(labeling=True):
    w = ra.MainWindow(video_path=PREP["frames_source"], fps=PREP["fps"],
                      total_frames=PREP["total_frames"], video_w=PREP["video_w"],
                      video_h=PREP["video_h"], is_frame_dir=True)
    w.resize(1280, 800); w.show()
    app.processEvents(); app.processEvents()
    w._labeling = labeling
    for k in w._label_axes: w._label_axes[k] = True
    return w

def draw_box(w, i=0):
    ox, oy = w._view[6], w._view[7]
    w._on_bbox(QPoint(ox + 60 + i*10, oy + 60), QPoint(ox + 240 + i*10, oy + 220))


print("\n== bounding boxes (the slot-crash regression) ==")

@check("box with no ID at all does not crash")
def _():
    w = win(); draw_box(w)
    assert len(w.store.features) == 1

@check("box identified only to family")
def _():
    w = win()
    w.category_combo.setCurrentText("Teleost fish"); w._on_category_changed("Teleost fish")
    w.taxon.set_value("Serranidae", "", "", ""); draw_box(w)
    f = w.store.features[0]
    assert f.family == "Serranidae" and f.id_level() == "family"

@check("box identified to species back-fills genus/family")
def _():
    w = win()
    w.category_combo.setCurrentText("Shark"); w._on_category_changed("Shark")
    w.taxon.set_value("Carcharhinidae", "Negaprion", "Negaprion brevirostris", "Lemon shark")
    draw_box(w)
    f = w.store.features[0]
    assert (f.species, f.genus, f.id_level()) == ("Negaprion brevirostris", "Negaprion", "species")

@check("several boxes in a row")
def _():
    w = win()
    for i in range(3): draw_box(w, i)
    assert len(w.store.features) == 3


print("\n== labels ==")

@check("visibility is the 5-point scale")
def _():
    assert ra.VISIBILITY_OPTIONS == ["Very bad", "Bad", "Fine", "Good", "Very good"]

@check("legacy 3-point visibility migrates on load")
def _():
    p = os.path.join(WORK, "old_vis.json")
    json.dump({"video_path": "v", "fps": 30, "total_frames": 10, "video_w": 80, "video_h": 60,
               "scene": {"visibility_segments": [{"start": 0, "end": 4, "value": "Poor"},
                                                 {"start": 5, "end": 9, "value": "Moderate"}]}},
              open(p, "w"))
    st = ra.FeatureStore.load_json(p)
    assert st.visibility_per_frame[0] == "Bad" and st.visibility_per_frame[9] == "Fine"

@check("a fresh video starts blank so an accidental scrub records nothing")
def _():
    w = win()
    assert w._current_movement is None and w._current_habitat is None
    w._on_slider_seek(9)
    assert all(v is None for v in w.store.movement_per_frame)

@check("selecting None erases a stretch on re-scrub")
def _():
    w = win()
    for f in range(5): w.store.set_habitat(f, "Mangrove")
    assert None in w._habitat_btns
    w._on_habitat_selected(None)
    w.current_frame_idx = 0
    for k in ("movement", "social", "visibility"): w._label_axes[k] = False
    w._on_slider_seek(4)
    assert all(v is None for v in w.store.habitat_per_frame[0:5])

@check("the pen persists over blank frames and adopts stored labels")
def _():
    w = win()
    for k in ("social", "habitat", "visibility"): w._label_axes[k] = False
    w._on_movement_selected("Foraging")
    w.current_frame_idx = 0
    w._on_slider_seek(3)
    assert w.store.movement_per_frame[2] == "Foraging"
    w.seek_to(7)
    assert w._current_movement == "Foraging"      # not lost over blanks
    w._on_slider_seek(9)
    assert w.store.movement_per_frame[8] == "Foraging"

@check("'With 1+ sharks' retired from the bar but kept in old data")
def _():
    assert "With 1+ sharks" not in ra.SOCIAL_OPTIONS
    p = os.path.join(WORK, "old_soc.json")
    json.dump({"video_path": "v", "fps": 30, "total_frames": 10, "video_w": 80, "video_h": 60,
               "scene": {"social_segments": [{"start": 0, "end": 9, "value": "With 1+ sharks"}]}},
              open(p, "w"))
    st = ra.FeatureStore.load_json(p)
    assert st.social_per_frame[5] == "With 1+ sharks"
    tl = ra.AnnotationTimeline(st, 30.0, 10)
    assert "With 1+ sharks" in tl._social_colors


print("\n== taxonomy ==")

@check("taxonomy builds with every genus mapped to a family")
def _():
    t = ra.TAXONOMY
    assert len(t.rows) > 250
    unmapped = [r["genus"] for r in t.rows if r["genus"] and not r["family"]]
    assert not unmapped, unmapped

@check("family common names cover every mapped family")
def _():
    missing = sorted(set(ra.GENUS_FAMILY.values()) - set(ra.FAMILY_COMMON))
    assert not missing, missing
    assert ra.family_display("Serranidae") == "Serranidae — groupers & sea basses"
    assert ra.family_from_display("Serranidae — groupers & sea basses") == "Serranidae"

@check("search finds species, genus and family (incl. common names)")
def _():
    t = ra.TAXONOMY
    assert any(h["kind"] == "species" for h in t.search("lemon"))
    assert any(h["kind"] == "genus" for h in t.search("Caranx"))
    assert any(h["kind"] == "family" for h in t.search("grouper"))

@check("legacy free-text species strings split into ranks")
def _():
    p = os.path.join(WORK, "old_feats.json")
    mk = lambda sp, cat: {"name": "x", "init_frame": 0, "end_frame": 0, "init_type": "bbox",
                          "init_coords": [1, 2, 3, 4], "species_category": cat, "species": sp}
    json.dump({"video_path": "v", "fps": 30, "total_frames": 10, "video_w": 80, "video_h": 60,
               "scene": {}, "features": [mk("Lemon shark (Negaprion brevirostris)", "Shark"),
                                         mk("Hypanus sp.", "Ray"),
                                         mk("Serranidae", "Teleost fish")]}, open(p, "w"))
    a, b, c = ra.FeatureStore.load_json(p).features
    assert (a.genus, a.species, a.common) == ("Negaprion", "Negaprion brevirostris", "Lemon shark")
    assert (b.genus, b.id_level()) == ("Hypanus", "genus")
    assert (c.family, c.id_level()) == ("Serranidae", "family")


print("\n== saving ==")

@check("autosave fires for every kind of work")
def _():
    for mut in (lambda s: s.shark_log.append({"frame": 1, "males": 1, "females": 0, "unknown": 0}),
                lambda s: setattr(s, "video_note", "murky"),
                lambda s: s.set_habitat(0, "Mangrove"),
                lambda s: s.add_note(2, "check"),
                lambda s: s.clips.append({"name": "c", "start": 0, "end": 3})):
        w = ra.MainWindow.__new__(ra.MainWindow)
        w.store = ra.FeatureStore("/v.MOV", 30.0, 100, 80, 60); w.video_path = "/v.MOV"
        mut(w.store)
        assert w._has_annotations()

@check("a drawn box round-trips through autosave")
def _():
    w = win(); draw_box(w); w._autosave("test")
    p = w._autosave_path()
    assert os.path.exists(p)
    assert len(ra.FeatureStore.load_json(p).features) == 1

@check("crash guard flushes work instead of aborting")
def _():
    w = win(); draw_box(w)
    ra._install_crash_guard()
    try: raise RuntimeError("simulated slot failure")
    except Exception: sys.excepthook(*sys.exc_info())
    assert os.path.exists(w._autosave_path())

@check("folder scan agrees with the autosave path")
def _():
    folder = os.path.join(WORK, "deployment"); os.makedirs(folder, exist_ok=True)
    vids = []
    for i in range(4):
        p = os.path.join(folder, f"2026052{i}_110000.MOV")
        open(p, "wb").write(b"\0"); vids.append(p)
    open(os.path.join(folder, "._junk.MOV"), "wb").write(b"x")
    found = ra.list_videos_in_folder(folder)
    assert len(found) == 4
    os.makedirs(ra.LOCAL_ANNOTATIONS_DIR, exist_ok=True)
    open(ra.autosave_path_for(found[0]), "w").write("{}")
    assert ra.is_annotated(found[0]) and not ra.is_annotated(found[1])


print("\n== ui state ==")

@check("focus mode hides and restores the panels")
def _():
    w = win()
    assert w._right_panel.isVisible()
    w._toggle_focus_mode()
    assert not w._right_panel.isVisible() and not w._bars_host.isVisible()
    w._toggle_focus_mode()
    assert w._right_panel.isVisible() and w._bars_host.isVisible()

@check("settings persist to a fresh window")
def _():
    w = win()
    w._label_axes["social"] = False
    w._axis_checkboxes["social"].setChecked(False)
    w.speed_combo.setCurrentText("2x")
    ra.save_settings(w._collect_settings())
    w2 = ra.MainWindow(video_path=PREP["frames_source"], fps=PREP["fps"],
                       total_frames=PREP["total_frames"], video_w=PREP["video_w"],
                       video_h=PREP["video_h"], is_frame_dir=True)
    assert w2._label_axes["social"] is False
    assert w2.speed_combo.currentText() == "2x"
    ra.save_settings({})      # don't leak state into later tests

@check("window fits a small laptop screen")
def _():
    w = win()
    assert w.minimumWidth() <= 1280 and w.minimumHeight() <= 800, \
        (w.minimumWidth(), w.minimumHeight())


print("\n== exports ==")

@check("excel, plots and video export")
def _():
    import matplotlib; matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    import openpyxl
    w = win()
    for f in range(2, 8):
        w.store.set_movement(f, "Foraging"); w.store.set_social(f, "Solo")
        w.store.set_habitat(f, "Mangrove"); w.store.set_visibility(f, "Very good")
    w.store.add_feature(ra.TrackedFeature("a", 3, 3, "bbox", [10, 10, 40, 40], count=1,
        species_category="Shark", family="Carcharhinidae", genus="Negaprion",
        species="Negaprion brevirostris", common="Lemon shark", confidence="High"),
        {}, {3: (10, 10, 40, 40)})
    wb = w._build_workbook(openpyxl)
    hdr = [wb["Sightings"].cell(row=1, column=c).value for c in range(1, 19)]
    for h in ("Family", "Genus", "Species", "Common Name", "ID Level"):
        assert h in hdr, (h, hdr)
    assert w._render_plots(os.path.join(WORK, "p"), plt, Counter, defaultdict)
    out = os.path.join(WORK, "o.mp4"); w._render(out)
    assert os.path.getsize(out) > 0


print(f"\n{'='*56}\n{len(PASS)} passed, {len(FAIL)} failed")
for name, err in FAIL:
    print(f"  FAILED: {name}\n          {type(err).__name__}: {err}")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if FAIL else 0)
