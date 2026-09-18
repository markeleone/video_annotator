"""
Randomised stress test: drive the annotator through thousands of random user
actions (draw/select/edit/delete boxes, undo, seek, label, switch videos,
save/load) and report every exception that escapes, with the action trail.

Run:  .venv/bin/python tests/fuzz_annotator.py [steps] [seed]
"""
import os, sys, random, traceback
sys.argv = sys.argv[:1] + sys.argv[1:]
_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
sys.argv = sys.argv[:1]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# reuse the suite's headless bootstrap (HOME isolation, Qt plugins, app)
import importlib.util
spec = importlib.util.spec_from_file_location("_boot", os.path.join(HERE, "test_annotator.py"))
src = open(os.path.join(HERE, "test_annotator.py")).read()
boot = src.split('print("\\n== bounding boxes')[0]
ns = {"__file__": os.path.join(HERE, "test_annotator.py"), "__name__": "_boot"}
exec(compile(boot, "boot", "exec"), ns)
ra, app, WORK, QPoint = ns["ra"], ns["app"], ns["WORK"], ns["QPoint"]
import numpy as np, cv2
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QMessageBox, QInputDialog
# no modal dialog may block the run
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QInputDialog.getText = staticmethod(lambda *a, **k: ("clip", True))

# second clip for playlist switching
F2 = os.path.join(WORK, "frames2")
os.makedirs(F2, exist_ok=True)
for i in range(14):
    cv2.imwrite(os.path.join(F2, f"{i:05d}.jpg"), np.full((720, 1280, 3), 90, np.uint8))

errors = []
trail = []
def hook(t, e, tb):
    errors.append(("".join(traceback.format_exception(t, e, tb)), list(trail[-12:])))
sys.excepthook = hook

w = ns["win"]()
w._playlist = [ns["FRAMES"], F2]
w._refresh_playlist()
rnd = random.Random(_SEED)

def pt():
    ox, oy, vw, vh = w._view[6], w._view[7], w._view[4], w._view[5]
    return QPoint(ox + rnd.randint(0, max(1, vw - 1)), oy + rnd.randint(0, max(1, vh - 1)))

def key(k):
    ev = QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier)
    w.keyPressEvent(ev)

def sel_row():
    n = w.feat_list.count()
    if n: w.feat_list.setCurrentRow(rnd.randrange(n))

def set_taxon():
    t = w.taxon
    ch = rnd.random()
    if ch < .25: t.set_value("", "", "", "")
    elif ch < .5 and t.family_combo.count() > 1:
        t.family_combo.setCurrentIndex(rnd.randrange(t.family_combo.count()))
    elif ch < .7 and t.genus_combo.count() > 1:
        t.genus_combo.setCurrentIndex(rnd.randrange(t.genus_combo.count()))
    elif ch < .85 and t.species_combo.count() > 1:
        t.species_combo.setCurrentIndex(rnd.randrange(t.species_combo.count()))
    else:
        t.species_combo.setEditText(rnd.choice(["lut", "Neg", "zzz", ""]))

def bar(axis):
    btns = getattr(w, f"_{axis}_btns")
    k = rnd.choice(list(btns.keys()))
    btns[k].click()

def edit_corner():
    fi = w._selected_feature_idx
    if fi is None: return
    a, b = pt(), pt()
    w._on_bbox_edited(a.x(), a.y(), b.x(), b.y())

ACTIONS = [
    ("draw", 6, lambda: w._on_bbox(pt(), pt())),
    ("arm", 2, lambda: w.bbox_btn.setChecked(not w.bbox_btn.isChecked())),
    ("select", 6, sel_row),
    ("click_video", 5, lambda: w._on_video_click(pt())),
    ("update", 4, w._update_feature),
    ("delete", 2, w._delete_feature),
    ("key_del", 2, lambda: key(Qt.Key.Key_Backspace)),
    ("key_esc", 2, lambda: key(Qt.Key.Key_Escape)),
    ("undo", 4, w._undo),
    ("seek", 6, lambda: w._on_slider_seek(rnd.randrange(w.total_frames))),
    ("step", 4, lambda: w.seek_to(w.current_frame_idx + rnd.choice([-1, 1]))),
    ("tick", 3, w._tick),
    ("category", 3, lambda: w.category_combo.setCurrentIndex(rnd.randrange(w.category_combo.count()))),
    ("taxon", 4, set_taxon),
    ("count", 2, lambda: w.count_spin.setValue(rnd.randint(1, 30))),
    ("conf", 2, lambda: w.confidence_combo.setCurrentIndex(rnd.randrange(w.confidence_combo.count()))),
    ("name", 2, lambda: w.name_edit.setText(rnd.choice(["", "a", "fish 2"]))),
    ("edit_corner", 4, edit_corner),
    ("movement", 2, lambda: bar("movement")),
    ("social", 2, lambda: bar("social")),
    ("habitat", 1, lambda: bar("habitat")),
    ("visibility", 1, lambda: bar("visibility")),
    ("labeling", 1, lambda: w._toggle_label_mode(not w._labeling)),
    ("shark", 1, lambda: (w.shark_m_spin.setValue(rnd.randint(0, 2)), w._log_sharks())),
    ("note", 1, lambda: (w.note_edit.setText("n"), w._add_note())),
    ("clip", 1, lambda: (w._mark_in(), w.seek_to(w.current_frame_idx + 2), w._mark_out(), w._save_clip())),
    ("zoom", 2, lambda: w._zoom_at(rnd.choice([1.3, 1 / 1.3]), pt())),
    ("focus", 1, w._toggle_focus_mode),
    ("switch", 1, lambda: w._on_playlist_changed(rnd.randrange(2))),
    ("autosave", 1, w._autosave),
]
pool = [a for a in ACTIONS for _ in range(a[1])]
for step in range(_STEPS):
    name, _, fn = rnd.choice(pool)
    trail.append(f"{step}:{name}")
    if os.environ.get("FUZZ_VERBOSE"): print(step, name, flush=True)
    try:
        fn()
        app.processEvents()
    except Exception:
        errors.append((traceback.format_exc(), list(trail[-12:])))
    # invariants
    s = w.store
    try:
        assert len(s.features) == len(s.feature_bboxes) == len(s.feature_masks), "parallel lists out of sync"
        assert w._selected_feature_idx is None or 0 <= w._selected_feature_idx < len(s.features), "stale selection"
        assert w.feat_list.count() == len(s.features), "list/store mismatch"
        for i, f in enumerate(s.features):
            assert s.feature_bboxes[i], f"feature {i} has no boxes"
    except AssertionError as e:
        errors.append((f"INVARIANT: {e}", list(trail[-12:])))

seen = {}
for tb, tr in errors:
    sig = tb.strip().splitlines()[-1]
    seen.setdefault(sig, (tb, tr, 0))
    seen[sig] = (seen[sig][0], seen[sig][1], seen[sig][2] + 1)
print(f"{_STEPS} steps, {len(errors)} errors, {len(seen)} distinct")
for sig, (tb, tr, n) in seen.items():
    print("=" * 70, f"\n[{n}x] {sig}\ntrail: {tr}\n{tb[-1500:]}")
sys.exit(1 if errors else 0)
