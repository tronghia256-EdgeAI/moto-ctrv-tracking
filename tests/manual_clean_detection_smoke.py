"""Native GUI smoke using temporary synthetic data only; not normal pytest.

python -c "import runpy; runpy.run_path('tests/manual_clean_detection_smoke.py', run_name='__main__')"
"""
import json
import contextlib
import io
import runpy
import tempfile
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from code.scripts.clean_detection_app import BoxEditor, CleanApp
from code.scripts.gt_review import sha256


def main():
    with tempfile.TemporaryDirectory(prefix="caap-dataset-gui-") as folder:
        fixture = runpy.run_path("tests/test_clean_detection_dataset.py")["dataset"]
        with contextlib.redirect_stdout(io.StringIO()):
            source, _, bundle, store = fixture.__wrapped__(Path(folder))
        source_hashes = {p: sha256(p) for p in source.rglob("*") if p.is_file()}
        root = tk.Tk()
        try:
            app = CleanApp(root, bundle)
            root.update(); app.draw(); root.update()
            assert all(photo is not None for photo in app.photos)
            path = app.current_path
            app.note.insert(0, "Ghi chú có dấu — synthetic smoke")
            assert app.save_note()
            app.metadata["source_id"].set("synthetic:"+path)
            app.metadata["domain"].set("cctv")
            app.metadata["provenance_confirmed"].set(True)
            app.metadata["unaugmented_confirmed"].set(True)
            app.save_metadata()
            assert app.store.eligible(path)
            with patch("code.scripts.clean_detection_app.simpledialog.askstring", return_value="Synthetic uncertain image"):
                app.set_status("QUARANTINE")
            assert app.store.image(path)["status"] == "QUARANTINE"
            app.set_status("KEEP")
            editor = BoxEditor(app); root.update()
            row = app.store.rows[path]
            editor.press(SimpleNamespace(x=.25*row["width"]*editor.scale, y=.25*row["height"]*editor.scale))
            assert editor.selected == 0
            editor.classes.set("car"); editor.reclassify()
            assert app.store.boxes(path)[0][0] == 1
            editor.set_mode("add")
            editor.press(SimpleNamespace(x=.4*row["width"]*editor.scale, y=.8*row["height"]*editor.scale))
            editor.release(SimpleNamespace(x=.6*row["width"]*editor.scale, y=.9*row["height"]*editor.scale))
            assert len(app.store.boxes(path)) == 3
            editor.undo(); assert len(app.store.boxes(path)) == 2
            editor.selected = 0; editor.delete(); assert len(app.store.boxes(path)) == 1
            editor.undo(); assert len(app.store.boxes(path)) == 2
            editor.close(); root.update()
            app.overlay.set(False); app.draw(); app.overlay.set(True)
            app.reload(); assert app.store.boxes(path)[0][0] == 1
            for path in app.store.rows:
                meta = dict(source_id="synthetic:"+path, clip_id="", event_id="", camera_id="",
                            domain="cctv", provenance_confirmed=True, unaugmented_confirmed=True)
                app.store.update_images([path], reviewer="Synthetic reviewer", metadata=meta)
            # Exercise actual navigation/display for every required review image.
            for i, case in enumerate(app.store.cases):
                app.case_index, app.member_index = i, 0; app.show_case(); root.update()
                for path in app.store.review_requirement(case):
                    app.member_index = case["members"].index(path)
                    app.show_member(); root.update()
                app.store.review_case(case["rank"], note="Synthetic fixture")
            app.show_case()
            with patch("code.scripts.clean_detection_app.messagebox.showinfo"), \
                 patch("code.scripts.clean_detection_app.messagebox.showerror") as error:
                app.finish()
                deadline = time.monotonic()+30
                while app.busy and time.monotonic() < deadline:
                    root.update(); time.sleep(.01)
                assert not app.busy
                error.assert_not_called()
            results = [p for p in (bundle / "releases").iterdir() if not p.name.startswith(".")]
            assert len(results) == 1
            report = json.loads((results[0]/"report.json").read_text())
            assert report["mode"] == "GROUPED_DETECTION"
            assert all(sha256(p) == digest for p, digest in source_hashes.items())
            app.close()
            print("GUI smoke passed: image comparisons, notes, provenance, quarantine/keep, draw/edit/undo boxes, resume, grouped export; source unchanged.")
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
