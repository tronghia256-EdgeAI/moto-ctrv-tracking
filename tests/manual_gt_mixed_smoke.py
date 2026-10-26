"""Desktop smoke on synthetic data only; no real mixed/dense decisions are touched.

python -c "import runpy; runpy.run_path('tests/manual_gt_mixed_smoke.py', run_name='__main__')"
"""
import json
import tempfile
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from code.scripts.gt_mixed_app import MixedReviewApp
from code.scripts.gt_mixed_editor import FrameEditor
from code.scripts.gt_mixed_review import finalize_mixed, prepare


def main():
    with tempfile.TemporaryDirectory(prefix="caap-mixed-gui-") as name:
        directory = Path(name)
        source = directory / "source"
        (source / "gt").mkdir(parents=True)
        (source / "img1").mkdir()
        original = ("1,1,100,100,20,40,1,2,1\n2,1,105,100,20,40,1,2,0.7\n"
                    "4,2,115,100,20,40,1,2,1\n5,2,120,100,20,40,1,2,1\n"
                    "3,3,250,100,20,40,1,1,1\n").encode("utf-8")
        (source / "gt/gt.txt").write_bytes(original)
        for frame in range(1, 7):
            Image.new("RGB", (800, 450), "white").save(source / f"img1/{frame:06d}.jpg")
        bundle = prepare(source, directory / "bundle", segment=(0, 6), progress=None)
        args = SimpleNamespace(bundle=bundle, tracks_dir=directory, reviewer="Nghĩa")
        root = tk.Tk()
        root.withdraw()
        try:
            with patch("code.scripts.gt_mixed_app.messagebox.showinfo"), \
                    patch("code.scripts.gt_mixed_app.messagebox.showerror") as error:
                app = MixedReviewApp(root, args)
                root.update()
                assert app.photo is not None
                assert app.store.manifest["cases"][0]["kind"] == "MERGE"
                app.accept()
                assert app.store.rows[0]["decision"] == "APPLY"
                assert app.store.preview()[0][(4, 2)][1] == "1"
                editor = FrameEditor(app, app.store.manifest["cases"][app.index])
                editor.window.withdraw()
                root.update()
                editor.zoom.set("100%")
                editor.render()
                editor.selected = 1
                editor.class_name.set("car")
                editor.reclassify()
                assert app.store.preview()[0][(5, 2)][7] == "1"
                editor.frame.set(2)
                editor.load_frame()
                editor.selected = 1
                editor.set_mode("draw")
                editor.press(SimpleNamespace(x=106 * editor.scale, y=99 * editor.scale))
                editor.drag(SimpleNamespace(x=130 * editor.scale, y=139 * editor.scale))
                editor.release(SimpleNamespace(x=130 * editor.scale, y=139 * editor.scale))
                assert float(app.store.preview()[0][(2, 1)][2]) == 106
                editor.frame.set(3)
                editor.load_frame()
                editor.new_vehicle()
                tid = editor.selected
                editor.class_name.set("motorcycle")
                editor.press(SimpleNamespace(x=300 * editor.scale, y=200 * editor.scale))
                editor.release(SimpleNamespace(x=325 * editor.scale, y=240 * editor.scale))
                assert (3, tid) in app.store.preview()[0]
                editor.copy_next()
                assert (4, tid) in app.store.preview()[0]
                editor.delete_box()
                assert (4, tid) not in app.store.preview()[0]
                editor.undo()
                assert (4, tid) in app.store.preview()[0]
                editor.split()
                assert app.store.preview()[0][(3, tid)][1] != app.store.preview()[0][(4, tid)][1]
                editor.undo()
                assert app.store.preview()[0][(3, tid)][1] == app.store.preview()[0][(4, tid)][1]
                for frame in range(1, 7):
                    editor.frame.set(frame)
                    editor.load_frame()
                editor.close()
                assert len(app.store.journal["seen_frames"]) == 6
                app.verdict("EDIT")
                for index, row in enumerate(app.store.rows):
                    if not row["decision"]:
                        app.store.save(index, "KEEP", "Synthetic review")
                app.refresh()
                app.show()
                app.reload()
                assert app.store.tally["PENDING"] == 0

                def synthetic_finalize(store, reviewer, *arguments, **kwargs):
                    return finalize_mixed(store, reviewer, out_dir=directory / "result",
                                          evaluate=False, **kwargs)

                with patch("code.scripts.gt_mixed_app.finalize_mixed", synthetic_finalize):
                    app.finish()
                    deadline = time.monotonic() + 15
                    while app.busy and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    assert not app.busy, "Finalization timed out"
                error.assert_not_called()
                assert (source / "gt/gt.txt").read_bytes() == original
                report = json.loads((directory / "result/review.json").read_text(encoding="utf-8"))
                assert report["reviewer"] == "Nghĩa"
                assert report["displayed_frames"] == 6
                app.close()
                print("Mixed GUI smoke passed: merge, class, draw, new box, copy, delete, split, undo, resume, audit, finalization.")
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
