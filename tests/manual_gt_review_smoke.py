"""Optional desktop smoke check. Uses only temporary synthetic GT and images.

Run from the repo root:
    python -c "import runpy; runpy.run_path('tests/manual_gt_review_smoke.py', run_name='__main__')"
Requires a graphical session; excluded from normal headless pytest collection.
"""
import csv
import json
import tempfile
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from code.scripts.gt_review_app import JoinPlayer, ReviewApp


def write_csv(path, row):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    with tempfile.TemporaryDirectory(prefix="caap-gui-smoke-") as name:
        directory = Path(name)
        row = dict(rank="1", sheet="001_id1_f0-1.png", gt_id="1", cut_at_frame="2",
                   decision="", note="")
        write_csv(directory / "decisions.csv", row)
        write_csv(directory / "candidates.csv", dict(rank="1", gt_id_INTERNAL_not_cvat="1",
                  cvat_before="0", cvat_after="1", mot_before="1", mot_after="2",
                  abs_before="0", abs_after="1"))
        (directory / "gt.txt").write_text("1,1,10,20,30,40,1,2,1\n2,1,12,22,30,40,1,2,1\n", encoding="utf-8")
        picture = Image.new("RGB", (990, 752), "white")
        picture.save(directory / row["sheet"])
        for frame in (1, 2):
            picture.save(directory / f"{frame:06d}.jpg")
        (directory / "tracks_baseline.json").write_text(json.dumps({"start_frame": 0, "frames": [
            [[1, 1, [10, 20, 40, 60]]], [[1, 1, [12, 22, 42, 62]]]]}), encoding="utf-8")
        args = SimpleNamespace(decisions=directory / "decisions.csv", candidates=directory / "candidates.csv",
                               gt=directory / "gt.txt", tracks_dir=directory, frames_dir=directory,
                               segment=(0, 2), reviewer="Synthetic test")
        root = tk.Tk()
        root.withdraw()
        original = args.gt.read_bytes()
        try:
            app = ReviewApp(root, args)
            root.update()
            assert app.photo is not None
            assert app.store.tally["PENDING"] == 1
            app.note.insert("1.0", "Ghi chú có dấu")
            app.verdict("CUT")
            assert app.store.rows[0]["note"] == "Ghi chú có dấu"
            assert app.store.tally["CUT"] == 1
            app.verdict("")
            assert app.store.tally["PENDING"] == 1
            app.verdict("KEEP")
            app.toggle_size()
            player = JoinPlayer(root, args, app.store.rows[0])
            player.window.withdraw()
            player.toggle()
            root.update()
            assert player.photo is not None
            player.close()
            # Redirect the final run to this test directory; never publish test scores
            # in the repo's real experiment directory.
            from code.scripts.gt_review import finalize

            def synthetic_finalize(*arguments, **kwargs):
                return finalize(*arguments, out_dir=directory / "result", **kwargs)

            with patch("code.scripts.gt_review_app.finalize", synthetic_finalize), \
                    patch("code.scripts.gt_review_app.messagebox.showinfo"), \
                    patch("code.scripts.gt_review_app.messagebox.showerror") as error:
                app.finish()
                deadline = time.monotonic() + 15
                while app.busy and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                assert not app.busy, "Background finalization timed out"
                error.assert_not_called()
            assert (directory / "result/gt_reviewed.txt").is_file()
            assert args.gt.read_bytes() == original
            app.close()
            print("GUI smoke passed: image, notes, CUT/KEEP/pending, clip viewer, finalization; original preserved.")
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
