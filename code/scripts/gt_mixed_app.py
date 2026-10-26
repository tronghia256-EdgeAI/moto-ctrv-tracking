"""Mixed GT desktop review: python -m code.scripts.gt_mixed_app."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from code.scripts.gt_review_app import ReviewApp
from code.scripts.gt_mixed_review import BUNDLE, TRACKS_DIR, MixedStore, finalize_mixed, prepare


class MixedReviewApp(ReviewApp):
    window_title = "CAAP — Mixed GT · Nghĩa review"
    help_text = "A: chấp nhận đề xuất · K: giữ nguyên · U: chưa rõ · V: xem video / chỉnh box · E: đã sửa thủ công"

    def load_store(self):
        return MixedStore(self.args.bundle)

    def check_store(self, store):
        store.check_inputs(images=True)
        store.preview()
        if store.manifest.get('review_scope')=='ROI_ONLY':
            self.window_title='CAAP — Mixed GT · ROI ONLY · Nghĩa review'

    def refresh(self):
        tally = self.store.tally
        case = self.store.manifest["cases"][self.index]
        row = self.store.rows[self.index]
        self.progress.set(f"Đã duyệt {len(self.store.rows) - tally['PENDING']}/{len(self.store.rows)}"
                          f" · Còn trống {tally['PENDING']} · Đã hiển thị {len(set(self.store.journal['seen_frames']))}"
                          f"/{self.store.manifest['n_frames']} frame")
        self.detail.set(f"#{case['rank']} · {case['kind']} · ID nguồn {case['ids'] or 'tất cả'}"
                        f" · MOT {case['before'][1]} → {case['after'][1]} · {row['decision'] or 'CHƯA DUYỆT'}")
        self.help_label.configure(text=case["question"])
        self.controls[1].configure(text={"SPLIT": "Tách ID [A]", "JUMP": "Tách ID [A]",
                                        "MERGE": "Nối ID [A]", "DUP": "Cùng xe: gộp [A]",
                                        "GHOST": "Không có xe: xóa [A]", "AUDIT": "Đã kiểm tra [A]"}
                                  .get(case["kind"], "Mở để kiểm tra [A]"))
        self.controls[2].configure(text={"SPLIT": "Cùng xe: giữ [K]", "JUMP": "Giữ ID [K]",
                                        "MERGE": "Hai xe: giữ riêng [K]", "DUP": "Hai xe: giữ [K]",
                                        "GHOST": "Có xe thật [K]", "CLASSCHECK": "Đúng class [K]"}
                                  .get(case["kind"], "Không cần sửa [K]"))
        self.finish_button.configure(state="normal" if not self.busy and not tally["PENDING"] else "disabled")

    def actions(self):
        return (("← Trước", lambda: self.select(self.index - 1)),
                ("Chấp nhận [A]", self.accept), ("Giữ nguyên [K]", lambda: self.verdict("KEEP")),
                ("Chưa rõ [U]", lambda: self.verdict("")), ("Sau →", lambda: self.select(self.index + 1)),
                ("Xem video / sửa [V]", self.preview), ("Đã sửa [E]", lambda: self.verdict("EDIT")),
                ("Chưa duyệt [N]", self.pending), ("Ảnh 100% / Vừa cửa sổ", self.toggle_size),
                ("Tải lại CSV", self.reload), ("Điều chỉnh ROI", self.roi_editor))

    def roi_editor(self):
        if self.busy or not self.save_note():
            return
        from code.scripts.gt_roi_app import ROIEditor
        import tkinter as tk
        try:
            window = tk.Toplevel(self.root)
            ROIEditor(window, self.args.bundle, self.reviewer.get().strip() or 'Nghĩa')
            window.grab_set()
        except (ValueError,OSError) as error:
            messagebox.showerror('Không mở được ROI',str(error),parent=self.root)

    def shortcuts(self):
        return {"a": self.accept, "k": lambda: self.verdict("KEEP"), "u": lambda: self.verdict(""),
                "e": lambda: self.verdict("EDIT"), "v": self.preview, "n": self.pending,
                "Left": lambda: self.select(self.index - 1), "Right": lambda: self.select(self.index + 1)}

    def accept(self):
        case = self.store.manifest["cases"][self.index]
        if case["kind"] == "AUDIT":
            from code.scripts.gt_mixed_review import event_relevant
            edited = any(event_relevant(e, case) for e in self.store.journal["events"])
            self.verdict("EDIT" if edited else "KEEP")
        elif case["proposal"] is None:
            self.preview()
        else:
            self.verdict("APPLY")

    def preview(self):
        if self.busy or not self.save_note():
            return
        from code.scripts.gt_mixed_editor import FrameEditor
        try:
            FrameEditor(self, self.store.manifest["cases"][self.index])
        except (ValueError, OSError) as error:
            messagebox.showerror("Không thể mở video", str(error), parent=self.root)

    def run_finalize(self, reviewer, progress):
        return finalize_mixed(self.store, reviewer, self.args.tracks_dir, progress=progress)

    def completion_message(self, result):
        return (f"GT mixed và báo cáo đã lưu tại:\n{result}\n\nPre-labels gốc giữ nguyên. "
                "Kết quả này dùng cho phát triển/validation; tập test độc lập cần xác nhận riêng.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument('--roi-segment',choices=('mixed','dense','freeflow'),default='mixed')
    parser.add_argument("--tracks-dir", type=Path, default=TRACKS_DIR)
    parser.add_argument("--reviewer", default="Nghĩa")
    args = parser.parse_args()
    if args.bundle is None:
        from code.scripts.gt_roi_scope import active_bundle
        args.bundle = active_bundle(args.roi_segment)
    if not args.bundle.exists():
        prepare(bundle=args.bundle)
    root = tk.Tk()
    try:
        MixedReviewApp(root, args)
    except (ValueError, OSError, KeyError) as error:
        root.withdraw()
        messagebox.showerror("Không thể mở mixed GT", str(error), parent=root)
        root.destroy()
        raise SystemExit(1) from error
    root.mainloop()


if __name__ == "__main__":
    main()
