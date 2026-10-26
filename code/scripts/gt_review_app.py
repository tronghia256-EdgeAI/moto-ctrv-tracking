"""Desktop reviewer: python -m code.scripts.gt_review_app (no API or server)."""
from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from code.scripts.gt_review import (
    CANDIDATES, DECISIONS, GT_IN, REPO_ROOT, TRACKS_DIR, DecisionStore, check_inputs, finalize,
)


class JoinPlayer:
    """Inspect actual intervening frames, including occlusions, without seeking raw ASF."""

    def __init__(self, parent, args, row):
        from code.scripts.gt_apply_splits import load_candidates
        from code.scripts.import_gt import load_mot

        candidate = next(r for r in load_candidates(args.candidates) if r["rank"] == row["rank"])
        self.args, self.tid = args, int(row["gt_id"])
        self.gt = load_mot(args.gt)
        self.low = max(1, int(candidate["mot_before"]) - 15)
        self.high = min(args.segment[1] - args.segment[0], int(candidate["mot_after"]) + 15)
        missing = [f for f in range(self.low, self.high + 1)
                   if not (args.frames_dir / f"{f:06d}.jpg").is_file()]
        if missing:
            raise ValueError(f"Missing frame images in {args.frames_dir}. First missing MOT frame: {missing[0]}")
        self.window = tk.Toplevel(parent)
        self.window.title(f"Đoạn nối — ID {self.tid}")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.label = ttk.Label(self.window)
        self.label.pack(padx=10, pady=8)
        self.info = tk.StringVar()
        ttk.Label(self.window, textvariable=self.info).pack()
        self.frame = tk.IntVar(value=self.low)
        tk.Scale(self.window, from_=self.low, to=self.high, orient="horizontal", length=800,
                 variable=self.frame, command=lambda _: self.draw()).pack(padx=10)
        self.button = ttk.Button(self.window, text="Phát / Dừng", command=self.toggle)
        self.button.pack(pady=8)
        self.job = None
        self.draw()

    def draw(self):
        frame = self.frame.get()
        with Image.open(self.args.frames_dir / f"{frame:06d}.jpg") as source:
            picture = source.convert("RGB")
        found = False
        for tid, box, _ in self.gt.get(frame, []):
            if tid == self.tid:
                ImageDraw.Draw(picture).rectangle(tuple(box), outline="#00ff70", width=3)
                found = True
        picture.thumbnail((1000, 640), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(picture)
        self.label.configure(image=self.photo)
        self.info.set(f"MOT {frame} · CVAT {frame - 1} · video {self.args.segment[0] + frame - 1}"
                      f" · {'Khung xanh: ID trong GT gốc' if found else 'GT gốc không có ID này ở frame này'}")

    def toggle(self):
        if self.job:
            self.window.after_cancel(self.job)
            self.job = None
        else:
            if self.frame.get() == self.high:
                self.frame.set(self.low)
            self.step()

    def step(self):
        self.job = None
        if self.frame.get() < self.high:
            self.frame.set(self.frame.get() + 1)
            self.draw()
            self.job = self.window.after(40, self.step)

    def close(self):
        if self.job:
            self.window.after_cancel(self.job)
        self.window.destroy()


class ReviewApp:
    help_text = "CUT: khác xe, tách ID.   KEEP: cùng xe, giữ ID.   Chưa rõ: để trống, xem lại."
    window_title = "CAAP — Review ground truth"

    def load_store(self):
        return DecisionStore(self.args.decisions)

    def check_store(self, store):
        check_inputs(store, self.args.candidates, self.args.gt, tuple(self.args.segment))

    def run_finalize(self, reviewer, progress):
        return finalize(self.store, self.args.candidates, self.args.gt, reviewer,
                        self.args.tracks_dir, tuple(self.args.segment), progress=progress)

    def actions(self):
        return (("← Trước", lambda: self.select(self.index - 1)),
                ("CUT [C]", lambda: self.verdict("CUT")),
                ("KEEP [K]", lambda: self.verdict("KEEP")),
                ("Chưa rõ [U]", lambda: self.verdict("")),
                ("Sau →", lambda: self.select(self.index + 1)),
                ("Chưa review [N]", self.pending), ("Xem đoạn nối [V]", self.preview),
                ("Ảnh 100% / Vừa cửa sổ", self.toggle_size), ("Tải lại CSV", self.reload))

    def shortcuts(self):
        return {"c": lambda: self.verdict("CUT"), "k": lambda: self.verdict("KEEP"),
                "u": lambda: self.verdict(""), "n": self.pending, "v": self.preview,
                "Left": lambda: self.select(self.index - 1),
                "Right": lambda: self.select(self.index + 1)}

    def completion_message(self, result):
        return (f"GT mới và báo cáo đã lưu tại:\n{result}\n\nGT gốc được giữ nguyên. "
                "Bước này chỉ xác nhận các vị trí tách ID đã liệt kê.")

    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        self.store = self.load_store()
        self.check_store(self.store)
        self.index = self.store.next_pending() or 0
        self.busy = False
        self.autosave = None
        self.resize_job = None
        self.messages = queue.Queue()
        self.full_size = False
        self.source_image = None
        self.photo = None
        root.title(self.window_title)
        root.geometry("1120x940")
        root.minsize(780, 650)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(root, padding=10)
        header.grid(row=0, column=0, sticky="ew")
        self.progress = tk.StringVar()
        self.detail = tk.StringVar()
        ttk.Label(header, textvariable=self.progress, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(header, textvariable=self.detail).pack(anchor="w", pady=(4, 0))
        self.help_label = ttk.Label(header, text=self.help_text, wraplength=1040)
        self.help_label.pack(anchor="w")

        panel = ttk.Frame(root)
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(panel, background="#20252b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(panel, orient="vertical", command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(panel, orient="horizontal", command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<Configure>", self.schedule_resize)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-int(e.delta / 120), "units"))

        footer = ttk.Frame(root, padding=10)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        ttk.Label(footer, text="Ghi chú:").grid(row=0, column=0, sticky="nw")
        self.note = tk.Text(footer, height=2, wrap="word", font=("Segoe UI", 10))
        self.note.grid(row=0, column=1, sticky="ew")
        self.note.bind("<KeyRelease>", self.schedule_save)
        self.note.bind("<<Paste>>", self.schedule_save)
        self.note.bind("<<Cut>>", self.schedule_save)
        actions = ttk.Frame(footer)
        actions.grid(row=1, column=0, columnspan=2, sticky="w", pady=8)
        self.controls = []
        for number, (label, command) in enumerate(self.actions()):
            button = ttk.Button(actions, text=label, command=command)
            button.grid(row=number // 5, column=number % 5, sticky="ew", padx=(0, 4), pady=2)
            self.controls.append(button)
        final = ttk.Frame(footer)
        final.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(final, text="Người review:").pack(side="left")
        self.reviewer = ttk.Entry(final, width=26)
        self.reviewer.insert(0, args.reviewer)
        self.reviewer.pack(side="left", padx=6)
        self.finish_button = ttk.Button(final, text="Hoàn tất → tạo GT và chấm lại", command=self.finish)
        self.finish_button.pack(side="left")
        self.status = tk.StringVar(value="Quyết định và ghi chú được tự lưu vào decisions.csv.")
        ttk.Label(footer, textvariable=self.status, wraplength=1020).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        for key, action in self.shortcuts().items():
            root.bind(f"<KeyPress-{key}>", lambda event, callback=action: self.shortcut(event, callback))
        root.bind("<Control-s>", lambda event: self.save_note())
        self.show()

    def shortcut(self, event, callback):
        if isinstance(self.root.focus_get(), (tk.Text, tk.Entry, ttk.Entry)) or self.busy:
            return
        callback()
        return "break"

    def refresh(self):
        tally = self.store.tally
        row = self.store.rows[self.index]
        self.progress.set(f"Đã review {len(self.store.rows) - tally['PENDING']}/{len(self.store.rows)}"
                          f"    ·    CUT {tally['CUT']}    KEEP {tally['KEEP']}    Còn trống {tally['PENDING']}")
        self.detail.set(f"Ảnh {self.index + 1}/{len(self.store.rows)} · rank {row['rank']}"
                        f" · ID {row['gt_id']} · tách từ frame MOT {row['cut_at_frame']}"
                        f" · Quyết định: {row['decision'] or 'CHƯA REVIEW'}")
        self.finish_button.configure(state="normal" if not self.busy and not tally["PENDING"] else "disabled")

    def show(self):
        row = self.store.rows[self.index]
        self.note.delete("1.0", "end")
        self.note.insert("1.0", row["note"])
        with Image.open(self.store.path.parent / row["sheet"]) as source:
            self.source_image = source.convert("RGB")
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.refresh()
        self.draw()

    def schedule_resize(self, event=None):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(100, self.draw)

    def draw(self):
        self.resize_job = None
        if self.source_image is None:
            return
        view = self.source_image.copy()
        if not self.full_size:
            view.thumbnail((max(100, self.canvas.winfo_width() - 8),
                            max(100, self.canvas.winfo_height() - 8)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(view)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, view.width, view.height))

    def toggle_size(self):
        self.full_size = not self.full_size
        self.draw()

    def preview(self):
        if self.busy:
            return
        try:
            JoinPlayer(self.root, self.args, self.store.rows[self.index])
        except (ValueError, OSError) as error:
            messagebox.showerror("Không thể xem đoạn nối", str(error), parent=self.root)

    def schedule_save(self, event=None):
        if self.autosave:
            self.root.after_cancel(self.autosave)
        self.autosave = self.root.after(700, self.save_note)

    def save_note(self, decision=None):
        if self.autosave:
            self.root.after_cancel(self.autosave)
            self.autosave = None
        if self.busy:
            return False
        row = self.store.rows[self.index]
        try:
            self.store.save(self.index, row["decision"] if decision is None else decision,
                            self.note.get("1.0", "end-1c"))
        except (ValueError, OSError) as error:
            self.status.set("Chưa lưu được. Giữ cửa sổ này để xử lý hoặc tải lại CSV.")
            messagebox.showerror("Chưa lưu", str(error), parent=self.root)
            return False
        self.status.set(f"Đã lưu ảnh rank {row['rank']} vào decisions.csv.")
        self.refresh()
        return True

    def select(self, index):
        if self.busy or not 0 <= index < len(self.store.rows) or not self.save_note():
            return
        self.index = index
        self.show()

    def pending(self):
        index = self.store.next_pending(self.index + 1)
        if index is not None:
            self.select(index)

    def verdict(self, decision):
        if self.busy or not self.save_note(decision):
            return
        index = self.store.next_pending(self.index + 1)
        self.index = self.index if index is None else index
        self.show()
        self.canvas.focus_set()

    def reload(self):
        if self.busy:
            return
        if self.note.get("1.0", "end-1c") != self.store.rows[self.index]["note"]:
            if not messagebox.askyesno("Tải lại CSV", "Bỏ ghi chú chưa lưu và đọc lại CSV?", parent=self.root):
                return
        if self.autosave:
            self.root.after_cancel(self.autosave)
            self.autosave = None
        try:
            replacement = self.load_store()
            self.check_store(replacement)
            self.store = replacement
            self.index = self.store.next_pending() or 0
            self.show()
            self.status.set("Đã đọc lại CSV. Tiếp tục từ ảnh chưa review.")
        except (ValueError, OSError) as error:
            messagebox.showerror("Không thể tải lại", str(error), parent=self.root)

    def finish(self):
        if self.busy or not self.save_note():
            return
        reviewer = self.reviewer.get().strip()
        if not reviewer:
            messagebox.showinfo("Tên người review", "Nhập tên người đã xác nhận CUT/KEEP trước khi hoàn tất.", parent=self.root)
            self.reviewer.focus_set()
            return
        self.busy = True
        for control in self.controls:
            control.configure(state="disabled")
        self.note.configure(state="disabled")
        self.reviewer.configure(state="disabled")
        self.refresh()
        self.status.set("Đang tạo GT mới và chấm lại các ablation…")

        def worker():
            try:
                result = self.run_finalize(reviewer, lambda text: self.messages.put((None, text)))
                self.messages.put((True, str(result)))
            except Exception as error:
                self.messages.put((False, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(150, self.poll)

    def poll(self):
        try:
            success, result = self.messages.get_nowait()
        except queue.Empty:
            self.root.after(150, self.poll)
            return
        if success is None:
            self.status.set(result)
            self.root.after(150, self.poll)
            return
        self.busy = False
        for control in self.controls:
            control.configure(state="normal")
        self.note.configure(state="normal")
        self.reviewer.configure(state="normal")
        self.refresh()
        if success:
            self.status.set(f"Đã hoàn tất. Kết quả: {result}")
            messagebox.showinfo("Hoàn tất review", self.completion_message(result), parent=self.root)
        else:
            self.status.set("Chưa hoàn tất. Quyết định đã lưu; có thể xử lý lỗi và thử lại.")
            messagebox.showerror("Không thể hoàn tất", result, parent=self.root)

    def close(self):
        if self.busy:
            messagebox.showinfo("Đang xử lý", "Đợi tạo GT và chấm điểm xong trước khi đóng.", parent=self.root)
            return
        if self.save_note():
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--gt", type=Path, default=GT_IN)
    parser.add_argument("--tracks-dir", type=Path, default=TRACKS_DIR)
    parser.add_argument("--frames-dir", type=Path,
                        default=REPO_ROOT / "data/annotations/prelabels/dense/img1")
    parser.add_argument("--segment", nargs=2, type=int, default=(10700, 11300))
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        ReviewApp(root, args)
    except (ValueError, OSError, KeyError) as error:
        root.withdraw()
        messagebox.showerror("Không thể mở GT review", str(error), parent=root)
        root.destroy()
        raise SystemExit(1) from error
    root.mainloop()


if __name__ == "__main__":
    main()
