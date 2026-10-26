"""Local, resumable reviewer for the prepared accident/vehicle dataset."""
from __future__ import annotations

import argparse
import copy
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageDraw, ImageTk

from code.scripts.clean_detection_dataset import (
    BUNDLE, DOMAINS, META, CleanStore, export, prepare,
)

COLORS = ("#ff4545", "#ffbf00", "#36dd80", "#42baff", "#ea64ff")


def picture(store, path, overlay=True):
    row = store.rows[path]
    with Image.open(store.bundle / "pool/images" / row["name"]) as source:
        image = source.convert("RGB")
    if overlay:
        draw = ImageDraw.Draw(image)
        for cid, cx, cy, width, height in store.boxes(path):
            bounds = ((cx-width/2)*image.width, (cy-height/2)*image.height,
                      (cx+width/2)*image.width, (cy+height/2)*image.height)
            color = COLORS[int(cid) % len(COLORS)]
            draw.rectangle(bounds, outline=color, width=2)
            draw.text((bounds[0], bounds[1]), store.names[int(cid)], fill=color)
    return image


class BoxEditor:
    def __init__(self, app):
        self.app, self.store, self.path = app, app.store, app.current_path
        self.window = tk.Toplevel(app.root)
        self.window.title("Sửa bounding box — chỉ ảnh đang chọn")
        self.window.geometry("1100x830")
        self.window.transient(app.root); self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.columnconfigure(0, weight=1); self.window.rowconfigure(1, weight=1)
        self.mode, self.start, self.selected, self.scale = "select", None, None, 1.
        self.undo_stack = []
        self.classes = tk.StringVar(value=self.store.names[0])
        self.status = tk.StringVar(value="Chọn box bằng click; vẽ thêm/sửa bằng kéo chuột. Mỗi thao tác tự lưu.")
        self.zoom = tk.StringVar(value="Vừa cửa sổ")
        top = ttk.Frame(self.window, padding=8); top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text=self.path, wraplength=1050).pack(anchor="w")
        ttk.Label(top, textvariable=self.status).pack(anchor="w")
        panel = ttk.Frame(self.window); panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1); panel.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(panel, background="#20252b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sx = ttk.Scrollbar(panel, orient="horizontal", command=self.canvas.xview)
        sy = ttk.Scrollbar(panel, orient="vertical", command=self.canvas.yview)
        sx.grid(row=1, column=0, sticky="ew"); sy.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        bar = ttk.Frame(self.window, padding=8); bar.grid(row=2, column=0, sticky="ew")
        combo = ttk.Combobox(bar, textvariable=self.classes, values=self.store.names, state="readonly", width=12)
        combo.pack(side="left")
        for label, action in (("Chọn", lambda: self.set_mode("select")),
                              ("Thêm box", lambda: self.set_mode("add")),
                              ("Vẽ lại box", lambda: self.set_mode("replace")),
                              ("Đổi class", self.reclassify), ("Xóa box", self.delete),
                              ("Hoàn tác sửa cuối", self.undo), ("Lưu và quay lại", self.close)):
            ttk.Button(bar, text=label, command=action).pack(side="left", padx=2)
        zoom = ttk.Combobox(bar, textvariable=self.zoom, values=("Vừa cửa sổ", "100%", "200%"), state="readonly", width=13)
        zoom.pack(side="left"); zoom.bind("<<ComboboxSelected>>", lambda _: self.render())
        self.render()

    def set_mode(self, mode):
        self.mode = mode
        self.status.set({"select": "Click vào box để chọn.", "add": "Kéo chuột để thêm box với class đang chọn.",
                         "replace": "Chọn box trước, rồi kéo để vẽ lại."}[mode])

    def render(self):
        image = picture(self.store, self.path)
        if self.selected is not None and self.selected < len(self.store.boxes(self.path)):
            _, cx, cy, w, h = self.store.boxes(self.path)[self.selected]
            ImageDraw.Draw(image).rectangle(((cx-w/2)*image.width, (cy-h/2)*image.height,
                                             (cx+w/2)*image.width, (cy+h/2)*image.height), outline="white", width=4)
        if self.zoom.get() == "Vừa cửa sổ":
            self.scale = min(max(150, self.canvas.winfo_width()-8)/image.width,
                             max(150, self.canvas.winfo_height()-8)/image.height, 2.)
        else:
            self.scale = int(self.zoom.get()[:-1])/100
        image = image.resize((max(1, int(image.width*self.scale)), max(1, int(image.height*self.scale))))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all"); self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))

    def position(self, event):
        row = self.store.rows[self.path]
        return (min(1., max(0., self.canvas.canvasx(event.x)/self.scale/row["width"])),
                min(1., max(0., self.canvas.canvasy(event.y)/self.scale/row["height"])))

    def press(self, event):
        x, y = self.position(event)
        if self.mode == "select":
            hits = [(w*h, i) for i, (_, cx, cy, w, h) in enumerate(self.store.boxes(self.path))
                    if cx-w/2 <= x <= cx+w/2 and cy-h/2 <= y <= cy+h/2]
            self.selected = min(hits)[1] if hits else None
            if self.selected is not None:
                self.classes.set(self.store.names[int(self.store.boxes(self.path)[self.selected][0])])
            self.render()
        else:
            self.start = (x, y)

    def drag(self, event):
        if self.start:
            row = self.store.rows[self.path]; x, y = self.position(event)
            self.canvas.delete("drag")
            self.canvas.create_rectangle(self.start[0]*row["width"]*self.scale, self.start[1]*row["height"]*self.scale,
                                         x*row["width"]*self.scale, y*row["height"]*self.scale, outline="white", tags="drag")

    def release(self, event):
        if not self.start:
            return
        x, y = self.position(event); x1, x2 = sorted((self.start[0], x)); y1, y2 = sorted((self.start[1], y))
        self.start = None; self.canvas.delete("drag")
        if min(x2-x1, y2-y1) <= 0:
            return
        boxes = self.store.boxes(self.path)
        box = [self.store.names.index(self.classes.get()), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
        if self.mode == "replace":
            if self.selected is None:
                self.status.set("Chọn box trước khi vẽ lại."); return
            boxes[self.selected] = box
        else:
            boxes.append(box); self.selected = len(boxes)-1
        self.save(boxes)

    def save(self, boxes, remember=True):
        before = self.store.boxes(self.path)
        try:
            self.store.update_images([self.path], reviewer=self.app.reviewer.get(), boxes=boxes)
            if remember:
                self.undo_stack.append(before)
            self.status.set("Đã tự lưu nhãn sửa. Cụm liên quan cần xác nhận lại.")
            self.render()
            return True
        except (ValueError, OSError) as exc:
            messagebox.showerror("Chưa lưu", str(exc), parent=self.window)
            return False

    def delete(self):
        if self.selected is not None:
            boxes = self.store.boxes(self.path); boxes.pop(self.selected); self.selected = None
            self.save(boxes)

    def reclassify(self):
        if self.selected is not None:
            boxes = self.store.boxes(self.path); boxes[self.selected][0] = self.store.names.index(self.classes.get())
            self.save(boxes)

    def undo(self):
        if self.undo_stack and self.save(self.undo_stack[-1], remember=False):
            self.undo_stack.pop(); self.selected = None; self.render()

    def close(self):
        self.window.destroy(); self.app.show_case()


class CleanApp:
    def __init__(self, root, bundle=BUNDLE):
        self.root, self.store = root, CleanStore(bundle)
        self.case_index, self.member_index, self.busy = 0, 0, False
        self.messages = queue.Queue()
        root.title("CAAP — Duyệt dataset Accident / Vehicle")
        root.geometry("1320x960"); root.minsize(1050, 790)
        root.columnconfigure(1, weight=1); root.rowconfigure(1, weight=1)
        self.progress, self.detail, self.status = tk.StringVar(), tk.StringVar(), tk.StringVar(value="Mỗi quyết định/sửa nhãn tự lưu. Dataset gốc được giữ nguyên.")
        self.reviewer = tk.StringVar(value=self.store.state["reviewer"])
        self.overlay = tk.BooleanVar(value=True)
        self.metadata = {k: tk.BooleanVar() if k.endswith("_confirmed") else tk.StringVar() for k in META}
        self.all_members = tk.BooleanVar(value=False)
        top = ttk.Frame(root, padding=8); top.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(top, textvariable=self.progress, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.detail, wraplength=1250).pack(anchor="w")
        left = ttk.Frame(root, padding=6); left.grid(row=1, column=0, sticky="ns")
        ttk.Label(left, text="Cụm / batch cần duyệt").pack(anchor="w")
        self.cases_list = tk.Listbox(left, width=28, height=18, exportselection=False)
        self.cases_list.pack(fill="both", expand=True); self.cases_list.bind("<<ListboxSelect>>", self.choose_case)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.cases_list.yview)
        scrollbar.pack(side="right", fill="y"); self.cases_list.configure(yscrollcommand=scrollbar.set)
        ttk.Label(left, text="Ảnh trong cụm").pack(anchor="w")
        self.members_list = tk.Listbox(left, width=28, height=9, exportselection=False)
        self.members_list.pack(fill="both", expand=True); self.members_list.bind("<<ListboxSelect>>", self.choose_member)
        main = ttk.Frame(root); main.grid(row=1, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1); main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)
        self.canvases = []
        self.photos = [None, None]
        for column in range(2):
            canvas = tk.Canvas(main, background="#20252b", highlightthickness=0)
            canvas.grid(row=0, column=column, sticky="nsew", padx=3)
            canvas.bind("<Configure>", lambda _: self.draw())
            self.canvases.append(canvas)
        self.image_info = tk.StringVar()
        ttk.Label(main, textvariable=self.image_info, wraplength=1030).grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        footer = ttk.Frame(root, padding=8); footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        nav = ttk.Frame(footer); nav.pack(anchor="w")
        for label, action in (("← Ảnh", lambda: self.move_member(-1)), ("Ảnh →", lambda: self.move_member(1)),
                              ("Xác nhận cụm/batch [A]", self.accept), ("Chưa rõ [U]", self.pending_case),
                              ("Cụm chưa duyệt [N]", self.next_pending), ("Sửa box [V]", self.editor),
                              ("Giữ ảnh [K]", lambda: self.set_status("KEEP")),
                              ("Loại ảnh dư [D]", lambda: self.set_status("EXCLUDE")),
                              ("Cách ly ảnh [Q]", lambda: self.set_status("QUARANTINE")), ("Tải lại", self.reload)):
            ttk.Button(nav, text=label, command=action).pack(side="left", padx=2)
        ttk.Checkbutton(footer, text="Hiện bounding box", variable=self.overlay, command=self.draw).pack(anchor="w")
        meta = ttk.LabelFrame(footer, text="Nguồn đã xác minh — có thể để trống và tiếp tục duyệt ảnh", padding=5)
        meta.pack(fill="x")
        for i, key in enumerate(("source_id", "clip_id", "event_id", "camera_id", "domain")):
            ttk.Label(meta, text=key).grid(row=0, column=i*2, sticky="w", padx=2)
            widget = ttk.Combobox(meta, textvariable=self.metadata[key], values=DOMAINS, state="readonly", width=9) if key == "domain" else ttk.Entry(meta, textvariable=self.metadata[key], width=17)
            widget.grid(row=0, column=i*2+1, sticky="w")
        flags = ttk.Frame(meta); flags.grid(row=1, column=0, columnspan=10, sticky="w", pady=4)
        ttk.Checkbutton(flags, text="Tôi đã xác minh nguồn", variable=self.metadata["provenance_confirmed"]).pack(side="left")
        ttk.Checkbutton(flags, text="Ảnh gốc, chưa augmentation", variable=self.metadata["unaugmented_confirmed"]).pack(side="left", padx=8)
        ttk.Checkbutton(flags, text="Áp dụng thông tin này cho cả cụm", variable=self.all_members).pack(side="left", padx=8)
        ttk.Button(flags, text="Lưu thông tin nguồn", command=self.save_metadata).pack(side="left")
        ttk.Button(flags, text="Nhập CSV nguồn", command=self.import_provenance).pack(side="left", padx=5)
        notes = ttk.Frame(footer); notes.pack(fill="x", pady=4)
        ttk.Label(notes, text="Ghi chú cụm:").pack(side="left")
        self.note = ttk.Entry(notes, width=110); self.note.pack(side="left", fill="x", expand=True)
        self.note.bind("<FocusOut>", lambda _: self.save_note())
        final = ttk.Frame(footer); final.pack(fill="x")
        ttk.Label(final, text="Người duyệt:").pack(side="left")
        ttk.Entry(final, textvariable=self.reviewer, width=18).pack(side="left", padx=5)
        self.training_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(final, text="Chỉ xuất bộ train (không có evaluation)", variable=self.training_only).pack(side="left", padx=8)
        self.finish_button = ttk.Button(final, text="Hoàn tất → xuất dataset", command=self.finish)
        self.finish_button.pack(side="left")
        ttk.Label(footer, textvariable=self.status, wraplength=1250).pack(anchor="w", pady=4)
        for key, action in {"a": self.accept, "u": self.pending_case, "n": self.next_pending, "v": self.editor,
                            "k": lambda: self.set_status("KEEP"), "d": lambda: self.set_status("EXCLUDE"),
                            "q": lambda: self.set_status("QUARANTINE"), "Left": lambda: self.move_member(-1),
                            "Right": lambda: self.move_member(1)}.items():
            root.bind(f"<KeyPress-{key}>", lambda event, fn=action: self.shortcut(event, fn))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-s>", lambda _: self.save_all())
        self.next_pending(initial=True)

    @property
    def case(self):
        return self.store.cases[self.case_index]

    @property
    def current_path(self):
        return self.case["members"][self.member_index]

    def shortcut(self, event, callback):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox)) or self.busy:
            return
        callback(); return "break"

    def refresh(self):
        tally = self.store.tally
        self.progress.set(f"Đã duyệt {tally['cases']-tally['pending']}/{tally['cases']} cụm/batch · Giữ {tally.get('KEEP',0)} · Loại {tally.get('EXCLUDE',0)} · Cách ly {tally.get('QUARANTINE',0)} · Nguồn/ảnh gốc đã xác minh {tally['metadata_confirmed']}")
        self.detail.set(f"#{self.case['rank']} · {self.case['kind']} · {len(self.case['members'])} ảnh · {self.case['question']}")
        self.finish_button.configure(state="normal" if not self.busy and not tally["pending"] else "disabled")
        self.cases_list.delete(0, "end")
        for case in self.store.cases:
            mark = "✓" if self.store.state["cases"][str(case["rank"])]["reviewed"] else "·"
            self.cases_list.insert("end", f"{mark} {case['rank']:03} {case['kind']} {len(case['members'])} {'x-split' if case['cross_split'] else ''}")
        self.cases_list.selection_set(self.case_index); self.cases_list.see(self.case_index)

    def show_case(self):
        self.member_index = min(self.member_index, len(self.case["members"])-1)
        self.members_list.delete(0, "end")
        for i, path in enumerate(self.case["members"]):
            self.members_list.insert("end", f"{i+1:03} {self.store.image(path)['status']} {Path(path).name[:20]}")
        self.note.delete(0, "end"); self.note.insert(0, self.store.state["cases"][str(self.case["rank"])]["note"])
        self.all_members.set(False)
        self.show_member(); self.refresh()

    def show_member(self):
        self.members_list.selection_clear(0, "end"); self.members_list.selection_set(self.member_index); self.members_list.see(self.member_index)
        meta = self.store.image(self.current_path)["metadata"]
        for key, variable in self.metadata.items():
            variable.set(meta[key])
        row = self.store.rows[self.current_path]
        self.image_info.set(f"Đang chọn: {self.current_path}\n{self.store.image(self.current_path)['status']} · {', '.join(row['flags']) or 'Không có cờ chất lượng'} · Camera gợi ý (chưa xác minh): {row['camera_hint'] or '?'}")
        self.draw()

    def draw(self):
        if self.busy or not self.store.cases:
            return
        paths = self.case["members"]
        reference = next((p for p in paths if self.case["kind"] == "RELATED" and
                          self.store.rows[p]["original_split"] != self.store.rows[self.current_path]["original_split"]), None)
        reference = reference or (paths[1] if self.member_index == 0 and len(paths)>1 else paths[0])
        try:
            for col, path in enumerate((reference, self.current_path)):
                image = picture(self.store, path, self.overlay.get())
                canvas = self.canvases[col]
                image.thumbnail((max(160, canvas.winfo_width()-8), max(160, canvas.winfo_height()-36)))
                self.photos[col] = ImageTk.PhotoImage(image)
                canvas.delete("all"); canvas.create_image(4, 30, image=self.photos[col], anchor="nw")
                canvas.create_text(5, 5, anchor="nw", text=("Đối chiếu: " if col == 0 else "Ảnh đang chọn: ")+Path(path).name[:50], fill="white")
                if self.root.winfo_viewable():
                    self.store.seen(self.case["rank"], path, reviewer=self.reviewer.get())
        except (ValueError, OSError) as exc:
            self.status.set(f"Chưa lưu lượt xem: {exc}. Tải lại trước khi tiếp tục.")

    def save_note(self):
        if self.busy:
            return False
        record = self.store.state["cases"][str(self.case["rank"])]
        if record["note"] == self.note.get():
            return True
        return self.run(lambda: self.store.review_case(self.case["rank"], record["reviewed"], self.note.get(), self.reviewer.get()))

    def run(self, action):
        if self.busy:
            return False
        try:
            action(); self.status.set("Đã lưu trên máy."); self.refresh(); return True
        except (ValueError, OSError) as exc:
            messagebox.showerror("Chưa hoàn tất", str(exc), parent=self.root); return False

    def choose_case(self, event=None):
        selected = self.cases_list.curselection()
        if not self.busy and selected and selected[0] != self.case_index and self.save_note():
            self.case_index, self.member_index = selected[0], 0; self.show_case()

    def choose_member(self, event=None):
        selected = self.members_list.curselection()
        if not self.busy and selected:
            self.member_index = selected[0]; self.show_member()

    def move_member(self, step):
        if not self.busy and 0 <= self.member_index + step < len(self.case["members"]):
            self.member_index += step; self.show_member()

    def accept(self):
        if self.run(lambda: self.store.review_case(self.case["rank"], True, self.note.get(), self.reviewer.get())):
            self.next_pending()

    def pending_case(self):
        self.run(lambda: self.store.review_case(self.case["rank"], False, self.note.get(), self.reviewer.get()))

    def next_pending(self, initial=False):
        if self.busy or (not initial and not self.save_note()):
            return
        order = list(range(self.case_index + (0 if initial else 1), len(self.store.cases))) + list(range(0, self.case_index+1))
        self.case_index = next((i for i in order if not self.store.state["cases"][str(self.store.cases[i]["rank"])]["reviewed"]), self.case_index)
        self.member_index = 0; self.show_case()

    def set_status(self, status):
        reason = "" if status == "KEEP" else simpledialog.askstring("Lý do cho ảnh đang chọn", "Ghi lý do (ví dụ: cùng ảnh nguồn, bản augmentation dư; nhãn chưa xác minh).", parent=self.root)
        if reason is None:
            return
        if self.run(lambda: self.store.update_images([self.current_path], reviewer=self.reviewer.get(), status=status, reason=reason)):
            self.show_case()

    def save_metadata(self):
        paths = self.case["members"] if self.all_members.get() else [self.current_path]
        metadata = {key: variable.get() for key, variable in self.metadata.items()}
        if self.run(lambda: self.store.update_images(paths, reviewer=self.reviewer.get(), metadata=metadata)):
            self.show_case()

    def save_all(self):
        if self.save_note():
            self.save_metadata()

    def editor(self):
        if not self.busy and self.save_note():
            BoxEditor(self)

    def import_provenance(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, initialdir=self.store.bundle,
                                         title="Nhập provenance_template.csv đã điền", filetypes=[("CSV", "*.csv")])
        if path and self.run(lambda: self.store.import_provenance(Path(path), self.reviewer.get())):
            self.show_case()

    def reload(self):
        if self.busy:
            return
        try:
            replacement = CleanStore(self.store.bundle)
            self.store = replacement; self.member_index = 0; self.show_case()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Không thể tải lại", str(exc), parent=self.root)

    def finish(self):
        if not self.save_note():
            return
        if self.store.tally["pending"]:
            self.status.set("Còn cụm/batch chưa duyệt."); return
        self.busy = True; self.refresh(); self.status.set("Đang kiểm tra nguồn và xuất phiên bản mới…")
        training_only, reviewer = self.training_only.get(), self.reviewer.get()
        def worker():
            try:
                result = export(self.store, reviewer, training_only, progress=None)
                self.messages.put((True, str(result)))
            except Exception as exc:
                self.messages.put((False, str(exc)))
        threading.Thread(target=worker, daemon=True).start(); self.root.after(150, self.poll)

    def poll(self):
        try:
            success, result = self.messages.get_nowait()
        except queue.Empty:
            self.root.after(150, self.poll); return
        self.busy = False; self.refresh(); self.status.set(result)
        if success:
            messagebox.showinfo("Dataset đã xuất", result, parent=self.root)
        else:
            messagebox.showerror("Chưa thể xuất dataset", result, parent=self.root)

    def close(self):
        if self.busy:
            self.status.set("Chờ xuất phiên bản hoàn tất trước khi đóng."); return
        if self.save_note():
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    if not args.bundle.exists():
        prepare(bundle=args.bundle)
    root = tk.Tk()
    try:
        CleanApp(root, args.bundle)
    except (ValueError, OSError) as exc:
        root.withdraw(); messagebox.showerror("Không thể mở review", str(exc), parent=root)
        root.destroy(); raise SystemExit(1) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
