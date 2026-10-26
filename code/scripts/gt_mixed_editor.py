"""Small local video annotator. Every rectangle/edit is explicitly saved by a human."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from code.scripts.gt_review_queue import CLASS_NAMES


class FrameEditor:
    def __init__(self, app, case):
        self.app, self.store, self.case = app, app.store, case
        self.low, self.high = case["window"]
        self.window = tk.Toplevel(app.root)
        segment=self.store.manifest.get('segment_name','Mixed')
        self.window.title(f"{segment} · {case['kind']} #{case['rank']} · video và chỉnh box")
        self.window.geometry("1120x870")
        self.window.minsize(850, 690)
        self.window.transient(app.root)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.frame = tk.IntVar(value=case.get('focus_frame',self.low))
        self.info = tk.StringVar()
        self.status = tk.StringVar(value="Click box để chọn xe. Vẽ/sửa: kéo khung bằng chuột; mỗi thay đổi tự lưu.")
        self.class_name = tk.StringVar(value="motorcycle")
        self.show_boxes = tk.BooleanVar(value=True)
        self.mode = "select"
        self.selected = case.get('focus_tid')
        self.marked = None
        self.seen = set()
        self.job = None
        self.photo = None
        self.start = None
        self.rectangle = None
        self.rows, _ = self.store.preview()
        self.events_at_open = len(self.store.journal["events"])
        self.zoom = tk.StringVar(value="Vừa cửa sổ")

        top = ttk.Frame(self.window, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text=case["question"], wraplength=1050).pack(anchor="w")
        ttk.Label(top, textvariable=self.info).pack(anchor="w")
        panel = ttk.Frame(self.window)
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(panel, background="#20252b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sx = ttk.Scrollbar(panel, orient="horizontal", command=self.canvas.xview)
        sy = ttk.Scrollbar(panel, orient="vertical", command=self.canvas.yview)
        sx.grid(row=1, column=0, sticky="ew")
        sy.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-int(e.delta / 120), "units"))

        bottom = ttk.Frame(self.window, padding=8)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.slider = tk.Scale(bottom, from_=self.low, to=self.high, variable=self.frame,
                               orient="horizontal", showvalue=True, command=lambda _: self.load_frame())
        self.slider.grid(row=0, column=0, sticky="ew")
        nav = ttk.Frame(bottom)
        nav.grid(row=1, column=0, sticky="w")
        ttk.Button(nav, text="← Frame", command=lambda: self.move(-1)).pack(side="left", padx=3)
        self.play_button = ttk.Button(nav, text="Phát / Dừng [Space]", command=self.toggle)
        self.play_button.pack(side="left", padx=3)
        ttk.Button(nav, text="Frame →", command=lambda: self.move(1)).pack(side="left", padx=3)
        if case.get('suggestions') or case.get('kind')=='REPAIR':
            ttk.Button(nav,text='Phát từ đầu đoạn',command=self.play_from_start).pack(side='left',padx=3)
        ttk.Label(nav, text="Tốc độ:").pack(side="left", padx=5)
        self.speed = ttk.Combobox(nav, state="readonly", width=5, values=(2, 5, 10, 25))
        self.speed.set(5)
        self.speed.pack(side="left")
        ttk.Checkbutton(nav, text="Hiện box", variable=self.show_boxes, command=self.render).pack(side="left", padx=8)
        zoom = ttk.Combobox(nav, state="readonly", textvariable=self.zoom, width=14,
                            values=("Vừa cửa sổ", "100%", "200%"))
        zoom.pack(side="left")
        zoom.bind("<<ComboboxSelected>>", lambda _: self.render())
        ttk.Label(nav, text="Class:").pack(side="left", padx=6)
        self.classes = ttk.Combobox(nav, state="readonly", textvariable=self.class_name,
                                    values=list(CLASS_NAMES.values()), width=12)
        self.classes.pack(side="left")
        operations = ttk.Frame(bottom)
        operations.grid(row=2, column=0, sticky="w", pady=6)
        for number, (label, command) in enumerate((
            ("Chọn box", lambda: self.set_mode("select")),
            ("Vẽ / sửa box", lambda: self.set_mode("draw")),
            ("Xe mới +", self.new_vehicle), ("Xóa box frame này", self.delete_box),
            ("Sao chép sang frame sau", self.copy_next),
            ("Đổi class cả ID đang xem", self.reclassify),
            ("Tách ID từ frame này", self.split),
            ("Đánh dấu xe A", self.mark), ("Nối A với xe đang chọn", self.merge),
            ("Hoàn tác sửa cuối", self.undo),
        )):
            ttk.Button(operations, text=label, command=command).grid(
                row=number // 5, column=number % 5, padx=3, pady=3, sticky="ew")
        if case.get('suggestions'):
            ttk.Button(operations,text='Gợi ý → xe đang chọn',command=lambda:self.use_suggestion(False)).grid(row=2,column=0,columnspan=2,sticky='ew',padx=3,pady=3)
            ttk.Button(operations,text='Gợi ý → xe mới',command=lambda:self.use_suggestion(True)).grid(row=2,column=2,columnspan=2,sticky='ew',padx=3,pady=3)
        finish = ttk.Frame(bottom)
        finish.grid(row=3, column=0, sticky="ew")
        self.allow_overlap = tk.BooleanVar(value=False)
        ttk.Checkbutton(finish, text="Nối box trùng của cùng xe (giữ box xe A)",
                        variable=self.allow_overlap).pack(side="left")
        ttk.Button(finish, text="Lưu và về review", command=self.close).pack(side="right")
        ttk.Label(bottom, textvariable=self.status, wraplength=1050).grid(row=4, column=0, sticky="w")
        self.window.bind("<space>", lambda e: self.key(e, self.toggle))
        self.window.bind("<Left>", lambda e: self.key(e, lambda: self.move(-1)))
        self.window.bind("<Right>", lambda e: self.key(e, lambda: self.move(1)))
        self.load_frame()

    def key(self, event, callback):
        if isinstance(self.window.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        callback()
        return "break"

    def load_frame(self):
        from pathlib import Path

        frame = self.frame.get()
        path = Path(self.store.manifest["frames_dir"]) / f"{frame:06d}.jpg"
        with Image.open(path) as source:
            self.picture = source.convert("RGB")
        self.seen.add(frame)
        self.render()

    def render(self):
        if not hasattr(self, "picture"):
            return
        picture = self.picture.copy()
        frame = self.frame.get()
        draw = ImageDraw.Draw(picture)
        from code.scripts.render_split_candidates import _font
        label_font = _font(11)
        self.visible = {key: fields for key, fields in self.rows.items() if key[0] == frame}
        from code.scripts.gt_roi import load_roi, contains_box
        active_roi = load_roi(self.store.bundle)
        if (self.store.bundle / 'scope_provenance.json').exists():
            self.visible = {key:p for key,p in self.visible.items() if contains_box(self.box(p),active_roi)}
        for key, fields in self.visible.items():
            if not self.show_boxes.get():
                continue
            box = self.box(fields)
            color = "#00ff70" if key[1] == self.selected else ("#ffd600" if key[1] in self.case["ids"] else "#5cceff")
            draw.rectangle(box, outline=color, width=3 if key[1] == self.selected else 1)
            draw.text((box[0], max(0, box[1] - 13)), f"{fields[1]} {CLASS_NAMES[int(fields[7]) - 1]}",
                      font=label_font, fill=color, stroke_width=1, stroke_fill="black")
        for suggestion in self.case.get('suggestions',[]):
            if suggestion['frame']==frame:
                b=suggestion['box'];draw.rectangle(b,outline='#ff7040',width=3)
                draw.text((b[0],max(0,b[1]-25)),'GOI Y / CHUA LA GT',font=label_font,fill='#ff7040',stroke_width=1,stroke_fill='black')
        from code.scripts.gt_roi import load_roi, overlay_roi
        picture = overlay_roi(picture, load_roi(self.store.bundle))
        if self.zoom.get() == "200%":
            self.scale = 2.0
        elif self.zoom.get() == "100%":
            self.scale = 1.0
        else:
            self.scale = min(max(100, self.canvas.winfo_width() - 6) / picture.width,
                             max(100, self.canvas.winfo_height() - 6) / picture.height)
        picture = picture.resize((max(1, round(picture.width * self.scale)),
                                  max(1, round(picture.height * self.scale))), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(picture)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, picture.width, picture.height))
        self.rectangle = None
        logged=self.store.seen_for(self.case['rank']) if hasattr(self.store,'seen_for') else set(self.store.journal['seen_frames'])
        shown = len(self.seen | (logged & set(range(self.low, self.high + 1))))
        selected_fields = self.visible.get((frame, self.selected))
        suffix = f" · đang chọn ID {selected_fields[1]} (nguồn {self.selected})" if selected_fields else ""
        self.info.set(f"MOT {frame} · CVAT {frame - 1} · video {self.store.manifest['segment'][0] + frame - 1}"
                      f" · Đã hiển thị {shown}/{self.high - self.low + 1} frame của đoạn này{suffix}")

    @staticmethod
    def box(fields):
        x, y, w, h = map(float, fields[2:6])
        return x, y, x + w, y + h

    def point(self, event):
        return (self.canvas.canvasx(event.x) / self.scale,
                self.canvas.canvasy(event.y) / self.scale)

    def press(self, event):
        self.stop()
        if self.mode == "draw":
            self.start = self.point(event)
            return
        x, y = self.point(event)
        options = []
        for key, fields in self.visible.items():
            x1, y1, x2, y2 = self.box(fields)
            if x1 <= x <= x2 and y1 <= y <= y2:
                options.append(((x2 - x1) * (y2 - y1), key, fields))
        if options:
            _, key, fields = min(options)
            self.selected = key[1]
            self.class_name.set(CLASS_NAMES[int(fields[7]) - 1])
            self.status.set(f"Đã chọn xe nguồn {self.selected}. Vẽ/sửa box hoặc chọn thao tác bên dưới.")
        else:
            self.selected = None
        self.render()

    def drag(self, event):
        if self.mode != "draw" or self.start is None:
            return
        x, y = self.point(event)
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(self.start[0] * self.scale, self.start[1] * self.scale,
                                                       x * self.scale, y * self.scale, outline="#00ff70", width=2)

    def release(self, event):
        if self.mode != "draw" or self.start is None:
            return
        x, y = self.point(event)
        width, height = self.store.manifest["image_size"]
        x1, x2 = sorted((max(0, min(width, self.start[0])), max(0, min(width, x))))
        y1, y2 = sorted((max(0, min(height, self.start[1])), max(0, min(height, y))))
        self.start = None
        if x2 - x1 < 2 or y2 - y1 < 2:
            self.status.set("Box quá nhỏ; kéo lại khung ít nhất 2 px mỗi chiều.")
            self.render()
            return
        if self.selected is None:
            self.selected = self.store.new_track_id()
        key = (self.frame.get(), self.selected)
        fields = list(self.visible.get(key, [key[0], key[1], 0, 0, 1, 1, 1, 2, 1]))
        fields[:2] = list(key)
        fields[2:6] = [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]
        fields[7] = list(CLASS_NAMES.values()).index(self.class_name.get()) + 1
        if self.edit({"op": "upsert", "fields": fields}):
            self.mode = "select"

    def edit(self, event):
        self.stop()
        try:
            event.update(case_rank=self.case["rank"], reviewer=self.app.reviewer.get().strip() or "Nghĩa")
            self.store.add_event(event)
            self.rows, _ = self.store.preview()
            self.status.set("Đã lưu thay đổi. Đóng cửa sổ rồi E để xác nhận đã sửa trường hợp này.")
            self.render()
            return True
        except (ValueError, OSError) as error:
            messagebox.showerror("Chưa áp dụng", str(error), parent=self.window)
            return False

    def set_mode(self, mode):
        self.stop()
        self.mode = mode
        self.status.set("Kéo khung quanh xe. Xe máy gồm cả người ngồi trên xe." if mode == "draw" else "Click box để chọn xe.")

    def new_vehicle(self):
        self.selected = self.store.new_track_id()
        self.set_mode("draw")
        self.status.set(f"Xe mới nguồn ID {self.selected}: chọn class rồi kéo khung. Dùng cùng ID khi vẽ các frame sau.")

    def use_suggestion(self,new_vehicle):
        suggestion=next((s for s in self.case.get('suggestions',[]) if s['frame']==self.frame.get()),None)
        if not suggestion:
            self.status.set('Frame này không có box gợi ý; vẽ thủ công nếu thấy xe.');return
        if new_vehicle:self.selected=self.store.new_track_id()
        elif self.selected is None:
            self.status.set('Chọn ID đúng trên frame có box trước; giữ ID đó khi chuyển tới frame cần thêm.');return
        elif any(int(p[7])!=2 for key,p in self.rows.items() if key[1]==self.selected):
            self.status.set('ID đang chọn không phải xe máy. Chọn đúng ID xe máy hoặc thêm xe mới.');return
        key=(self.frame.get(),self.selected)
        if key in self.rows and not messagebox.askyesno('Thay box hiện tại?',f'ID {self.selected} đã có box ở frame này. Thay bằng gợi ý?',parent=self.window):return
        x1,y1,x2,y2=suggestion['box']
        fields=[self.frame.get(),self.selected,x1,y1,x2-x1,y2-y1,1,2,1.0]
        self.edit(dict(op='upsert',fields=fields))

    def play_from_start(self):
        self.stop();self.frame.set(self.low);self.load_frame();self.step()

    def require_selected(self):
        if (self.frame.get(), self.selected) not in self.visible:
            self.status.set("Click một box đang có trong frame để chọn xe trước.")
            return False
        return True

    def delete_box(self):
        if self.require_selected():
            self.edit({"op": "delete_box", "frame": self.frame.get(), "tid": self.selected})

    def copy_next(self):
        if not self.require_selected() or self.frame.get() >= self.high:
            return
        frame = self.frame.get() + 1
        if (frame, self.selected) in self.rows:
            self.status.set("Frame sau đã có box cho xe này; chọn Vẽ/sửa nếu cần chỉnh.")
            return
        fields = list(self.visible[(self.frame.get(), self.selected)])
        fields[:2] = [frame, self.selected]
        if self.edit({"op": "upsert", "fields": fields}):
            self.frame.set(frame)
            self.load_frame()
            self.status.set("Đã sao chép box. Kiểm tra vị trí và kéo sửa cho khít xe ở frame mới.")

    def reclassify(self):
        if not self.require_selected():
            return
        canonical_id = self.visible[(self.frame.get(), self.selected)][1]
        keys = [list(key) for key, fields in self.rows.items() if fields[1] == canonical_id]
        self.edit({"op": "reclassify", "keys": keys,
                   "class_id": list(CLASS_NAMES.values()).index(self.class_name.get()) + 1})

    def split(self):
        if self.require_selected():
            self.edit({"op": "split", "tid": self.selected, "frame": self.frame.get()})

    def mark(self):
        if self.require_selected():
            self.marked = [self.selected, self.frame.get()]
            self.status.set(f"Đã đánh dấu xe A nguồn {self.selected}. Tìm xe B, click box rồi bấm Nối.")

    def merge(self):
        if self.marked is None:
            self.status.set("Chọn xe đầu tiên và bấm Đánh dấu xe A trước.")
            return
        if self.require_selected():
            self.edit({"op": "merge", "anchors": [self.marked, [self.selected, self.frame.get()]],
                       "allow_overlap": self.allow_overlap.get()})

    def undo(self):
        self.stop()
        try:
            self.store.undo()
            self.rows, _ = self.store.preview()
            self.render()
            self.status.set("Đã hoàn tác sửa cuối; quyết định cần sửa lại có thể trở về trống.")
        except (ValueError, OSError) as error:
            messagebox.showerror("Chưa hoàn tác", f"{error}\nĐưa quyết định phụ thuộc về U trước khi hoàn tác.", parent=self.window)

    def move(self, amount):
        self.stop()
        self.frame.set(max(self.low, min(self.high, self.frame.get() + amount)))
        self.load_frame()

    def stop(self):
        if self.job:
            self.window.after_cancel(self.job)
            self.job = None

    def toggle(self):
        if self.job:
            self.stop()
        else:
            if self.frame.get() == self.high:
                self.frame.set(self.low)
                self.load_frame()
            self.step()

    def step(self):
        self.job = None
        if self.frame.get() < self.high:
            self.frame.set(self.frame.get() + 1)
            self.load_frame()
            self.job = self.window.after(round(1000 / float(self.speed.get())), self.step)

    def close(self):
        self.stop()
        saved = True
        try:
            self.store.record_seen(self.seen)
        except (ValueError, OSError) as error:
            if not messagebox.askyesno("Chưa lưu lượt xem", f"{error}\n\n"
                                        "Đóng trình xem, bỏ lượt xem chưa lưu và tải lại CSV ở cửa sổ review?",
                                        parent=self.window):
                return
            saved = False
        self.window.grab_release()
        self.window.destroy()
        self.app.refresh()
        self.app.canvas.focus_set()
        self.app.status.set("Đã lưu lượt xem và mọi chỉnh sửa. A/K để duyệt hoặc E nếu đã sửa thủ công."
                            if saved else "Chưa lưu lượt xem. Bấm Tải lại CSV trước khi tiếp tục.")
