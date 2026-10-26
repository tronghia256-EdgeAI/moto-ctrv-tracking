"""Interactive polygon ROI editor for the existing mixed GT review bundle."""
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from PIL import Image,ImageDraw,ImageTk
from code.scripts.gt_roi import load_roi,save_roi,validate_polygon,overlay_roi,contains_box,RULE
from code.scripts.gt_review import sha256


class ROIEditor:
    def __init__(self, root, bundle, reviewer='Nghĩa'):
        from code.scripts.gt_mixed_review import MixedStore
        self.root,self.bundle = root,Path(bundle)
        self.store = MixedStore(self.bundle)
        self.rows,_ = self.store.preview()
        self.roi = load_roi(self.bundle)
        self.fingerprint = sha256(self.bundle/'roi.json') if self.roi else None
        self.width,self.height = self.store.manifest['image_size']
        self.points = [list(p) for p in self.roi['polygon']] if self.roi else []
        self.saved = [list(p) for p in self.points]
        self.dragging = None
        self.frame = tk.IntVar(value=self.roi['reference_frame'] if self.roi else 359)
        self.frame.set(min(self.store.manifest['n_frames'],self.frame.get()))
        self.enabled = tk.BooleanVar(value=self.roi['enabled'] if self.roi else True)
        self.reviewer = tk.StringVar(value=self.roi['reviewer'] if self.roi else reviewer)
        self.reason = tk.StringVar(value=self.roi['reason'] if self.roi else '')
        self.status = tk.StringVar()
        self.dirty = False
        root.title('CAAP — Điều chỉnh ROI cho GT mixed')
        root.geometry('1150x820')
        ttk.Label(root,text='Click để thêm đỉnh theo vòng quanh vùng cần chấm; kéo đỉnh để sửa. Vùng tối nằm ngoài ROI.').pack(anchor='w',padx=10,pady=5)
        ttk.Label(root,text='ROI theo khả năng gán nhãn/đường quan sát; chốt trước khi xem metric. Lọc GT và dự đoán bằng tâm đáy box.').pack(anchor='w',padx=10)
        toolbar = ttk.Frame(root);toolbar.pack(fill='x',padx=10,pady=5)
        ttk.Button(toolbar,text='Bỏ đỉnh cuối',command=self.undo).pack(side='left')
        ttk.Button(toolbar,text='Vẽ lại',command=self.clear).pack(side='left')
        ttk.Button(toolbar,text='Toàn khung',command=self.full).pack(side='left')
        ttk.Checkbutton(toolbar,text='Áp dụng ROI khi xuất/chấm',variable=self.enabled,command=self.changed).pack(side='left',padx=12)
        ttk.Label(toolbar,text='Frame MOT:').pack(side='left')
        ttk.Spinbox(toolbar,from_=1,to=self.store.manifest['n_frames'],textvariable=self.frame,width=6,command=self.render).pack(side='left')
        ttk.Button(toolbar,text='Xem frame',command=self.render).pack(side='left')
        self.canvas = tk.Canvas(root,bg='#20252b');self.canvas.pack(fill='both',expand=True,padx=10)
        self.canvas.bind('<Configure>',lambda e:self.render())
        self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'dragging',None))
        ttk.Label(root,textvariable=self.status).pack(anchor='w',padx=10)
        fields = ttk.Frame(root);fields.pack(fill='x',padx=10)
        ttk.Label(fields,text='Người duyệt:').pack(side='left')
        ttk.Entry(fields,textvariable=self.reviewer,width=18).pack(side='left')
        ttk.Label(fields,text='Lý do phạm vi:').pack(side='left',padx=8)
        ttk.Entry(fields,textvariable=self.reason).pack(side='left',fill='x',expand=True)
        buttons=ttk.Frame(root);buttons.pack(fill='x',padx=10,pady=8)
        ttk.Button(buttons,text='Lưu ROI',command=self.save).pack(side='right')
        ttk.Button(buttons,text='Đóng',command=self.close).pack(side='right',padx=8)
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.scale=1.;self.offset=(0,0)
        self.reason.trace_add('write',lambda *args:setattr(self,'dirty',True))
        self.reviewer.trace_add('write',lambda *args:setattr(self,'dirty',True))
        root.after(100,self.render)

    def current(self):
        return dict(enabled=self.enabled.get(),polygon=self.points,rule=RULE)

    def changed(self):
        self.dirty=True;self.render()

    def undo(self):
        if self.points:self.points.pop();self.changed()

    def clear(self):
        self.points=[];self.changed()

    def full(self):
        self.points=[[0,0],[self.width,0],[self.width,self.height],[0,self.height]];self.changed()

    def coord(self,e):
        return [max(0,min(self.width,(e.x-self.offset[0])/self.scale)),max(0,min(self.height,(e.y-self.offset[1])/self.scale))]

    def press(self,e):
        p=self.coord(e)
        nearest=next((i for i,q in enumerate(self.points) if ((p[0]-q[0])**2+(p[1]-q[1])**2)**.5*self.scale <= 12),None)
        if nearest is None:self.points.append(p)
        else:self.dragging=nearest
        self.changed()

    def drag(self,e):
        if self.dragging is not None:self.points[self.dragging]=self.coord(e);self.changed()

    def render(self):
        try:
            f=self.frame.get()
            if not 1<=f<=self.store.manifest['n_frames']:return
            with Image.open(Path(self.store.manifest['frames_dir'])/f'{f:06d}.jpg') as im:picture=im.convert('RGB')
            roi=self.current() if len(self.points)>=3 else None
            valid=True
            if roi:
                try:validate_polygon(self.points,[self.width,self.height])
                except ValueError:valid=False;roi=None
            if roi:picture=overlay_roi(picture,roi)
            draw=ImageDraw.Draw(picture);inside=outside=0
            for (frame,tid),p in self.rows.items():
                if frame!=f:continue
                x,y,w,h=map(float,p[2:6]);box=[x,y,x+w,y+h]
                keep=contains_box(box,roi)
                inside+=keep;outside+=not keep
                draw.rectangle(box,outline='#00ff70' if keep else '#777777',width=1)
            if self.points:
                pts=[tuple(p) for p in self.points]
                draw.line(pts+(pts[:1] if len(pts)>2 else []),fill='#ffd600',width=2)
                for i,(x,y) in enumerate(pts):
                    r=4;draw.ellipse((x-r,y-r,x+r,y+r),fill='#ffd600');draw.text((x+6,y),str(i+1),fill='yellow')
            cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
            self.scale=min(cw/self.width,ch/self.height)
            size=(max(1,int(self.width*self.scale)),max(1,int(self.height*self.scale)))
            self.offset=((cw-size[0])/2,(ch-size[1])/2)
            self.photo=ImageTk.PhotoImage(picture.resize(size))
            self.canvas.delete('all');self.canvas.create_image(*self.offset,image=self.photo,anchor='nw')
            self.status.set(f'MOT {f} | {len(self.points)} đỉnh | Trong vùng: {inside}, ngoài: {outside} | '+('Chưa lưu' if self.dirty else 'Đã lưu' if self.roi else 'Chưa có ROI')+(' | Polygon tự cắt: vẽ/sửa lại' if not valid else ''))
        except (OSError,tk.TclError) as error:self.status.set(str(error))

    def save(self):
        try:
            self.roi=save_roi(self.bundle,self.points,self.enabled.get(),self.reviewer.get(),self.reason.get(),self.frame.get(),self.fingerprint)
            self.fingerprint=sha256(self.bundle/'roi.json');self.dirty=False;self.render()
            self.status.set('Đã lưu ROI riêng; GT gốc và các quyết định review được giữ nguyên. Đóng/mở lại app GT cũ trước khi xuất.')
        except (ValueError,OSError) as error:messagebox.showerror('Chưa lưu ROI',str(error),parent=self.root)

    def close(self):
        if self.dirty:
            answer=messagebox.askyesnocancel('ROI chưa lưu','Lưu ROI trước khi đóng?',parent=self.root)
            if answer is None:return
            if answer:
                self.save()
                if self.dirty:return
        self.root.destroy()


def main():
    from code.scripts.gt_mixed_review import BUNDLE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=BUNDLE)
    args=parser.parse_args();root=tk.Tk();ROIEditor(root,args.bundle);root.mainloop()


if __name__=='__main__':main()
