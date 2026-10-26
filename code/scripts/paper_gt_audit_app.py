"""One question at a time, autosaved; tracker and segment names stay hidden."""
import argparse
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import Image,ImageTk
from code.scripts.paper_gt_audit import AuditStore,active,prepare,HOME
from code.scripts.gt_review import sha256
from code.scripts.complete_paper_gt_audit import validate,complete,saved_source_ref

class AuditApp:
    def __init__(self,root,bundle):
        self.root=root; self.store=AuditStore(bundle);validate(self.store,saved_source_ref(bundle))
        self.index=next((i for i,r in enumerate(self.store.rows) if not r['verdict']),0)
        root.title('CAAP — Kiểm tra chéo GT ROI · Nghĩa'); root.geometry('1150x850')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.progress=tk.StringVar();self.question=tk.StringVar();self.status=tk.StringVar()
        ttk.Label(root,textvariable=self.progress,font=('Segoe UI',13,'bold')).pack(anchor='w',padx=12,pady=8)
        ttk.Label(root,textvariable=self.question,wraplength=1100,font=('Segoe UI',11)).pack(anchor='w',padx=12)
        ttk.Label(root,text='Đỏ: box cần đánh giá. Xanh: GT hiện tại. Tên tracker được ẩn.').pack(anchor='w',padx=12,pady=4)
        self.full=tk.BooleanVar(value=False)
        ttk.Checkbutton(root,text='Xem toàn khung để có ngữ cảnh',variable=self.full,command=self.render).pack(anchor='w',padx=12)
        self.canvas=tk.Canvas(root,background='#181818',height=480);self.canvas.pack(fill='both',expand=True,padx=10,pady=8)
        self.canvas.bind('<Configure>',lambda e:self.render())
        row=ttk.Frame(root);row.pack(fill='x',padx=12)
        self.buttons=[]
        for label,value in [('Có xe, box hợp lý [Y]','Y'),('Không có xe / box sai [N]','N'),('Xe đã có box xanh [D]','D'),('Chưa rõ [?]','?')]:
            b=ttk.Button(row,text=label,command=lambda v=value:self.verdict(v));b.pack(side='left',padx=3);self.buttons.append(b)
        ttk.Label(root,text='Ghi chú (tự lưu):').pack(anchor='w',padx=12,pady=(8,0))
        self.note=tk.Text(root,height=2);self.note.pack(fill='x',padx=12)
        self.timer=None;self.note.bind('<KeyRelease>',self.schedule)
        controls=ttk.Frame(root);controls.pack(fill='x',padx=12,pady=8)
        ttk.Button(controls,text='← Trước',command=lambda:self.select(self.index-1)).pack(side='left',padx=3)
        ttk.Button(controls,text='Sau →',command=lambda:self.select(self.index+1)).pack(side='left',padx=3)
        ttk.Button(controls,text='Bỏ quyết định',command=lambda:self.verdict('',advance=False)).pack(side='left',padx=3)
        ttk.Button(controls,text='Tải lại bản lưu',command=self.reload).pack(side='left',padx=3)
        ttk.Button(controls,text='Hoàn tất → tổng hợp điểm cần xử lý',command=self.finish).pack(side='left',padx=12)
        ttk.Label(root,textvariable=self.status,wraplength=1100).pack(anchor='w',padx=12,pady=5)
        for k,v in [('y','Y'),('n','N'),('d','D'),('question','?')]: root.bind(f'<KeyPress-{k}>',lambda e,value=v:self.shortcut(e,value))
        self.show()
    def shortcut(self,event,value):
        if event.widget is not self.note and (value!='D' or self.store.cases[self.index]['type']=='box'): self.verdict(value)
    def schedule(self,event=None):
        if self.timer is not None:self.root.after_cancel(self.timer)
        self.timer=self.root.after(500,self.autosave)
    def autosave(self):
        self.timer=None;self.save()
    def save(self,value=None):
        if self.timer is not None:self.root.after_cancel(self.timer);self.timer=None
        row=self.store.rows[self.index];note=self.note.get('1.0','end-1c');verdict=row['verdict'] if value is None else value
        if note==row['note'] and verdict==row['verdict']: return True
        try:self.store.save(self.index,verdict,note);self.status.set('Đã lưu.');return True
        except (ValueError,OSError) as e:messagebox.showerror('Chưa lưu',str(e),parent=self.root);return False
    def verdict(self,value,advance=True):
        if not self.save(value):return
        if advance:
            self.index=next((i for i in range(self.index+1,len(self.store.rows)) if not self.store.rows[i]['verdict']),self.index)
        self.show()
    def select(self,index):
        if 0<=index<len(self.store.rows) and self.save():self.index=index;self.show()
    def show(self):
        count=sum(bool(r['verdict']) for r in self.store.rows);case=self.store.cases[self.index];row=self.store.rows[self.index]
        self.progress.set(f"Đã lưu {count}/{len(self.store.rows)} · Mục #{case['n']} · {row['verdict'] or 'CHƯA DUYỆT'}")
        identity=case['type']=='identity'
        self.question.set('Hai box đỏ trước/sau có phải CÙNG MỘT XE? Y: cùng xe. N: hai xe khác nhau. ?: chưa rõ.' if identity else 'Có xe máy thật trong box đỏ, và box bám hợp lý vào xe không?')
        self.buttons[0].configure(text='Cùng một xe [Y]' if identity else 'Có xe, box hợp lý [Y]')
        self.buttons[1].configure(text='Hai xe khác nhau [N]' if identity else 'Không có xe / box sai [N]')
        self.buttons[2].configure(state='disabled' if identity else 'normal')
        self.note.delete('1.0','end');self.note.insert('1.0',row['note']);self.render()
    def render(self):
        case=self.store.cases[self.index];i=int(self.full.get());p=self.store.bundle/case['images'][i]
        try:
            if p.resolve().parent!=self.store.bundle.resolve() or sha256(p)!=case['image_sha256'][i]:raise ValueError('Ảnh audit đã đổi')
            with Image.open(p) as source:im=source.convert('RGB')
            w=max(100,self.canvas.winfo_width());h=max(100,self.canvas.winfo_height());scale=min(w/im.width,h/im.height)
            im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))))
            self.photo=ImageTk.PhotoImage(im);self.canvas.delete('all');self.canvas.create_image(w/2,h/2,image=self.photo)
        except (ValueError,OSError) as e:self.status.set(str(e))
    def reload(self):
        if messagebox.askyesno('Tải lại','Tải lại CSV đã lưu? Nội dung đang gõ chưa lưu sẽ bị bỏ.',parent=self.root):
            if self.timer is not None:self.root.after_cancel(self.timer);self.timer=None
            self.store=AuditStore(self.store.bundle);validate(self.store,saved_source_ref(self.store.bundle));self.show()
    def finish(self):
        if not self.save():return
        try:
            result=complete(self.store.bundle,saved_source_ref(self.store.bundle))
            messagebox.showinfo('Đã tổng hợp',f"Đã lưu audit_result.json.\n{len(result['issues'])} mục cần xác minh/sửa GT hoặc chưa rõ.\nGT chưa được thay đổi. Test độc lập chưa được đăng ký.",parent=self.root)
        except (ValueError,OSError) as e:messagebox.showerror('Chưa hoàn tất',str(e),parent=self.root)
    def close(self):
        if self.save():self.root.destroy()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle');args=p.parse_args()
    bundle=args.bundle or (active() if (HOME/'active.json').exists() else prepare())
    root=tk.Tk()
    try:AuditApp(root,bundle)
    except (ValueError,OSError) as e:
        root.withdraw();messagebox.showerror('Không mở được audit',str(e),parent=root);root.destroy();return
    root.mainloop()

if __name__=='__main__':main()
