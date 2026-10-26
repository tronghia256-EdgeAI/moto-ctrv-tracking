"""A single window for the 32 targeted GT corrections across three segments."""
import argparse,json,queue,threading,tkinter as tk
from pathlib import Path
from tkinter import ttk,messagebox
from PIL import Image,ImageTk
from code.scripts.gt_audit_reconcile import ReconcileStore,active_collection,prepare,finalize_collection,HOME

class ReconcileApp:
    def __init__(self,root,path):
        self.root=root;self.path=Path(path);data=json.loads(self.path.read_text(encoding='utf-8'))
        self.stores={name:ReconcileStore(bundle) for name,bundle in data['bundles'].items()}
        for store in self.stores.values():store.check_inputs(images=True)
        self.entries=[(name,i) for name,store in self.stores.items() for i in range(len(store.rows))]
        self.index=next((i for i,(name,j) in enumerate(self.entries) if not self.stores[name].rows[j]['decision']),0)
        self.busy=False;self.messages=queue.Queue();root.title(f'CAAP — Sửa GT sau audit · {len(self.entries)} vị trí · Nghĩa');root.geometry('1180x890')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.progress=tk.StringVar();self.detail=tk.StringVar();self.status=tk.StringVar(value='Box cam là gợi ý. V để sửa; A/E xác nhận sau khi xem đoạn. K cần lý do giữ GT.')
        ttk.Label(root,textvariable=self.progress,font=('Segoe UI',13,'bold')).pack(anchor='w',padx=12,pady=8)
        ttk.Label(root,textvariable=self.detail,wraplength=1130).pack(anchor='w',padx=12)
        ttk.Label(root,text='Ảnh dưới là bằng chứng audit trước sửa. Mở V để xem và chỉnh GT hiện tại.').pack(anchor='w',padx=12)
        self.canvas=tk.Canvas(root,background='#181818',height=490);self.canvas.pack(expand=True,fill='both',padx=12,pady=8)
        self.canvas.bind('<Configure>',lambda e:self.render())
        ttk.Label(root,text='Lý do / ghi chú (giữ GT sau audit phải ghi lý do):').pack(anchor='w',padx=12)
        self.note=tk.Text(root,height=2);self.note.pack(fill='x',padx=12)
        controls=ttk.Frame(root);controls.pack(fill='x',padx=12,pady=8);self.controls=[]
        for label,command in [('← Trước',lambda:self.select(self.index-1)),('Xem / sửa video [V]',self.preview),('Áp dụng đề xuất tách [A]',self.accept),('Đã sửa [E]',lambda:self.verdict('EDIT')),('Giữ GT + lý do [K]',lambda:self.verdict('KEEP')),('Sau →',lambda:self.select(self.index+1))]:
            b=ttk.Button(controls,text=label,command=command);b.pack(side='left',padx=3);self.controls.append(b)
        final=ttk.Frame(root);final.pack(fill='x',padx=12,pady=6);ttk.Label(final,text='Người review:').pack(side='left')
        self.reviewer=ttk.Entry(final,width=20);self.reviewer.insert(0,'Nghĩa');self.reviewer.pack(side='left',padx=6)
        self.finish_button=ttk.Button(final,text='Hoàn tất → xuất GT mới và chấm lại',command=self.finish);self.finish_button.pack(side='left',padx=12)
        reset=ttk.Button(final,text='Đặt lại mục này',command=self.reset);reset.pack(side='left',padx=6);self.controls.append(reset)
        ttk.Label(root,textvariable=self.status,wraplength=1130).pack(anchor='w',padx=12,pady=5)
        for key,callback in [('v',self.preview),('a',self.accept),('e',lambda:self.verdict('EDIT')),('k',lambda:self.verdict('KEEP'))]:root.bind(f'<KeyPress-{key}>',lambda e,c=callback:self.shortcut(e,c))
        self.timer=None;self.note.bind('<KeyRelease>',self.schedule);self.show()
    def shortcut(self,event,callback):
        if event.widget not in (self.note,self.reviewer):callback()
    def schedule(self,event=None):
        if self.timer is not None:self.root.after_cancel(self.timer)
        self.timer=self.root.after(500,self.autosave)
    def autosave(self):self.timer=None;self.save_note()
    def save_note(self):
        if self.timer is not None:self.root.after_cancel(self.timer);self.timer=None
        if not hasattr(self,'store'):return True
        row=self.store.rows[self.local_index];note=self.note.get('1.0','end-1c')
        if note==row['note']:return True
        try:self.store.save(self.local_index,row['decision'],note);return True
        except (ValueError,OSError) as e:messagebox.showerror('Chưa lưu',str(e),parent=self.root);return False
    def show(self):
        self.segment,self.local_index=self.entries[self.index];self.store=self.stores[self.segment]
        case=self.store.manifest['cases'][self.local_index];row=self.store.rows[self.local_index]
        self.note.delete('1.0','end');self.note.insert('1.0',row['note']);self.refresh();self.render()
    def refresh(self):
        done=sum(bool(r['decision']) for s in self.stores.values() for r in s.rows)
        case=self.store.manifest['cases'][self.local_index];row=self.store.rows[self.local_index]
        self.progress.set(f"Đã xử lý {done}/{len(self.entries)} · {self.segment.upper()} · Mục {self.index+1} · {row['decision'] or 'CHƯA SỬA'}")
        self.detail.set(case['question']);self.controls[2].configure(state='normal' if case['proposal'] and not self.busy else 'disabled')
        self.finish_button.configure(state='normal' if done==len(self.entries) and not self.busy else 'disabled')
    def render(self):
        if not hasattr(self,'store'):return
        case=self.store.manifest['cases'][self.local_index]
        with Image.open(self.store.bundle/case['sheet']) as im:pic=im.convert('RGB')
        w=max(100,self.canvas.winfo_width());h=max(100,self.canvas.winfo_height());scale=min(w/pic.width,h/pic.height)
        pic=pic.resize((max(1,int(pic.width*scale)),max(1,int(pic.height*scale))))
        self.photo=ImageTk.PhotoImage(pic);self.canvas.delete('all');self.canvas.create_image(w/2,h/2,image=self.photo)
    def select(self,index):
        if not self.busy and 0<=index<len(self.entries) and self.save_note():self.index=index;self.show()
    def preview(self):
        if self.busy or not self.save_note():return
        from code.scripts.gt_mixed_editor import FrameEditor
        case=self.store.manifest['cases'][self.local_index];self.store.current_case_rank=case['rank']
        try:self.editor=FrameEditor(self,case)
        except (ValueError,OSError) as e:messagebox.showerror('Không mở được video',str(e),parent=self.root)
    def reset(self):
        if self.busy:return
        try:
            self.store.save(self.local_index,'',self.note.get('1.0','end-1c'));self.refresh();self.status.set('Đã đặt lại quyết định, các chỉnh sửa GT vẫn giữ nguyên.')
        except (ValueError,OSError) as e:messagebox.showerror('Chưa đặt lại',str(e),parent=self.root)
    def accept(self):
        if self.store.manifest['cases'][self.local_index]['proposal']:self.verdict('APPLY')
    def verdict(self,value):
        if self.busy or not self.save_note():return
        try:
            self.store.save(self.local_index,value,self.note.get('1.0','end-1c'))
            self.status.set('Đã lưu xử lý điểm audit.');self.refresh()
            pending=next((i for i,(name,j) in enumerate(self.entries) if i>self.index and not self.stores[name].rows[j]['decision']),None)
            if pending is not None:self.index=pending;self.show()
        except (ValueError,OSError) as e:messagebox.showerror('Chưa xác nhận',str(e),parent=self.root)
    def finish(self):
        if self.busy or not self.save_note():return
        reviewer=self.reviewer.get().strip()
        if not reviewer:messagebox.showerror('Thiếu người review','Điền tên người review.',parent=self.root);return
        self.busy=True;self.note.configure(state='disabled');self.reviewer.configure(state='disabled');self.refresh()
        for b in self.controls:b.configure(state='disabled')
        def work():
            try:self.messages.put((True,str(finalize_collection(self.path,lambda m:self.messages.put((None,m)),reviewer))))
            except Exception as e:self.messages.put((False,str(e)))
        threading.Thread(target=work,daemon=True).start();self.root.after(150,self.poll)
    def poll(self):
        try:ok,msg=self.messages.get_nowait()
        except queue.Empty:self.root.after(150,self.poll);return
        self.status.set(msg)
        if ok is None:self.root.after(150,self.poll);return
        self.busy=False;self.note.configure(state='normal');self.reviewer.configure(state='normal')
        for b in self.controls:b.configure(state='normal')
        self.refresh();(messagebox.showinfo if ok else messagebox.showerror)('GT sau audit',msg,parent=self.root)
    def close(self):
        if self.busy:messagebox.showinfo('Đang xuất GT','Đợi xuất/chấm hoàn tất trước khi đóng.',parent=self.root);return
        if hasattr(self,'editor') and self.editor.window.winfo_exists():
            self.editor.close()
            if self.editor.window.winfo_exists():return
        if self.save_note():self.root.destroy()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection');args=p.parse_args()
    path=args.collection or (active_collection() if (HOME/'active.json').exists() else prepare())
    root=tk.Tk()
    try:ReconcileApp(root,path)
    except (ValueError,OSError) as e:root.withdraw();messagebox.showerror('Không mở được sửa GT',str(e),parent=root);root.destroy();return
    root.mainloop()

if __name__=='__main__':main()
