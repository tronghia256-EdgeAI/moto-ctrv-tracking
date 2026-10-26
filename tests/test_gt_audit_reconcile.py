import csv,json
from pathlib import Path
import pytest
from PIL import Image
from tests.test_gt_mixed_review import factory
from code.scripts.gt_audit_reconcile import ReconcileStore,resolution_matches
from code.scripts.gt_mixed_review import DECISION_FIELDS,finalize_mixed
from code.scripts.gt_review import sha256
from code.scripts.gt_roi import save_roi
from code.scripts.paper_gt_audit import encoded
import code.scripts.gt_audit_reconcile as module
from code.scripts.gt_mixed_editor import FrameEditor
from types import SimpleNamespace

def correction(factory,kind,raw=None,verdict=None):
    raw=raw or ''.join(f'{f},1,100,100,20,40,1,2,1\n' for f in range(1,7))
    original=factory(raw=raw);p=original.bundle
    parent=p.parent/'parent.txt';parent.write_text('immutable parent')
    ref=dict(kind=kind,tid=1,box=[200,200,220,240] if kind=='extra' else [100,100,120,140])
    if verdict:ref['verdict']=verdict
    c=dict(rank=1,kind='REPAIR',ids=[] if kind=='extra' else [1],before=[0,1] if kind=='extra' else [1,1],
           after=[0,1] if kind=='extra' else [1,4] if kind=='identity' else [1,1],proposal=dict(op='split',tid=1,frame=4) if kind=='identity' else None,
           focus_frame=1,audit_reference=ref,question='Correction',window=[1,6],sheet='repair.png')
    manifest=dict(original.manifest,cases=[c],review_workflow='AUDIT_RECONCILIATION',review_scope='ROI_ONLY',parent_inputs={str(parent):sha256(parent)})
    (p/'manifest.json').write_bytes(encoded(manifest));Image.new('RGB',(800,450),'white').save(p/c['sheet'])
    save_roi(p,[[0,0],[800,0],[800,450],[0,450]],True,'Nghĩa','Readable road scope',1,None)
    (p/'edits.json').write_bytes(encoded(dict(schema_version=1,events=[],seen_frames=list(range(1,7)),reconcile_seen={})))
    with (p/'decisions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=DECISION_FIELDS);w.writeheader();w.writerow(dict(rank=1,sheet=c['sheet'],kind='REPAIR',ids=' '.join(map(str,c['ids'])),frame_before=1,frame_after=c['after'][1],decision='',note=''))
    s=ReconcileStore(p);s.current_case_rank=1
    return s,parent

def test_parent_playback_does_not_approve_new_window_and_split_preserves_source(factory):
    s,parent=correction(factory,'identity');before=s.source_path.read_bytes()
    with pytest.raises(ValueError,match='chưa xem'):s.save(0,'APPLY','')
    s.record_seen(range(1,7));s.save(0,'APPLY','Confirmed different vehicles')
    rows,_=s.preview(strict=True)
    assert rows[(1,1)][1]!=rows[(4,1)][1]
    assert s.source_path.read_bytes()==before and parent.read_text()=='immutable parent'

def test_missing_box_requires_explicit_edit_and_exports_only_new_gt(factory,tmp_path):
    s,parent=correction(factory,'extra');before=s.source_path.read_bytes()
    s.record_seen(range(1,7))
    with pytest.raises(ValueError):s.save(0,'EDIT','')
    s.add_event(dict(op='upsert',fields=[1,2,200,200,20,40,1,2,1],case_rank=1,reviewer='Nghĩa'))
    s.save(0,'EDIT','Added sampled motorcycle')
    out=finalize_mixed(s,'Nghĩa',out_dir=tmp_path/'export',evaluate=False)
    assert len((out/'gt_roi.txt').read_text().splitlines())==7
    assert s.source_path.read_bytes()==before
    assert json.loads((out/'review.json').read_text())['gt_changes']['added_boxes']==1

def test_bad_gt_box_delete_and_documented_keep_are_separate(factory):
    s,parent=correction(factory,'missing');s.record_seen(range(1,7))
    with pytest.raises(ValueError,match='lý do'):s.save(0,'KEEP','')
    s.save(0,'KEEP','Reinspection shows original box is valid')
    assert len(s.preview(strict=True)[0])==6
    s.save(0,'','');s.add_event(dict(op='delete_box',frame=1,tid=1,case_rank=1));s.save(0,'EDIT','Deleted wrong box')
    assert len(s.preview(strict=True)[0])==5

def test_formatting_or_confidence_only_does_not_resolve_bad_geometry(factory):
    s,parent=correction(factory,'missing')
    case=s.manifest['cases'][0];records={k:list(v) for k,v in s.source_rows.items()}
    records[(1,1)][2]='100.0';records[(1,1)][6]='0.9'
    assert not resolution_matches(case,s.source_rows,records)

def test_duplicate_can_be_resolved_by_deleting_other_box(factory):
    raw=''.join(f'{f},1,100,100,20,40,1,2,1\n' for f in range(1,7))+'1,2,101,100,20,40,1,2,1\n'
    s,parent=correction(factory,'missing',raw=raw,verdict='D');s.record_seen(range(1,7))
    original=list(s.source_rows[(1,1)])
    s.add_event(dict(op='delete_box',frame=1,tid=2,case_rank=1))
    s.save(0,'EDIT','Kept the audited box and deleted its duplicate')
    records,_=s.preview(strict=True)
    assert records[(1,1)]==original and (1,2) not in records

def test_parent_changes_and_outside_roi_edits_are_rejected(factory):
    s,parent=correction(factory,'extra');before=s.edit_path.read_bytes()
    with pytest.raises(ValueError,match='outside'):s.add_event(dict(op='upsert',fields=[1,2,200,440,20,40,1,2,1],case_rank=1))
    assert s.edit_path.read_bytes()==before
    parent.write_text('modified source')
    with pytest.raises(ValueError,match='Parent'):s.check_inputs()

def test_collection_checks_all_segments_before_any_export(tmp_path,monkeypatch):
    path=tmp_path/'collection.json';path.write_bytes(encoded(dict(bundles=dict(mixed='mixed',dense='dense',freeflow='freeflow'))))
    exported=[]
    class Store:
        def __init__(self,bundle):self.bundle=bundle
        def check_inputs(self,images=False):pass
        def preview(self,strict=False):
            if self.bundle=='dense':raise ValueError('Pending dense review')
            return {},{}
    monkeypatch.setattr(module,'ReconcileStore',Store)
    monkeypatch.setattr(module,'finalize_mixed',lambda *a,**k:exported.append(a))
    with pytest.raises(ValueError,match='Pending'):module.finalize_collection(path)
    assert not exported and not (tmp_path/'latest_release.json').exists()

def test_new_receipt_preserves_parent_and_original_viewer_pointer(tmp_path,monkeypatch):
    path=tmp_path/'collection.json';path.write_bytes(encoded(dict(bundles=dict(mixed='mixed',dense='dense',freeflow='freeflow'))))
    before=path.read_bytes();pointer=tmp_path/'roi_active.json';pointer.write_text('original pointer')
    class Store:
        def __init__(self,bundle):self.bundle=bundle
        def check_inputs(self,images=False):pass
        def preview(self,strict=False):return {},{}
    def export(store,reviewer,progress):
        assert reviewer=='Reviewer supplied'
        out=tmp_path/store.bundle;out.mkdir();(out/'review.json').write_text('{}');return out
    monkeypatch.setattr(module,'ReconcileStore',Store);monkeypatch.setattr(module,'finalize_mixed',export)
    out=module.finalize_collection(path,reviewer='Reviewer supplied')
    assert path.read_bytes()==before and pointer.read_text()=='original pointer'
    assert len(json.loads(out.read_text())['segments'])==3
    assert json.loads((tmp_path/'latest_release.json').read_text())['collection']==str(out)

def test_suggestion_needs_explicit_button_and_correct_motorcycle_id():
    editor=FrameEditor.__new__(FrameEditor);events=[];messages=[]
    editor.case=dict(suggestions=[dict(frame=1,box=[10,20,30,60])]);editor.frame=SimpleNamespace(get=lambda:1)
    editor.store=SimpleNamespace(new_track_id=lambda:7);editor.status=SimpleNamespace(set=messages.append)
    editor.rows={};editor.selected=None;editor.edit=lambda e:events.append(e)
    editor.use_suggestion(False);assert not events
    editor.selected=2;editor.rows={(2,2):['2','2','0','0','10','20','1','1','1']}
    editor.use_suggestion(False);assert not events and 'không phải xe máy' in messages[-1]
    editor.rows[(2,2)][7]='2';editor.use_suggestion(False)
    assert events[-1]['fields'][:6]==[1,2,10,20,20,40]
    editor.use_suggestion(True);assert events[-1]['fields'][1]==7
