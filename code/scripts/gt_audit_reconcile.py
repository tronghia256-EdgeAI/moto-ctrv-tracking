"""Targeted corrections on copies of the completed GT, preserving its audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,tempfile,uuid
from pathlib import Path
from code.scripts.paper_gt_audit import ROOT,active as active_audit,encoded
from code.scripts.complete_paper_gt_audit import complete,saved_source_ref,group_locations
from code.scripts.gt_mixed_review import MixedStore,DECISION_FIELDS,apply_review,read_rows,now,finalize_mixed,BUNDLE
from code.scripts.gt_review import sha256
from code.scripts.gt_roi import load_roi,filter_frames,contains_box
from code.scripts.import_gt import load_mot,to_framelist
from code.scripts.audit_disagreements import unmatched,runs
from code.scripts.gt_review_queue import _iou

HOME=ROOT/'data/annotations/paper_gt_reconciliation'

def resolution_matches(case,source,records):
    frame=case['focus_frame'];ref=case['audit_reference']
    if ref['kind']=='extra':
        return any(int(p[0])==frame and int(p[7])==2 and _iou(ref['box'],[float(p[2]),float(p[3]),float(p[2])+float(p[4]),float(p[3])+float(p[5])])>=.5 for p in records.values())
    if ref['kind']=='missing':
        key=(frame,ref['tid']);before=source[key];after=records.get(key)
        if ref.get('verdict')=='D':
            def overlapping(rows):
                return sum(int(p[0])==frame and int(p[7])==2 and _iou(ref['box'],[float(p[2]),float(p[3]),float(p[2])+float(p[4]),float(p[3])+float(p[5])])>=.5 for p in rows.values())
            # The judge may keep the red box and delete/correct the OTHER copy.
            if overlapping(source)>1 and overlapping(records)==1:return True
        return after is None or any(abs(float(a)-float(b))>1e-6 for a,b in zip(after[2:6],before[2:6])) or int(after[7])!=int(before[7])
    a=records.get((case['before'][1],ref['tid']));b=records.get((case['after'][1],ref['tid']))
    return a is None or b is None or a[1]!=b[1]

class ReconcileStore(MixedStore):
    def check_inputs(self,images=False):
        self.assert_unchanged()
        if self.manifest.get('review_workflow')!='AUDIT_RECONCILIATION':raise ValueError('Not a reconciliation bundle')
        if self.manifest['segment'][1]-self.manifest['segment'][0]!=self.manifest['n_frames']:raise ValueError('Frame count mismatch')
        for path,digest in self.manifest['parent_inputs'].items():
            if sha256(Path(path))!=digest:raise ValueError('Parent GT or saved audit changed; prepare a new correction version')
        for rank,c in enumerate(self.manifest['cases'],1):
            if c['rank']!=rank or not (self.bundle/c['sheet']).is_file():raise ValueError('Missing correction case')
            if not 1<=c['window'][0]<=c['window'][1]<=self.manifest['n_frames']:raise ValueError('Invalid correction window')
            for tid,frame in (c['before'],c['after']):
                if tid and (frame,tid) not in self.source_rows:raise ValueError('Correction anchor missing')
        roi=load_roi(self.bundle)
        if not roi or not roi['enabled']:raise ValueError('Frozen ROI required')
        if images:
            directory=Path(self.manifest['frames_dir'])
            for filename,digest in self.manifest['images_sha256'].items():
                if sha256(directory/filename)!=digest:raise ValueError('Source image changed')
    def seen_for(self,rank):
        return set(self.journal.get('reconcile_seen',{}).get(str(rank),[]))
    def record_seen(self,frames):
        rank=self.current_case_rank
        seen=dict(self.journal.get('reconcile_seen',{}));seen[str(rank)]=sorted(self.seen_for(rank)|set(frames))
        self.update_journal(dict(self.journal,reconcile_seen=seen))
    def save(self,index,decision,note):
        case=self.manifest['cases'][index]
        if decision:
            missing=set(range(case['window'][0],case['window'][1]+1))-self.seen_for(case['rank'])
            if missing:raise ValueError(f"Mở V và xem toàn đoạn quanh lỗi trước ({len(missing)} frame chưa xem).")
            if decision=='KEEP' and len(note.strip())<8:raise ValueError('Ghi lý do giữ GT sau khi đối chiếu lại audit')
            if decision in ('APPLY','EDIT'):
                rows=[dict(r) for r in self.rows];rows[index].update(decision=decision,note=note)
                records,_=apply_review(self.source_rows,self.manifest,rows,self.journal)
                if not resolution_matches(case,self.source_rows,records):raise ValueError('Điểm audit vẫn chưa được xử lý: kiểm tra đúng box/ID hoặc ghi lý do giữ GT.')
        return super().save(index,decision,note)
    def preview(self,rows=None,journal=None,strict=False):
        result=super().preview(rows=rows,journal=journal,strict=strict)
        if strict:
            for case,row in zip(self.manifest['cases'],self.rows if rows is None else rows):
                if set(range(case['window'][0],case['window'][1]+1))-self.seen_for(case['rank']):raise ValueError('Correction playback incomplete')
                if row['decision']=='KEEP':
                    if len(row['note'].strip())<8:raise ValueError('Unchanged GT needs a documented adjudication')
                elif not resolution_matches(case,self.source_rows,result[0]):raise ValueError('Unresolved audit issue')
        return result

def candidate_run(issue,truth,pr,start,end):
    found=runs(pr,[unmatched(pr[k],truth[k]) for k in range(end-start)])
    frame=issue['frame']-start
    matches=[run for run in found if run[0][1]==issue['tid'] and any(k==frame for k,_,_ in run)]
    if len(matches)!=1 or len(matches[0])!=issue['run_frames']:raise ValueError('Audited prediction run cannot be reproduced')
    return matches[0]

def prepare(audit=None):
    audit=Path(audit or active_audit());result=complete(audit,saved_source_ref(audit))
    groups=group_locations(result['issues'])
    if not groups:raise ValueError('No remaining correction cases')
    protocol=json.loads((audit/'protocol.json').read_text(encoding='utf-8'))
    parent_collection=next(Path(p) for p in protocol['inputs'] if Path(p).name=='completed_roi_collection.json')
    collection=json.loads(parent_collection.read_text(encoding='utf-8'))
    identity=hashlib.sha256(encoded(dict(audit_result=sha256(audit/'audit_result.json'),generator=sha256(Path(__file__))))).hexdigest()[:16]
    target=HOME/f'reconcile_{identity}';HOME.mkdir(parents=True,exist_ok=True)
    if target.exists():
        receipt=json.loads((target/'collection.json').read_text(encoding='utf-8'))
        for bundle in receipt['bundles'].values():ReconcileStore(bundle).check_inputs(images=True)
        (HOME/'active.json').write_bytes(encoded(dict(collection=str(target/'collection.json'))));return target/'collection.json'
    target.mkdir()
    by_segment={}
    for group in groups:by_segment.setdefault(group['reference']['segment'],[]).append(group)
    dumps={name:json.loads((ROOT/f'experiments/gt_ablation/tracks_{name}.json').read_text(encoding='utf-8')) for name in ('baseline','ctrv_gate_nsa')}
    bundles={}
    for segment,segment_groups in sorted(by_segment.items()):
        item=collection['segments'][segment];release=Path(item['release']);old=json.loads((release/'manifest.json').read_text(encoding='utf-8'))
        start,end=old['segment'];truth=load_mot(release/'gt_roi.txt');gt=to_framelist(truth,end-start,1)
        roi=load_roi(release);source=read_rows(release/'gt_roi.txt',end-start)
        predictions={}
        for name,dump in dumps.items():
            fl={k+1:[(tid,b,cls) for tid,cls,b in rows] for k,rows in enumerate(dump['frames'][start:end])}
            predictions[name]=to_framelist(filter_frames(fl,roi),end-start,1)
        cases=[]
        for group in sorted(segment_groups,key=lambda g:({'identity':0,'missing':1,'extra':2}[g['reference']['kind']],g['reference']['frame'])):
            ref=dict(group['reference']);focus=ref['frame']-start+1;suggestions=[]
            before=ref.get('before',ref['frame'])-start+1;ids=[];proposal=None
            if ref['kind']=='extra':
                run=candidate_run(ref,gt,predictions[ref['tracker']],start,end)
                lo=max(1,run[0][0]+1-8);hi=min(end-start,run[-1][0]+1+8)
                suggestions=[dict(frame=k+1,box=list(b),audited=(k+start==ref['frame'])) for k,tid,b in run]
                anchors=[[0,focus],[0,focus]]
                question=f"Audit #{', #'.join(group['members'])}: có xe thật chưa khớp GT. V: xem đoạn, thêm/chỉnh box và chọn đúng ID. Box cam chỉ là gợi ý; E xác nhận đã sửa."
            else:
                lo=max(1,before-8);hi=min(end-start,focus+8);ids=[ref['tid']]
                anchors=[[ref['tid'],before],[ref['tid'],focus]]
                if ref['kind']=='identity':
                    proposal=dict(op='split',tid=ref['tid'],frame=focus)
                    question=f"Audit #{', #'.join(group['members'])}: hai xe khác nhau đang cùng ID {ref['tid']}. Xem đoạn để chốt điểm tách. A đề xuất tách tại MOT {focus}; V/E nếu điểm tách khác."
                else:question=f"Audit #{', #'.join(group['members'])}: box GT ID {ref['tid']} bị đánh dấu sai/trùng. V sửa/xóa đúng box ở MOT {focus}; E xác nhận."
            rank=len(cases)+1
            cases.append(dict(rank=rank,kind='REPAIR',ids=ids,before=anchors[0],after=anchors[1],proposal=proposal,
                question=question,window=[lo,hi],sheet=f'{rank:03d}_REPAIR.png',focus_frame=before if ref['kind']=='identity' else focus,
                focus_tid=ref['tid'] if ids else None,audit_reference=dict(ref,frame=focus,before=before),
                audit_cases=group['members'],suggestions=suggestions))
        bundle=BUNDLE/'roi_reviews'/f'reconcile_{identity}_{segment}'
        if bundle.exists():raise ValueError('Partial correction preparation exists; keep it and choose a new version')
        staging=Path(tempfile.mkdtemp(prefix='.reconcile-',dir=bundle.parent))
        try:
            shutil.copyfile(release/'gt_roi.txt',staging/'source_gt.txt')
            parent_inputs={str(p):sha256(p) for p in (release/'gt_roi.txt',release/'manifest.json',release/'roi.json',audit/'audit_result.json',audit/'verdicts.csv',audit/'cases.json',audit/'key.json')}
            manifest=dict(old,created_utc=now(),review_workflow='AUDIT_RECONCILIATION',review_scope='ROI_ONLY',cases=cases,
                          segment_name=segment,
                          parent_inputs=parent_inputs,source=dict(path=str(release/'gt_roi.txt'),sha256=sha256(release/'gt_roi.txt'),seed='Completed human-reviewed ROI GT, copied for audit corrections'))
            (staging/'manifest.json').write_bytes(encoded(manifest))
            (staging/'roi.json').write_bytes(encoded(dict(roi,manifest_sha256=sha256(staging/'manifest.json'))))
            provenance=dict(schema_version=1,workflow='AUDIT_RECONCILIATION',segment=segment,parent_release=str(release),
                            audit=str(audit),audit_verdicts_sha256=sha256(audit/'verdicts.csv'),parent_scope=item['report'].get('review_scope_provenance'),
                            inherited_playback='Parent review displayed all 600 frames; correction windows require fresh playback.')
            (staging/'scope_provenance.json').write_bytes(encoded(provenance))
            (staging/'edits.json').write_bytes(encoded(dict(schema_version=1,events=[],seen_frames=list(range(1,end-start+1)),reconcile_seen={})))
            with (staging/'decisions.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=DECISION_FIELDS);writer.writeheader()
                for c in cases:
                    writer.writerow(dict(rank=c['rank'],sheet=c['sheet'],kind=c['kind'],ids=' '.join(map(str,c['ids'])),
                        frame_before=c['before'][1],frame_after=c['after'][1],decision='',note=''))
                    original=int(c['audit_cases'][0]);shutil.copyfile(audit/f'case_{original:03d}_context.png',staging/c['sheet'])
            staging.rename(bundle)
        finally:
            if staging.exists():
                if staging.resolve().parent!=bundle.parent.resolve() or not staging.name.startswith('.reconcile-'):raise ValueError('Unsafe staging cleanup')
                shutil.rmtree(staging)
        store=ReconcileStore(bundle);store.check_inputs(images=True);bundles[segment]=str(bundle)
        print(segment,len(cases),'correction cases prepared',flush=True)
    receipt=dict(schema_version=1,audit=str(audit),audit_result_sha256=sha256(audit/'audit_result.json'),bundles=bundles,
                 status='HUMAN_CORRECTION_REQUIRED',locations=len(groups),created_utc=now())
    (target/'collection.json').write_bytes(encoded(receipt));(HOME/'active.json').write_bytes(encoded(dict(collection=str(target/'collection.json'))))
    return target/'collection.json'

def active_collection():return Path(json.loads((HOME/'active.json').read_text(encoding='utf-8'))['collection'])

def finalize_collection(path=None,progress=print,reviewer='Nghĩa'):
    if not reviewer.strip():raise ValueError('Reviewer is required')
    path=Path(path or active_collection());collection=json.loads(path.read_text(encoding='utf-8'))
    stores={name:ReconcileStore(bundle) for name,bundle in collection['bundles'].items()}
    for store in stores.values():store.check_inputs(images=True);store.preview(strict=True)
    releases={}
    for name,store in stores.items():
        out=finalize_mixed(store,reviewer,progress=lambda msg:progress(f'{name}: {msg}'))
        releases[name]=dict(bundle=str(store.bundle),release=str(out),report=json.loads((out/'review.json').read_text(encoding='utf-8')))
    result=dict(schema_version=1,parent_collection=str(path),segments=releases,created_utc=now(),role='development; rescored cached old tracks',independent_test=False)
    destination=path.parent/f'reconciled_{uuid.uuid4().hex[:8]}.json';destination.write_bytes(encoded(result))
    # The original collection and viewer pointers are bound by the original audit.
    # Publish a separate receipt; future runs must name this corrected GT version.
    (path.parent/'latest_release.json').write_bytes(encoded(dict(collection=str(destination))))
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('step',choices=('prepare','finalize'));args=p.parse_args()
    print(prepare() if args.step=='prepare' else finalize_collection(),flush=True)

if __name__=='__main__':main()
