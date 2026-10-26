"""Prepare a source-bound, fixed-sample audit of completed ROI GT; never edit GT."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from code.scripts.audit_disagreements import unmatched, runs, switch_events, crop
from code.scripts.gt_review import REPO_ROOT, atomic_write, sha256
from code.scripts.gt_roi import load_roi, filter_frames
from code.scripts.import_gt import load_mot, to_framelist

ROOT=REPO_ROOT
HOME=ROOT/'data/annotations/paper_gt_audit'
COLLECTION=ROOT/'data/processed/gt_diagnostics/completed_roi_collection.json'
FIELDS=['n','verdict','note','reviewer','updated_utc']
ALLOWED={'box':{'Y','N','D','?'},'identity':{'Y','N','?'}}

def encoded(obj):
    return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')

def bindings_valid(manifest):
    for path,expected in manifest['inputs'].items():
        if sha256(Path(path))!=expected:
            raise ValueError(f'Đầu vào đã đổi; cần tạo phiên audit mới: {path}')

def select_indices(size,limit,rng):
    return sorted(rng.choice(size,size=min(size,limit),replace=False).tolist())

def gt_issue(kind,verdict):
    return ((kind=='extra' and verdict=='Y') or
            (kind=='missing' and verdict in {'N','D'}) or
            (kind=='identity' and verdict=='N'))

def checkpoint(root):
    root=Path(root); uploaded=root/'ket_qua_train/best.pt'; previous=root/'best.pt'
    if not uploaded.is_file(): raise ValueError('Missing ket_qua_train/best.pt')
    if previous.exists() and sha256(previous)!=sha256(uploaded):
        raise ValueError('best.pt ở root khác bản trong ket_qua_train; cần xác nhận checkpoint')
    return uploaded

def prepare(collection=COLLECTION,home=HOME,per_stratum=10,switch_limit=10,seed=20261012):
    if per_stratum<1 or switch_limit<1:
        raise ValueError('Sample sizes must be positive')
    collection=Path(collection).resolve(); home=Path(home).resolve()
    data=json.loads(collection.read_text(encoding='utf-8'))
    inputs={str(collection):sha256(collection)}
    config_paths=[ROOT/'configs'/f for f in ('baseline.yaml','exp1_C2.yaml','exp2_C1C2.yaml','exp3_C1C2C3.yaml','mixed_roi.yaml','default.yaml')]
    code_paths=[Path(__file__),ROOT/'code/scripts/audit_disagreements.py',ROOT/'code/pipeline.py',ROOT/'code/evaluation/metrics.py']
    for directory in ('models','detection','tracker'):
        code_paths.extend(sorted((ROOT/'code'/directory).rglob('*.py')))
    weights=checkpoint(ROOT); training=ROOT/'ket_qua_train/training_protocol.json'
    protocol=json.loads(training.read_text(encoding='utf-8'))
    if protocol.get('status')!='COMPLETED':
        raise ValueError('Training protocol is not completed')
    if protocol.get('class_names')!=['Accident','bus','car','motorcycle','truck']:
        raise ValueError('Training class order does not match the verified detector mapping')
    for p in config_paths+code_paths+[weights,training]: inputs[str(p.resolve())]=sha256(p)
    if (ROOT/'best.pt').exists():inputs[str(ROOT/'best.pt')]=sha256(ROOT/'best.pt')
    dumps={}
    for tracker in ('baseline','ctrv_gate_nsa'):
        p=ROOT/f'experiments/gt_ablation/tracks_{tracker}.json'
        inputs[str(p)]=sha256(p); dumps[tracker]=json.loads(p.read_text(encoding='utf-8'))
    reviews={}
    for name,item in sorted(data['segments'].items()):
        release=Path(item['release']); report=json.loads((release/'review.json').read_text(encoding='utf-8'))
        if report!=item['report'] or report['evaluation_scope']!='ROI':
            raise ValueError('Collection does not match completed ROI reports')
        bundle=Path(item['bundle'])
        for filename,digest in report['snapshot_sha256'].items():
            for parent in (release,bundle):
                p=parent/filename
                if sha256(p)!=digest: raise ValueError(f'Review changed: {p}')
                inputs[str(p.resolve())]=digest
        for filename in ('review.json','gt_roi.txt'):
            p=release/filename; inputs[str(p.resolve())]=sha256(p)
        if sha256(release/'gt_roi.txt')!=report['roi_gt']['sha256']:
            raise ValueError('GT fingerprint does not match its release')
        manifest=json.loads((release/'manifest.json').read_text(encoding='utf-8'))
        reviews[name]=(release,report,manifest,load_roi(release))
    plan=dict(schema_version=1,inputs=inputs,seed=seed,membership_per_stratum=per_stratum,
              switches_per_tracker_segment=switch_limit,
              stopping_rule='Fixed sample: judge all items. No interim outcome-based stopping.',
              scope='Motorcycle disagreements versus baseline and C2 on all three completed ROI segments; cached old tracks, not new best.pt inference.',
              future_run=dict(weights_path=str(weights),weights_sha256=sha256(weights),input_sizes=[640,448],
                  detector_confidence_floor=0.1,roi_policy='Mask outside before detection; bottom-center filtering before tracking; expire outside active/lost tracks.',
                  role='Development only. No independent camera test has been registered.',
                  config_policy='Snapshot candidate configs; log resolved config per run. A change creates a new protocol version; no test tuning.'),
              training_protocol_sha256=sha256(training),
              pending=['human audit','adjudication of confirmed GT problems or uncertain cases','independent test footage and source/permission declaration','validation of ROI ablation runner before execution'])
    identity=hashlib.sha256(encoded(plan)).hexdigest()[:16]
    target=home/f'roi_{identity}'
    if target.exists():
        store=AuditStore(target); store.check_inputs()
        (home/'active.json').write_bytes(encoded(dict(bundle=str(target))))
        return target
    home.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.audit-',dir=home))
    try:
        rng=np.random.default_rng(seed); keys=[]; strata=[]
        truth_frames={}; image_specs={}
        for name,(release,report,manifest,roi) in reviews.items():
            start,end=report['segment_zero_based_end_exclusive']
            truth=filter_frames(load_mot(release/'gt_roi.txt'),roi)
            gt=to_framelist(truth,end-start,1)
            for local in range(1,end-start+1):
                absolute=start+local-1
                truth_frames[absolute]=[b for _,b,_ in truth.get(local,[])]
                image_specs[absolute]=(Path(manifest['frames_dir'])/f'{local:06d}.jpg',manifest['images_sha256'][f'{local:06d}.jpg'])
            for tracker,dump in dumps.items():
                if dump.get('start_frame',0)!=0 or len(dump['frames'])<end: raise ValueError('Track dump does not cover segment')
                predictions=filter_frames({i+1:[(tid,b,cls) for tid,cls,b in entries] for i,entries in enumerate(dump['frames'][start:end])},roi)
                pr=to_framelist(predictions,end-start,1)
                for kind,a,b in (('extra',pr,gt),('missing',gt,pr)):
                    found=runs(a,[unmatched(a[k],b[k]) for k in range(end-start)])
                    chosen=select_indices(len(found),per_stratum,rng)
                    strata.append(dict(segment=name,tracker=tracker,kind=kind,available=len(found),sampled=len(chosen)))
                    for j in chosen:
                        k,tid,box=found[j][len(found[j])//2]
                        keys.append(dict(type='box',kind=kind,segment=name,tracker=tracker,tid=tid,
                                         frame=start+k,box=list(box),run_frames=len(found[j])))
                events=switch_events(gt,pr)
                expected=report['variants'][tracker]['metrics']['motorcycle']['num_switches']
                if len(events)!=expected: raise ValueError('Switch audit does not replicate the completed scoring report')
                chosen=select_indices(len(events),switch_limit,rng)
                strata.append(dict(segment=name,tracker=tracker,kind='identity',available=len(events),sampled=len(chosen)))
                for j in chosen:
                    before,k,tid,bbox,box=events[j]
                    keys.append(dict(type='identity',kind='identity',segment=name,tracker=tracker,tid=tid,
                                     before=start+before,frame=start+k,before_box=list(bbox),box=list(box)))
        keys=[dict(n=str(i+1),**keys[j]) for i,j in enumerate(rng.permutation(len(keys)))]
        cases=[]
        def read_image(frame):
            path,digest=image_specs[frame]
            if sha256(path)!=digest: raise ValueError(f'Frame changed: {path}')
            inputs[str(path.resolve())]=digest
            img=cv2.imread(str(path))
            if img is None or img.shape[:2]!=(450,800): raise ValueError('Invalid source frame')
            return img
        def context(img,frame,box):
            img=img.copy()
            for b in truth_frames[frame]: cv2.rectangle(img,tuple(map(int,b[:2])),tuple(map(int,b[2:])),(0,200,0),1)
            cv2.rectangle(img,tuple(map(int,box[:2])),tuple(map(int,box[2:])),(0,0,255),2)
            return img
        for row in keys:
            frame=row['frame']; img=read_image(frame)
            tile=crop(img,row['box'],truth_frames[frame]); full=context(img,frame,row['box'])
            if row['type']=='identity':
                previous=read_image(row['before'])
                tile=np.hstack([crop(previous,row['before_box'],truth_frames[row['before']]),tile])
                full=np.hstack([context(previous,row['before'],row['before_box']),full])
            filenames=[]
            for label,pixels in (('crop',tile),('context',full)):
                filename=f"case_{int(row['n']):03d}_{label}.png"; path=staging/filename
                if not cv2.imwrite(str(path),pixels): raise OSError('Cannot write audit image')
                filenames.append(filename)
            cases.append(dict(n=row['n'],type=row['type'],images=filenames,
                              image_sha256=[sha256(staging/p) for p in filenames]))
        # Hash package identity before adding sampled image bindings; stable across reruns.
        plan['strata']=strata; plan['cases_sha256']=hashlib.sha256(encoded(cases)).hexdigest()
        plan['key_sha256']=hashlib.sha256(encoded(keys)).hexdigest()
        plan['created_utc']=datetime.now(timezone.utc).isoformat()
        (staging/'protocol.json').write_bytes(encoded(plan))
        (staging/'key.json').write_bytes(encoded(keys))
        (staging/'cases.json').write_bytes(encoded(cases))
        (staging/'verdicts.csv').write_text(','.join(FIELDS)+'\n'+''.join(f"{r['n']},,,,\n" for r in cases),encoding='utf-8')
        bindings_valid(plan)
        staging.rename(target)
    finally:
        if staging.exists():
            if staging.resolve().parent!=home or not staging.name.startswith('.audit-'): raise ValueError('Unsafe temporary cleanup')
            shutil.rmtree(staging)
    (home/'active.json').write_bytes(encoded(dict(bundle=str(target))))
    return target

class AuditStore:
    def __init__(self,bundle):
        self.bundle=Path(bundle); self.path=self.bundle/'verdicts.csv'
        self.protocol_raw=(self.bundle/'protocol.json').read_bytes()
        self.protocol=json.loads(self.protocol_raw)
        raw=(self.bundle/'cases.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.protocol['cases_sha256']: raise ValueError('Case inventory changed')
        self.cases=json.loads(raw)
        self.reload()
    def reload(self):
        raw=self.path.read_bytes(); reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        if reader.fieldnames!=FIELDS: raise ValueError('Invalid verdict CSV schema')
        self.rows=list(reader)
        if [r['n'] for r in self.rows]!=[c['n'] for c in self.cases]: raise ValueError('Verdict inventory changed')
        for case,row in zip(self.cases,self.rows):
            if None in row or any(v is None for v in row.values()) or row['verdict'] not in ALLOWED[case['type']]|{''}: raise ValueError('Invalid verdict')
        self.fingerprint=hashlib.sha256(raw).hexdigest()
    def check_inputs(self):
        if (self.bundle/'protocol.json').read_bytes()!=self.protocol_raw: raise ValueError('Audit protocol changed')
        bindings_valid(self.protocol)
    def save(self,index,verdict,note='',reviewer='Nghĩa'):
        case=self.cases[index]
        if verdict not in ALLOWED[case['type']]|{''}: raise ValueError('Invalid verdict for this question')
        if not reviewer.strip(): raise ValueError('Reviewer is required')
        if (self.bundle/'protocol.json').read_bytes()!=self.protocol_raw: raise ValueError('Audit protocol changed')
        rows=[dict(r) for r in self.rows]
        rows[index].update(verdict=verdict,note=note,reviewer=reviewer.strip(),updated_utc=datetime.now(timezone.utc).isoformat())
        output=io.StringIO(newline=''); writer=csv.DictWriter(output,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
        payload=output.getvalue().encode('utf-8'); atomic_write(self.path,payload,self.fingerprint)
        self.rows=rows; self.fingerprint=hashlib.sha256(payload).hexdigest()
    def finish(self):
        self.check_inputs(); self.reload()
        if any(not r['verdict'] for r in self.rows): raise ValueError('Duyệt hết các mục trước khi tổng hợp')
        raw=(self.bundle/'key.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.protocol['key_sha256']: raise ValueError('Private key changed')
        keys=json.loads(raw)
        issues=[dict(k,**{f:v[f] for f in FIELDS if f!='n'}) for k,v in zip(keys,self.rows) if gt_issue(k['kind'],v['verdict']) or v['verdict']=='?']
        result=dict(status='HUMAN_RECONCILIATION_REQUIRED' if issues else 'SAMPLED_AUDIT_COMPLETE',
                    sample_size=len(keys),issues=issues,verdicts_sha256=self.fingerprint,
                    limitations='Sampled audit of old tracker disagreements, not proof of error-free GT or independent test. No GT changed.',
                    independent_test_registered=False)
        (self.bundle/'audit_result.json').write_bytes(encoded(result)); return result

def active():
    return Path(json.loads((HOME/'active.json').read_text(encoding='utf-8'))['bundle'])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('step',choices=('prepare','status','finish'))
    p.add_argument('--bundle',type=Path); p.add_argument('--per-stratum',type=int,default=10)
    args=p.parse_args()
    if args.step=='prepare': print(prepare(per_stratum=args.per_stratum),flush=True)
    else:
        s=AuditStore(args.bundle or active()); s.check_inputs()
        print(s.finish() if args.step=='finish' else dict(total=len(s.rows),reviewed=sum(bool(r['verdict']) for r in s.rows)),flush=True)

if __name__=='__main__': main()
