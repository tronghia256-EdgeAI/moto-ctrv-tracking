"""Build an isolated ROI-only review from saved human work; preserve originals."""
import argparse
import hashlib
import json
import uuid
import tempfile
import shutil
from pathlib import Path

from code.scripts.gt_mixed_review import BUNDLE, MixedStore, prepare, json_bytes, now
from code.scripts.gt_roi import load_roi, save_roi, contains_box
from code.scripts.gt_review import REPO_ROOT, atomic_write, sha256

POINTER = BUNDLE / 'roi_active.json'


def active_bundle(segment='mixed'):
    if segment!='mixed':
        collection=BUNDLE/'roi_scopes.json'
        if not collection.exists():raise ValueError('Prepare the camera ROI scopes before opening this segment')
        path=Path(json.loads(collection.read_text(encoding='utf-8'))['bundles'][segment]).resolve()
        if (BUNDLE/'roi_reviews').resolve() not in path.parents or not (path/'scope_provenance.json').is_file():
            raise ValueError('Invalid ROI segment pointer')
        return path
    if not POINTER.exists(): return BUNDLE
    path = Path(json.loads(POINTER.read_text(encoding='utf-8'))['bundle']).resolve()
    if (BUNDLE / 'roi_reviews').resolve() not in path.parents or not (path/'scope_provenance.json').is_file():
        raise ValueError('Invalid active ROI review pointer')
    return path


def prepare_camera_scopes(progress=print):
    """Known NKKN camera segments; geometry was checked against their source frames."""
    roi=load_roi(BUNDLE)
    if not roi or not roi['enabled']:raise ValueError('Save the NKKN ROI first')
    current=active_bundle()
    if MixedStore(current).manifest.get('review_scope')!='ROI_ONLY':current=prepare_scope(BUNDLE,progress=progress)
    outputs={'mixed':str(current.resolve())}
    for segment,interval in (('dense',(10700,11300)),('freeflow',(6400,7000))):
        frames=REPO_ROOT/'data/annotations/prelabels'/segment/'img1'
        original=frames.parent/'gt/gt.txt'
        if segment=='dense':
            reviewed=sorted((REPO_ROOT/'experiments/gt_qa/reviewed').glob('*/gt_reviewed.txt'))
            if reviewed:original=reviewed[-1]
        digest=sha256(original)[:16]
        source=REPO_ROOT/'data/processed/gt_roi_sources'/f'{segment}_full_{digest}'
        (source/'gt').mkdir(parents=True,exist_ok=True)
        target=source/'gt/gt.txt'
        if not target.exists():target.write_bytes(original.read_bytes())
        elif sha256(target)!=sha256(original):raise ValueError('Existing full source snapshot changed')
        bridge=REPO_ROOT/'data/processed/gt_roi_parent_reviews'/f'{segment}_{digest}'
        prepare(source,bridge,interval,progress,frames_dir=frames)
        previous=load_roi(bridge)
        if not previous or previous['polygon']!=roi['polygon'] or not previous['enabled']:
            save_roi(bridge,roi['polygon'],True,roi['reviewer'],roi['reason'],roi['reference_frame'],
                     sha256(bridge/'roi.json') if previous else None)
        output=prepare_scope(bridge,progress=progress,activate=False)
        provenance=json.loads((output/'scope_provenance.json').read_text(encoding='utf-8'))
        provenance.update(segment_name=segment,full_source_snapshot=str(original.resolve()),
                          full_source_sha256=sha256(original),camera='NKKN-VoThiSau',
                          camera_scope_basis='Same camera, resolution and field of view checked on source frames')
        (output/'scope_provenance.json').write_bytes(json_bytes(provenance))
        outputs[segment]=str(output.resolve())
    collection=BUNDLE/'roi_scopes.json'
    raw=json_bytes(dict(created_utc=now(),camera='NKKN-VoThiSau',bundles=outputs))
    if collection.exists():atomic_write(collection,raw,sha256(collection))
    else:collection.write_bytes(raw)
    if progress:progress(json.dumps(outputs,indent=2))
    return outputs


def prepare_scope(bundle=BUNDLE, output=None, progress=print, activate=True):
    old = MixedStore(Path(bundle))
    roi = load_roi(old.bundle)
    if not roi or not roi['enabled']:
        raise ValueError('Save an enabled ROI in the ROI window before filtering GT')
    snapshots = {name:(old.bundle/name).read_bytes() for name in
                 ('source_gt.txt','manifest.json','decisions.csv','edits.json','roi.json')}
    digest = hashlib.sha256(b''.join(snapshots[k] for k in sorted(snapshots))).hexdigest()
    output = Path(output) if output else BUNDLE / 'roi_reviews' / ('roi_'+digest[:12])
    if output.exists():
        provenance = json.loads((output/'scope_provenance.json').read_text(encoding='utf-8'))
        if provenance['input_snapshot_sha256'] != digest:raise ValueError('Existing scope has different input work')
        return output
    records, _ = old.preview()
    def in_roi(p):
        x,y,w,h = map(float,p[2:6]);return contains_box([x,y,x+w,y+h],roi)
    selected = {key:p for key,p in records.items() if in_roi(p)}
    if not selected:raise ValueError('ROI contains no GT annotations')
    # Human edits are applied exactly once into the seed. Keep the full history
    # as provenance, not as replay events that could apply those edits twice.
    source = REPO_ROOT / 'data/processed/gt_roi_sources' / ('seed_'+digest[:16])
    source.mkdir(parents=True,exist_ok=True)
    (source/'gt').mkdir(exist_ok=True)
    text = ''.join(','.join(p)+'\n' for _,p in sorted(selected.items()))
    seed = source/'gt/gt.txt'
    if seed.exists() and seed.read_text(encoding='utf-8') != text:
        raise ValueError('ROI source snapshot already exists with different annotations')
    if not seed.exists():seed.write_text(text,encoding='utf-8')
    output.parent.mkdir(parents=True,exist_ok=True)
    container = Path(tempfile.mkdtemp(prefix='.scope-',dir=output.parent))
    staged = container/'bundle'
    try:
        prepare(source,staged,tuple(old.manifest['segment']),progress,frames_dir=old.manifest['frames_dir'])
        finish_scope(old,roi,snapshots,digest,selected,projected_original=None,staged=staged,output=output,activate=activate,progress=progress)
    finally:
        if container.exists() and container.resolve().parent==output.parent.resolve() and container.name.startswith('.scope-'):
            shutil.rmtree(container)
    return output


def finish_scope(old,roi,snapshots,digest,selected,projected_original,staged,output,activate,progress):
    records,_=old.preview()
    manifest=json.loads((staged/'manifest.json').read_text(encoding='utf-8'))
    manifest['review_scope']='ROI_ONLY'
    manifest['source']['seed']='Saved human edits from parent review, filtered by human ROI; inherited geometry is not a new approval'
    (staged/'manifest.json').write_bytes(json_bytes(manifest))
    new = MixedStore(staged)
    save_roi(staged,roi['polygon'],True,roi['reviewer'],roi['reason'],roi['reference_frame'],None)
    new.record_seen(old.journal['seen_frames'])
    def signature(c):
        return json.dumps({k:c[k] for k in ('kind','before','after','proposal')},sort_keys=True)
    previous = {signature(c):(c,r) for c,r in zip(old.manifest['cases'],old.rows) if r['decision']=='KEEP'}
    def in_roi(p):
        x,y,w,h=map(float,p[2:6]);return contains_box([x,y,x+w,y+h],roi)
    projected_original = {key:p for key,p in old.source_rows.items() if in_roi(p)}
    def relevant(data,c):
        return sorted(tuple(p) for (frame,tid),p in data.items()
                      if c['window'][0]<=frame<=c['window'][1] and (not c['ids'] or tid in c['ids']))
    carried = []
    for i,c in enumerate(new.manifest['cases']):
        match = previous.get(signature(c))
        if match and relevant(projected_original,match[0]) == relevant(selected,c):
            new.save(i,'KEEP',match[1]['note']);carried.append(c['rank'])
    for name,raw in snapshots.items():
        if (old.bundle/name).read_bytes()!=raw:
            raise ValueError('Original review/ROI changed during filtering; retry with current work')
    provenance = dict(created_utc=now(),mode='ROI_ONLY_REVIEW',parent_bundle=str(old.bundle.resolve()),
                      input_snapshot_sha256=digest, boxes_before=len(records),boxes_in_roi=len(selected),
                      boxes_excluded=len(records)-len(selected), inherited_manual_edits=len(old.journal['events']),
                      previous_verdicts=old.tally, cases_before=len(old.rows),cases_after=len(new.rows),
                      carried_keep_cases=carried, new_review_counts=new.tally,
                      note='Human geometry/identity edits are inherited as source; only unchanged KEEP verdicts carried. No new human approval inferred.')
    (staged/'scope_provenance.json').write_bytes(json_bytes(provenance))
    archive = staged/'parent_snapshot';archive.mkdir()
    for name,raw in snapshots.items():(archive/name).write_bytes(raw)
    if staged.resolve().parent.name.startswith('.scope-') and staged.resolve().parent.parent==output.parent.resolve():
        staged.rename(output)
    else:raise ValueError('Unsafe scope publication paths')
    if activate:
        POINTER.parent.mkdir(parents=True,exist_ok=True)
        raw=json_bytes(dict(bundle=str(output.resolve()),created_utc=now()))
        if POINTER.exists():atomic_write(POINTER,raw,sha256(POINTER))
        else:POINTER.write_bytes(raw)
    if progress:progress(f'ROI review: {output}; boxes {len(selected)}/{len(records)}; cases {len(new.rows)}; KEEP carried {len(carried)}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--output',type=Path);parser.add_argument('--all-camera-segments',action='store_true')
    args=parser.parse_args()
    print(prepare_camera_scopes() if args.all_camera_segments else prepare_scope(args.bundle,args.output))
