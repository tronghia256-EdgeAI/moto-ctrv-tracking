"""ROI-only review regeneration and masked inference/tracker state isolation."""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_gt_mixed_review import factory,complete_audit
from code.scripts.gt_roi import save_roi,load_roi,contains_box,pixel_mask
from code.scripts.gt_roi_scope import prepare_scope
from code.scripts.gt_mixed_review import MixedStore
from code.pipeline import Pipeline


def roi_for(store):
    return save_roi(store.bundle,[[0,0],[200,0],[200,450],[0,450]],True,'Nghia',
                    'Synthetic readable road scope',1,None)


def test_scope_filters_all_gt_and_preserves_original_human_edits(factory,tmp_path):
    old=factory()
    fields=list(old.source_rows[(2,1)]);fields[2]='150'
    old.add_event(dict(op='upsert',fields=fields,case_rank=1))
    roi_for(old)
    before={n:(old.bundle/n).read_bytes() for n in ('decisions.csv','edits.json','source_gt.txt','manifest.json','roi.json')}
    out=prepare_scope(old.bundle,tmp_path/'roi_review',activate=False,progress=None)
    new=MixedStore(out);roi=load_roi(out)
    assert len(new.source_rows)==4 and (3,3) not in new.source_rows
    assert new.source_rows[(2,1)][2]=='150'
    assert not new.journal['events']  # edits inherited once in seed, not applied twice
    assert all((old.bundle/n).read_bytes()==raw for n,raw in before.items())
    assert all(contains_box([float(p[2]),float(p[3]),float(p[2])+float(p[4]),float(p[3])+float(p[5])],roi) for p in new.source_rows.values())
    assert json.loads((out/'scope_provenance.json').read_text())['inherited_manual_edits']==1
    before_edits=new.edit_path.read_bytes()
    with pytest.raises(ValueError,match='outside'):
        new.add_event(dict(op='upsert',fields=[6,10,600,100,20,40,1,2,1],case_rank=1))
    assert new.edit_path.read_bytes()==before_edits
    assert prepare_scope(old.bundle,out,activate=False,progress=None)==out


def test_scope_only_carries_unchanged_prior_keep_audit(factory,tmp_path):
    old=factory();complete_audit(old);roi_for(old)
    new=MixedStore(prepare_scope(old.bundle,tmp_path/'roi_review',activate=False,progress=None))
    assert next(r for c,r in zip(new.manifest['cases'],new.rows) if c['kind']=='AUDIT')['decision']=='KEEP'
    assert new.journal['seen_frames']==list(range(1,7))


def test_scope_requires_saved_roi(factory,tmp_path):
    old=factory()
    with pytest.raises(ValueError,match='Save an enabled ROI'):
        prepare_scope(old.bundle,tmp_path/'roi_review',activate=False)
    assert not (tmp_path/'roi_review').exists()


def test_roi_pipeline_excludes_detections_before_tracker_and_expires_outside_state():
    pipe=Pipeline.from_config('configs/exp3_C1C2C3.yaml',build_detector=False)
    pipe.roi=dict(enabled=True,image_size=[800,450],polygon=[[0,0],[200,0],[200,450],[0,450]])
    pipe.context_estimator.hour_override=8
    frame=np.zeros((450,800,3),dtype=np.uint8);frame[:,:201]=200
    detections=np.array([[100,100,130,150,.9,1],[600,100,630,150,.9,0]])
    for _ in range(6):output=pipe.process(frame,detections)
    assert output['state'].brightness==pytest.approx(200)
    assert len(pipe.tracker.tracked_stracks)==1
    assert all(contains_box(t.tlbr,pipe.roi) for t in output['tracks_mappo'])
    # Move an existing state outside before the next predict/update; it must
    # disappear from both active and lost pools instead of remaining hidden.
    track=pipe.tracker.tracked_stracks[0]
    if track.ctrv is not None:track.ctrv.x[0]=600
    track.mean[0]=600
    pipe.process(frame,np.empty((0,6)))
    assert all(contains_box(t.tlbr,pipe.roi) for t in pipe.tracker.tracked_stracks+pipe.tracker.lost_stracks)
    with pytest.raises(ValueError,match='resolution'):
        pixel_mask(np.zeros((100,100,3),dtype=np.uint8),pipe.roi)


def test_detector_masks_pixels_and_removes_outside_outputs_before_return():
    from types import SimpleNamespace
    from code.detection.detector import Detector
    class Values:
        def __init__(self,value):self.value=np.array(value)
        def cpu(self):return self
        def numpy(self):return self.value
    class Boxes:
        xyxy=Values([[100,100,130,150],[600,100,630,150]])
        conf=Values([.9,.9]);cls=Values([0,0])
        def __len__(self):return 2
    received=[]
    def model(frame,**kwargs):
        received.append(frame.copy());assert kwargs['conf']==.1
        return [SimpleNamespace(boxes=Boxes())]
    detector=Detector.__new__(Detector);detector.model=model;detector.img_size=640;detector.remap={0:0}
    detector.roi=dict(enabled=True,image_size=[800,450],polygon=[[0,0],[200,0],[200,450],[0,450]])
    source=np.full((450,800,3),200,dtype=np.uint8)
    result=detector.detect(source)
    assert result.shape==(1,6)
    assert np.all(received[0][:,201:]==0) and np.all(received[0][:,:200]==200)
    assert np.all(source==200)


def test_runtime_uses_frozen_active_review_roi_instead_of_mutable_parent(factory,monkeypatch):
    import code.scripts.gt_roi_scope as scope
    from code.scripts.gt_review import sha256
    old=factory();roi_for(old)
    monkeypatch.setattr(scope,'BUNDLE',old.bundle)
    monkeypatch.setattr(scope,'POINTER',old.bundle/'roi_active.json')
    out=scope.prepare_scope(old.bundle,old.bundle/'roi_reviews/test',progress=None)
    save_roi(old.bundle,[[0,0],[50,0],[50,450],[0,450]],True,'Nghia','New parent draft',1,sha256(old.bundle/'roi.json'))
    pipe=Pipeline.from_config('configs/mixed_roi.yaml',build_detector=False)
    assert pipe.roi['manifest_sha256']==sha256(out/'manifest.json')
    assert pipe.roi['polygon'][1][0]==200
