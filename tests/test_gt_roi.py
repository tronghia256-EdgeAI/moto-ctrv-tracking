"""ROI geometry, symmetric scoring and preservation of annotation work."""
import json
from unittest.mock import patch

import pytest
from PIL import Image

from tests.test_gt_mixed_review import factory, complete_audit
from code.scripts.gt_roi import validate_polygon, contains_box, load_roi, save_roi, RULE
from code.scripts.gt_mixed_review import finalize_mixed
from code.scripts.gt_review import sha256, score_variants


def save(store, polygon=None, expected=None, enabled=True):
    return save_roi(store.bundle,polygon or [[0,0],[200,0],[200,450],[0,450]],enabled,'Nghĩa',
                    'Observable road region; synthetic test only',1,expected)


def test_membership_bottom_center_includes_boundary_not_center_or_overlap():
    roi=dict(enabled=True,polygon=[[0,50],[100,50],[100,100],[0,100]],rule=RULE)
    assert contains_box([20,0,40,60],roi)  # center outside, footpoint inside
    assert not contains_box([20,60,40,120],roi)  # center inside, footpoint outside
    assert contains_box([80,80,120,100],roi)  # bottom point on boundary
    assert contains_box([200,200,210,210],None)
    assert contains_box([200,200,210,210],dict(enabled=False))


@pytest.mark.parametrize('polygon',[
    [[0,0],[10,10]], [[0,0],[10,10],[0,10],[10,0]],
    [[0,0],[10,0],[0,0]], [[-1,0],[10,0],[0,10]],
    [[0,0],[float('nan'),0],[0,10]], [[0,0],[10,0],[20,0]],
])
def test_invalid_polygons_are_rejected(polygon):
    with pytest.raises(ValueError):validate_polygon(polygon,[100,100])


def test_roi_persistence_cas_and_binding_preserve_gt_review(factory):
    store=factory()
    original={name:(store.bundle/name).read_bytes() for name in ('source_gt.txt','decisions.csv','edits.json','manifest.json')}
    roi=save(store)
    assert load_roi(store.bundle)['polygon']==roi['polygon']
    with pytest.raises(FileExistsError):save(store)
    fingerprint=sha256(store.bundle/'roi.json')
    updated=save(store,expected=fingerprint,enabled=False)
    assert len(updated['history'])==1 and not updated['enabled']
    with pytest.raises(ValueError):save(store,expected=fingerprint)
    assert all((store.bundle/name).read_bytes()==raw for name,raw in original.items())
    raw=json.loads((store.bundle/'roi.json').read_text());raw['image_size']=[900,450]
    (store.bundle/'roi.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='different camera'):load_roi(store.bundle)


def dumps(store,tmp_path):
    tracks=tmp_path/'tracks';tracks.mkdir()
    frames=[[] for _ in range(6)]
    for (frame,tid),p in store.source_rows.items():
        x,y,w,h=map(float,p[2:6]);frames[frame-1].append([tid,int(p[7])-1,[x,y,x+w,y+h]])
    frames[0].append([999,0,[600,100,630,140]]) # outside-ROI false positive
    (tracks/'tracks_baseline.json').write_text(json.dumps(dict(start_frame=0,frames=frames)))
    return tracks


def test_export_preserves_full_gt_and_filters_both_gt_and_predictions(factory,tmp_path):
    store=factory();complete_audit(store);save(store)
    source=(store.bundle/'source_gt.txt').read_bytes()
    tracks=dumps(store,tmp_path)
    full=score_variants(store.source_path,tracks,(0,6))
    assert full['baseline']['metrics']['ALL']['mota']<1
    output=finalize_mixed(store,'Nghĩa',tracks,tmp_path/'result')
    report=json.loads((output/'review.json').read_text())
    assert report['evaluation_scope']=='ROI'
    assert report['roi_gt']['boxes_included']==4
    assert report['roi_gt']['boxes_excluded']==1
    assert report['variants']['baseline']['metrics']['ALL']['mota']==1
    assert report['variants']['baseline']['roi_counts']['predictions_excluded']==2
    assert set((output/'gt_reviewed.txt').read_text().splitlines())==set(source.decode().splitlines())
    assert len((output/'gt_roi.txt').read_text().splitlines())==4
    assert (store.bundle/'source_gt.txt').read_bytes()==source
    assert (output/'roi.json').read_bytes()==(store.bundle/'roi.json').read_bytes()


def test_roi_cannot_approve_unfinished_review(factory,tmp_path):
    store=factory();save(store)
    with pytest.raises(ValueError,match='verdict'):
        finalize_mixed(store,'Nghĩa',out_dir=tmp_path/'result',evaluate=False)
    assert store.tally['PENDING']==1


def test_roi_change_during_scoring_blocks_publication(factory,tmp_path):
    store=factory();complete_audit(store);save(store)
    def changed(*args,**kwargs):
        with (store.bundle/'roi.json').open('ab') as f:f.write(b'\n')
        return {}
    with patch('code.scripts.gt_mixed_review.score_variants',side_effect=changed):
        with pytest.raises(ValueError,match='ROI changed'):
            finalize_mixed(store,'Nghĩa',out_dir=tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_overlay_does_not_change_source_image():
    from code.scripts.gt_roi import overlay_roi
    image=Image.new('RGB',(100,100),'white');before=image.tobytes()
    rendered=overlay_roi(image,dict(enabled=True,polygon=[[0,50],[100,50],[100,100],[0,100]]))
    assert image.tobytes()==before
    assert rendered.getpixel((25,25))!=image.getpixel((25,25))
    assert rendered.getpixel((25,75))==image.getpixel((25,75))
