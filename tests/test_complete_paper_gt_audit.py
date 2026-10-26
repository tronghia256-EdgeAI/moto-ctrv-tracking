import pytest
from code.scripts.complete_paper_gt_audit import metadata_only,complete,group_locations
from tests.test_paper_gt_audit import packet

def test_only_pinned_review_metadata_and_comments_are_compatible():
    old=b'from code.scripts.check_seed_bias import DUMPS, MOTO, REVIEWS\nIOU=.5\ndef unmatched():\n return IOU\n'
    new=b'from code.scripts.check_seed_bias import DUMPS, MOTO\n# retired sample\nREVIEWS={"dense":"old release"}\nIOU=.5\ndef unmatched():\n return IOU\n'
    assert metadata_only(old,new)
    assert not metadata_only(old,new.replace(b'IOU=.5',b'IOU=.6'))
    assert not metadata_only(old,new.replace(b'return IOU',b'return 0'))
    assert not metadata_only(old,new.replace(b'{"dense":"old release"}',b'run_code()'))

def test_other_module_imports_must_stay_identical():
    assert not metadata_only(b'from scipy import old\n',b'from scipy import new\n')
    assert not metadata_only(b'import numpy as np\n',b'import numpy as np\nfrom code.scripts.check_seed_bias import REVIEWS as np\n')

def test_removing_a_helper_or_changing_sampling_is_not_metadata():
    old=b'REVIEWS={}\ndef runs():\n return 1\nSEED=42\n'
    assert not metadata_only(old,b'REVIEWS={}\nSEED=42\n')
    assert not metadata_only(old,old.replace(b'SEED=42',b'SEED=43'))

def test_changed_gt_is_not_treated_as_metadata(tmp_path):
    store,source=packet(tmp_path)
    source.write_text('changed GT')
    with pytest.raises(ValueError,match='inputs changed'):complete(tmp_path,'c5e7def')

def test_groups_keep_distinct_frames_and_gt_identities_separate():
    a=dict(n='1',kind='identity',segment='mixed',frame=90,before=80,tid=2)
    b=dict(a,n='2');c=dict(a,n='3',tid=3)
    groups=group_locations([a,b,c])
    assert [g['members'] for g in groups]==[['1','2'],['3']]
