import csv
import hashlib
import json
import pytest
import numpy as np
from code.scripts.paper_gt_audit import AuditStore, encoded, FIELDS, select_indices, gt_issue, checkpoint

def packet(tmp_path):
    source=tmp_path/'gt.txt';source.write_text('untouched source')
    cases=[dict(n='1',type='box'),dict(n='2',type='identity'),dict(n='3',type='box')]
    keys=[dict(n='1',kind='extra'),dict(n='2',kind='identity'),dict(n='3',kind='missing')]
    protocol=dict(inputs={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                  cases_sha256=hashlib.sha256(encoded(cases)).hexdigest(),key_sha256=hashlib.sha256(encoded(keys)).hexdigest())
    for name,data in [('protocol.json',protocol),('cases.json',cases),('key.json',keys)]: (tmp_path/name).write_bytes(encoded(data))
    with (tmp_path/'verdicts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(dict(n=c['n'],verdict='',note='',reviewer='',updated_utc='') for c in cases)
    return AuditStore(tmp_path),source

def test_saved_verdict_roundtrip_and_stale_window_cannot_overwrite(tmp_path):
    s,source=packet(tmp_path);other=AuditStore(tmp_path)
    s.save(0,'Y','Ghi chú, nhiều dòng\nsecond line')
    assert AuditStore(tmp_path).rows[0]['note']=='Ghi chú, nhiều dòng\nsecond line'
    with pytest.raises(ValueError,match='changed'):other.save(0,'N')
    assert AuditStore(tmp_path).rows[0]['verdict']=='Y'
    assert source.read_text()=='untouched source'

def test_fixed_sample_completion_preserves_gt_and_flags_uncertainty(tmp_path):
    s,source=packet(tmp_path)
    s.save(0,'Y')
    with pytest.raises(ValueError):s.finish()
    with pytest.raises(ValueError):s.save(1,'D')
    s.save(1,'?');s.save(2,'Y')
    result=s.finish()
    assert result['status']=='HUMAN_RECONCILIATION_REQUIRED'
    assert len(result['issues'])==2
    assert source.read_text()=='untouched source'
    assert not result['independent_test_registered']

def test_binding_change_blocks_completion_and_corrupt_inventory_rejected(tmp_path):
    s,source=packet(tmp_path);source.write_text('modified')
    with pytest.raises(ValueError,match='đổi'):s.check_inputs()
    (tmp_path/'cases.json').write_text('[]')
    with pytest.raises(ValueError,match='inventory'):AuditStore(tmp_path)

def test_sample_is_reproducible_unique_and_capped():
    a=select_indices(100,10,np.random.default_rng(42));b=select_indices(100,10,np.random.default_rng(42))
    assert a==b and len(set(a))==10
    assert select_indices(0,10,np.random.default_rng(42))==[]
    assert select_indices(3,10,np.random.default_rng(42))==[0,1,2]

def test_uploaded_checkpoint_can_be_used_after_user_moves_root_copy(tmp_path):
    uploaded=tmp_path/'ket_qua_train/best.pt';uploaded.parent.mkdir();uploaded.write_bytes(b'model')
    assert checkpoint(tmp_path)==uploaded
    previous=tmp_path/'best.pt';previous.write_bytes(b'model')
    assert checkpoint(tmp_path)==uploaded
    previous.write_bytes(b'different')
    with pytest.raises(ValueError,match='khác'):checkpoint(tmp_path)

@pytest.mark.parametrize('kind,verdict,issue',[('extra','Y',True),('extra','D',False),('missing','N',True),('missing','D',True),('identity','N',True),('identity','Y',False)])
def test_reconciliation_targets_gt_errors_not_tracker_errors(kind,verdict,issue):
    assert gt_issue(kind,verdict)==issue
