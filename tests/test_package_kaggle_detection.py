"""Portable upload contents, notebook preprocessing and integrity guards."""
import ast
import json
import zipfile
from pathlib import Path

import pytest

from tests.test_clean_detection_dataset import dataset
from code.scripts.audit_detection_dataset import audit
from code.scripts.clean_detection_dataset import CleanStore, prepare
from code.scripts.kaggle_detection_helpers import NAMES, verify_package
from code.scripts.package_kaggle_detection import package


@pytest.fixture
def upload(dataset, tmp_path):
    source, _, _, _ = dataset
    (source / 'data.yaml').write_text('nc: 5\nnames: [Accident, bus, car, motorcycle, truck]\n')
    report, bundle = tmp_path / 'audit5', tmp_path / 'bundle5'
    audit(source, report, workers=1)
    prepare(source, report, bundle, progress=None)
    weights = tmp_path / 'yolo11n.pt'; weights.write_bytes(b'Fixture weights; not loaded as a model')
    output = tmp_path / 'upload/caap_accident_vehicle_dev_fixture'
    archive = package(output, bundle, weights)
    return archive, output, CleanStore(bundle)


def test_upload_zip_portable_and_excludes_local_review(upload, tmp_path):
    archive, output, store = upload
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert not any('review_snapshot' in p or '/smoke/' in p or '/runs/' in p for p in names)
        assert all(not p.startswith('/') and '..' not in Path(p).parts for p in names)
        z.extractall(tmp_path / 'extracted')
    moved = tmp_path / 'extracted' / output.name
    report, rows = verify_package(moved)
    assert report['names'] == NAMES
    assert len(rows) == 12
    assert all(len(Path(r['image']).name) < 40 for r in rows)
    assert store.tally['metadata_confirmed'] == 0
    assert store.tally['pending'] > 0


def test_uploaded_label_tampering_blocks_training(upload):
    _, output, _ = upload
    label = next((output / 'train/labels').glob('*.txt'))
    label.write_text('0 .5 .5 .2 .2\n')
    with pytest.raises(ValueError, match='Image/label changed'):
        verify_package(output)


def test_notebook_code_and_data_loading_without_network_gpu(upload, tmp_path, capsys):
    _, output, _ = upload
    notebook = json.loads(Path('notebooks/CAAP_YOLO11_Kaggle.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in code:
        ast.parse(source)
    loader = next(source for source in code if 'checks.verify_package' in source)
    work = tmp_path / 'kaggle_working'; work.mkdir()
    namespace = dict(Path=Path, WORK=work, INPUT_ROOT=tmp_path / 'input', PACKAGE_ROOT=output,
                     DATA_ROOT=work / 'caap_dataset')
    exec(compile(loader, '<notebook-data-loader>', 'exec'), namespace)
    assert len(namespace['MANIFEST']) == 12
    assert namespace['REPORT']['names'] == NAMES
    assert len(namespace['DATASET_HASH']) == 64
    messages = capsys.readouterr().out
    assert 'Copying dataset to working' in messages
    assert 'Checked 1/12 images' in messages
    assert 'Dataset verification complete: 12 images' in messages
    assert all(not c.get('outputs') for c in notebook['cells'])
