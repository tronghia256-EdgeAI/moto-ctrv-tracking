"""Create a private Kaggle upload ZIP; does not train or upload anything."""
import argparse
import os
import shutil
import uuid
import zipfile
from pathlib import Path

import yaml

from code.scripts.clean_detection_dataset import BUNDLE, REPO, CleanStore, payload, stamp
from code.scripts.gt_review import sha256
from code.scripts.train_detection_trial import prepare_trial
from code.scripts.kaggle_detection_helpers import verify_package

DEFAULT = REPO / 'data/processed/kaggle/caap_accident_vehicle_dev_20261011'


def package(output=DEFAULT, bundle=BUNDLE, weights=REPO / 'weights/yolo11n.pt'):
    output, weights = Path(output).resolve(), Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    prepare_trial(CleanStore(bundle), output)
    # Keep smoke/local review snapshots out of the upload, without removing them locally.
    shutil.copy2(weights, output / 'yolo11n.pt')
    helper = Path(__file__).with_name('kaggle_detection_helpers.py')
    shutil.copy2(helper, output / 'kaggle_detection_helpers.py')
    config = {'path': '.', 'train': 'train/images', 'val': 'val/images', 'nc': 5,
              'names': ['Accident', 'bus', 'car', 'motorcycle', 'truck']}
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    metadata = ['manifest.csv', 'report.json', 'skipped.csv', 'data.yaml', 'yolo11n.pt', 'kaggle_detection_helpers.py']
    receipt = {'created': stamp(), 'file_hashes': {name: sha256(output / name) for name in metadata},
               'purpose': 'Private Kaggle development training; incomplete human review; no independent test'}
    (output / 'package_receipt.json').write_bytes(payload(receipt))
    report, _ = verify_package(output)
    temporary = archive.with_name(archive.name + '.' + uuid.uuid4().hex[:8] + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name in metadata + ['package_receipt.json']:
                z.write(output / name, output.name + '/' + name)
            for split in ('train', 'val'):
                for kind in ('images', 'labels'):
                    for path in sorted((output / split / kind).glob('*')):
                        z.write(path, output.name + '/' + path.relative_to(output).as_posix())
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise ValueError('ZIP CRC validation failed')
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Upload this ZIP as a private Kaggle Dataset: {archive}', flush=True)
    print(f'ZIP SHA256: {sha256(archive)}; bytes: {archive.stat().st_size}', flush=True)
    print(report['counts']['trial'], flush=True)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--weights', type=Path, default=REPO / 'weights/yolo11n.pt')
    args = parser.parse_args()
    package(args.output, args.bundle, args.weights)
