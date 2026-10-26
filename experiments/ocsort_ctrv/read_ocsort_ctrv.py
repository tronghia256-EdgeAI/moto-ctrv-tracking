"""Post-hoc numbers of the pre-registered OC-SORT+C2 reading (evidence.md):
the paired moving-block bootstrap of the UNcorrected mixed motorcycle IDF1
difference ocsort_matched_ctrv - ocsort_matched (30-frame blocks, 10,000
replicates, seed [20261013, 30], the runner's own block_table/paired_ci), plus
the secondary dense/freeflow differences and the exact-box diagnostic from the
run's metrics.json. The seed-corrected bracket comes from seed_bracket.

Run from the run's own tree (../CAAP-ocsort):
    PYTHONPATH=. python <this file> <run>/<size> <out.json>
"""
import json
import sys
from pathlib import Path

from code.scripts import run_roi_paper as run
from code.scripts.eval_hota_ci import block_table, frame_data

NEW, REF, LENGTH = 'ocsort_matched_ctrv', 'ocsort_matched', 30

directory, out = Path(sys.argv[1]), Path(sys.argv[2])
protocol = json.loads((directory.parent / 'protocol.json').read_text(encoding='utf-8'))
metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
item = protocol['segments']['mixed']
start, end = item['segment']
n = end - start
truth = run.load_mot(item['gt'])
cid = run.CLASSES['motorcycle']
tables = {}
for name in (NEW, REF):
    preds = run.tracks_json_slice(directory / f'tracks_{name}.json', start, end)
    tables[name] = block_table(frame_data(run.to_framelist(truth, n, cid), run.to_framelist(preds, n, cid)), LENGTH)
ci = run.paired_ci(tables[NEW], tables[REF], LENGTH, n)

secondary = {}
for segment in ('mixed', 'dense', 'freeflow'):
    m = {name: metrics['segments'][segment]['metrics'][name]['motorcycle'] for name in (NEW, REF)}
    secondary[segment] = dict(idf1={name: m[name]['idf1'] for name in m}, idf1_diff=m[NEW]['idf1'] - m[REF]['idf1'],
                              idsw={name: m[name]['num_switches'] for name in m},
                              exact_box_share={name: m[name]['exact_box_share'] for name in m})
result = dict(run=str(directory), pair=f'{NEW} - {REF}', mixed_block_ci=dict(block_frames=LENGTH, seed=[20261013, LENGTH],
              replicates=10000, **ci), secondary=secondary)
out.write_text(json.dumps(result, indent=1), encoding='utf-8')
print(json.dumps(result, indent=1))
