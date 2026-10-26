"""Finish an audit, verifying narrowly scoped legacy-source metadata changes."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from code.scripts.paper_gt_audit import AuditStore,ROOT,active,encoded,gt_issue
from code.scripts.gt_review import sha256
from code.scripts.gt_review_queue import _iou

LEGACY=ROOT/'code/scripts/audit_disagreements.py'

def saved_source_ref(bundle):
    p=Path(bundle)/'audit_result.json'
    if not p.exists():return None
    proof=json.loads(p.read_text(encoding='utf-8')).get('input_validation',{})
    changes=proof.get('changes',[])
    return changes[0].get('original_git_ref') if proof.get('mode')=='VERIFIED_LEGACY_METADATA_CHANGE' and len(changes)==1 else None

def group_locations(issues):
    """Group repeated views, never infer/apply GT corrections from the grouping."""
    groups=[]
    for issue in issues:
        def same(reference):
            if (reference['segment'],reference['kind'],reference['frame'])!=(issue['segment'],issue['kind'],issue['frame']):return False
            if issue['kind']=='identity':return (reference['tid'],reference['before'])==(issue['tid'],issue['before'])
            return _iou(reference['box'],issue['box'])>=0.65
        found=next((g for g in groups if same(g['reference'])),None)
        if found is None:groups.append(dict(reference=issue,members=[issue['n']]))
        else:found['members'].append(issue['n'])
    return groups

def metadata_only(old: bytes,new: bytes) -> bool:
    """Allow only moving REVIEWS from an import to its pinned assignment.

    This variable is not read by the four helpers used to prepare this packet.
    Every other executable statement (including helper definitions) must match.
    """
    def normalized(raw):
        tree=ast.parse(raw.decode('utf-8-sig'))
        body=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='REVIEWS':
                # The pinned metadata must be a literal: no executable expressions.
                ast.literal_eval(node.value)
                continue
            if isinstance(node,ast.ImportFrom) and node.module=='code.scripts.check_seed_bias':
                if any(n.name=='REVIEWS' and n.asname not in (None,'REVIEWS') for n in node.names):
                    raise ValueError('REVIEWS import must not replace a helper dependency')
                node.names=[n for n in node.names if n.name!='REVIEWS']
            body.append(node)
        tree.body=body
        return ast.dump(tree,include_attributes=False)
    try:return normalized(old)==normalized(new)
    except (ValueError,SyntaxError):return False

def validate(store,source_ref=None):
    changed=[]
    for path,expected in store.protocol['inputs'].items():
        p=Path(path)
        actual=sha256(p) if p.is_file() else None
        if actual!=expected:changed.append((p,expected,actual))
    if not changed:return dict(mode='EXACT_INPUT_HASHES',changes=[])
    if len(changed)!=1 or changed[0][0].resolve()!=LEGACY.resolve() or not source_ref:
        raise ValueError('Audit inputs changed. Verify the original source version; do not resample completed verdicts.')
    p,expected,current=changed[0]
    old=subprocess.check_output(['git','show',f'{source_ref}:code/scripts/audit_disagreements.py'],cwd=ROOT)
    if hashlib.sha256(old).hexdigest()!=expected:
        raise ValueError('Git source version does not match the frozen original hash')
    if not metadata_only(old,p.read_bytes()):
        raise ValueError('Executable audit logic changed; metadata compatibility is not established')
    return dict(mode='VERIFIED_LEGACY_METADATA_CHANGE',changes=[dict(path=str(p),original_sha256=expected,
                current_sha256=current,original_git_ref=source_ref,
                proof='Only literal REVIEWS metadata/import changed; all other executable AST nodes match. GT, cases and image inputs unchanged.')])

def complete(bundle,source_ref=None):
    store=AuditStore(bundle)
    if (store.bundle/'protocol.json').read_bytes()!=store.protocol_raw:raise ValueError('Protocol changed')
    proof=validate(store,source_ref)
    if any(not r['verdict'] for r in store.rows):raise ValueError('Audit is incomplete')
    raw=(store.bundle/'key.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=store.protocol['key_sha256']:raise ValueError('Private key changed')
    keys=json.loads(raw)
    if [k['n'] for k in keys]!=[c['n'] for c in store.cases]:raise ValueError('Private/public inventory mismatch')
    for case in store.cases:
        for name,digest in zip(case['images'],case['image_sha256']):
            p=store.bundle/name
            if p.resolve().parent!=store.bundle.resolve() or sha256(p)!=digest:raise ValueError('Rendered audit image changed')
    issues=[dict(k,**{f:v[f] for f in ('verdict','note','reviewer','updated_utc')}) for k,v in zip(keys,store.rows)
            if gt_issue(k['kind'],v['verdict']) or v['verdict']=='?']
    result=dict(status='HUMAN_RECONCILIATION_REQUIRED' if issues else 'SAMPLED_AUDIT_COMPLETE',
                sample_size=len(keys),issues=issues,verdicts_sha256=store.fingerprint,input_validation=proof,
                limitations='Completed sampled audit of old tracker disagreements. Confirmed sampled GT issues require reconciliation; no GT changed. Not independent test.',
                independent_test_registered=False)
    groups=group_locations(issues)
    result['location_groups']=[dict(case_numbers=g['members'],segment=g['reference']['segment'],kind=g['reference']['kind']) for g in groups]
    result['grouping_rule']='Same segment/kind/frame; identity uses GT ID and before frame; box views use IoU>=0.65. Grouping is a review aid, not proof of one object.'
    if sha256(store.path)!=store.fingerprint:raise ValueError('Verdicts changed during completion')
    if (store.bundle/'protocol.json').read_bytes()!=store.protocol_raw or validate(store,source_ref)!=proof:
        raise ValueError('Audit inputs changed during completion')
    collection_path=next(Path(p) for p in store.protocol['inputs'] if Path(p).name=='completed_roi_collection.json')
    collection=json.loads(collection_path.read_text(encoding='utf-8'))
    lines=['# Audit đã duyệt: các vị trí cần xử lý GT','',
           f"Đã lưu {len(keys)}/{len(keys)} mục. {len(issues)} mục báo lỗi GT hoặc chưa rõ, gợi ý {len(groups)} nhóm vị trí để đối chiếu. GT chưa bị thay đổi.",'',
           'Các câu trả lời xác nhận ảnh được lấy mẫu, không xác nhận toàn bộ run hoặc quyết định ID mới. Cần xem video quanh vị trí trước khi sửa track/điền các frame khác.','',
           '| Mục audit | Segment / MOT frame | GT ID hoặc ID của tracker | Việc cần xử lý | Ảnh |',
           '|---|---|---|---|---|']
    for issue in issues:
        start=collection['segments'][issue['segment']]['report']['segment_zero_based_end_exclusive'][0]
        frame=issue['frame']-start+1
        span=f"{issue['before']-start+1} → {frame}" if issue['kind']=='identity' else str(frame)
        action={'extra':'Đối chiếu xe thiếu nhãn/box không khớp; chọn đúng GT ID trước khi thêm.',
                'missing':'Sửa/xóa box GT hoặc xác minh box trùng.',
                'identity':'Xem đoạn nối ID; tách nếu là hai xe khác nhau.'}[issue['kind']]
        if issue['verdict']=='?':action='Cần xác minh thêm, chưa kết luận.'
        case=store.cases[int(issue['n'])-1]
        lines.append(f"| #{issue['n']} ({issue['verdict']}) | {issue['segment']} / {span} | {issue['tid']} | {action} | [Toàn khung]({case['images'][1]}) |")
    repeated=[', '.join('#'+n for n in g['members']) for g in groups if len(g['members'])>1]
    lines+=['','Các mục lặp/box chồng cùng vị trí cần đối chiếu chung: '+('; '.join(repeated) or 'không có')+'.',
            'Gộp nhóm chỉ hỗ trợ điều hướng, không tự chứng minh một xe hay tự áp dụng thao tác GT.',
            'Mọi kết quả vẫn ở nhóm development; camera test độc lập chưa được đăng ký.']
    (store.bundle/'audit_result.json').write_bytes(encoded(result))
    (store.bundle/'issues.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path);p.add_argument('--source-ref')
    args=p.parse_args();bundle=args.bundle or active();result=complete(bundle,args.source_ref)
    print(json.dumps(dict(bundle=str(bundle),status=result['status'],reviewed=result['sample_size'],issues=len(result['issues']),location_groups=len(result['location_groups']),validation=result['input_validation']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
