"""Freeze 20 old-Train videos; held-out records supply identities only."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_grpo_group_selection import ROOT, OUTS, digest, save

SPLIT = OUTS / 'agent_split_v1_20260905_action_aware_v3_review_conflicts'
TEACHER = OUTS / 'thin_teacher_v3_all_downloaded_20260928'


def jsonl(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def identity(row, merged):
    sid = (row.get('merged_id') or row.get('id') or '').split('::')[0]
    raw = merged.get(sid, {})
    keys = set()
    for k in ['sha256', 'source_group', 'source_component', 'parent_video_sha256']:
        for obj in [row, raw]:
            if obj.get(k): keys.add((k, str(obj[k])))
    if sid: keys.add(('id', sid))
    for obj in [row, raw]:
        path = obj.get('video_path') or obj.get('local_path')
        if path:
            p = (ROOT / path).resolve()
            keys.add(('path', str(p)))
            # The suffix after a content-hash prefix is the upstream media name.
            keys.add(('media_name', p.name.split('__', 1)[-1]))
    for source in raw.get('sources', []):
        for k in ['source_group', 'parent_video_sha256']:
            if source.get(k): keys.add((k, str(source[k])))
        if source.get('source_path'):
            keys.add(('upstream_path', str((ROOT / source['source_path']).resolve())))
        if source.get('source_id'):
            keys.add(('upstream_id', str(source.get('source_dataset')) + ':' + str(source['source_id'])))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--exclude-selection', type=Path, action='append', default=[])
    ap.add_argument('--selection-salt', default='legacy-pilot-20261102')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    merged_path = ROOT / 'data/merged_video_safety_v1/manifest.jsonl'
    merged = {r['id']: r for r in jsonl(merged_path)}
    train = jsonl(SPLIT / 'train.jsonl')
    canonical_path = TEACHER / 'canonical_index.jsonl'
    canonical = {r['id']: r for r in jsonl(canonical_path)}
    frozen = {str(p): digest(p) for p in [SPLIT/'train.jsonl', SPLIT/'dev.jsonl',
              SPLIT/'test.jsonl', merged_path, canonical_path]}
    excluded = set()
    exclusion_files = []
    for name in ['dev', 'test']:
        p = SPLIT / (name+'.jsonl')
        for row in jsonl(p): excluded.update(identity(row, merged))
        exclusion_files.append({'path': str(p), 'sha256': frozen[str(p)], 'purpose': 'IDENTITIES_ONLY'})
    names = [
        'grpo_rapid128_live_20261027_v1/shard-01_stratified32/MANIFEST128.json',
        'grpo_rapid128_live_20261027_v1/shard-00/MANIFEST128.json',
        'grpo32_initial_screened_20261023_v1/MANIFEST128.json',
        'next_iteration_train_screen128_20261026_v1/MANIFEST32.json',
        'grpo_rapid128_live_20261027_v1/holdout120_v1/SAMPLES120.json',
        'grpo_group_selection_20261030_final_v1/SELECTION.json',
        'grpo_teacher_category_validation_20261101_v1/paired_sft/SELECTION.json',
        'grpo_candidate_screen_20261029_v1/SOURCE_SELECTION48.jsonl',
    ]
    for name in names:
        p = OUTS / name
        values = jsonl(p) if p.suffix == '.jsonl' else json.loads(p.read_text())
        for row in values: excluded.update(identity(row.get('source', row), merged))
        exclusion_files.append({'path': str(p), 'sha256': digest(p), 'purpose': 'IDENTITIES_ONLY'})
    for p in args.exclude_selection:
        p = p.resolve()
        for row in json.loads(p.read_text()): excluded.update(identity(row.get('source', row), merged))
        exclusion_files.append({'path': str(p), 'sha256': digest(p), 'purpose': 'IDENTITIES_ONLY'})
    thvl = OUTS / 'thvl_current_protocol_20261009/inputs.jsonl'
    for row in jsonl(thvl): excluded.update(identity(row, merged))
    exclusion_files.append({'path': str(thvl), 'sha256': digest(thvl), 'purpose': 'IDENTITIES_ONLY'})
    eligible, quarantine = [], []
    for row in train:
        reason = None
        keys = identity(row, merged)
        teacher = canonical.get('thin-all-' + row['sha256'][:20])
        raw = merged.get(row['id'])
        if keys & excluded: reason = 'HELDOUT_OR_PRIOR_PILOT_SOURCE_OVERLAP'
        elif raw is None or not keys or not row.get('source_component'): reason = 'INCOMPLETE_SOURCE_IDENTITY'
        elif row.get('severity_mode') not in {'clean', 'boundary'}: reason = 'AMBIGUOUS_OLD_REFERENCE'
        elif teacher is None or teacher.get('status') != 'ANSWERED': reason = 'NO_COMPLETE_AUTONOMOUS_TEACHER'
        elif teacher.get('frozen_split') != 'train': reason = 'TEACHER_SPLIT_NOT_TRAIN'
        elif teacher.get('category') != row['category']: reason = 'OLD_LABEL_VS_AUTONOMOUS_TEACHER_CATEGORY_CONFLICT'
        elif row['severity_mode'] == 'clean' and teacher.get('severity') != row['severity']:
            reason = 'CLEAN_LEVEL_VS_AUTONOMOUS_TEACHER_CONFLICT'
        elif not (ROOT / row['local_path']).is_file(): reason = 'MEDIA_MISSING'
        if reason:
            quarantine.append({'id': row['id'], 'sha256': row['sha256'], 'reason': reason})
        else:
            eligible.append((row, teacher, keys))
    selected, used = [], set()
    categories = ['Harmless', 'Sexual Content', 'Violence & Dangerous', 'Sensitive Figures & Ideologies']
    for category in categories:
        cells = [('L0', 0), ('L0', 1), ('L0', 0), ('L0', 1), ('L0', 0)] if category == 'Harmless' else [
                 ('L1', 0), ('L1', 1), ('L2', 0), ('L2', 1), ('L2', 2)]
        for level, requested_tools in cells:
            options = [(r,t,k) for r,t,k in eligible if r['category'] == category and r['severity'] == level
                       and not k & used]
            def order(value):
                r,t,k = value
                # Route hint is historic metadata, not a claim about difficulty.
                return (abs(int(r.get('teacher_tool_rounds', 0))-requested_tools),
                        level == 'L1' and r['severity_mode'] != 'boundary',
                        float(r['duration_seconds']) > 180,
                        hashlib.sha256((args.selection_salt+':'+r['id']).encode()).hexdigest())
            options.sort(key=order)
            if not options: raise ValueError('Insufficient safe old-Train coverage: '+category+'/'+level)
            r, teacher, keys = options[0]
            used.update(keys)
            if digest(ROOT/r['local_path']) != r['sha256']: raise ValueError('Media hash mismatch')
            path, line = Path(teacher['record_file']), int(teacher['record_line'])
            with path.open('rb') as handle:
                data = next(v for n,v in enumerate(handle,1) if n == line)
            obj = json.loads(data)
            final = obj['result']['final_action']
            if obj['id'] != teacher['id'] or final['category'] != r['category']:
                raise ValueError('Teacher index/line conflict')
            gi = len(selected)
            source = {**r, 'merged_id': r['id'], 'video_path': str((ROOT/r['local_path']).resolve()),
                'mapped_category': r['category'], 'frozen_split': 'train',
                'source_group': r['source_component'], 'teacher_id': teacher['id'],
                'teacher_model': obj['model'], 'teacher_category': final['category'],
                'teacher_severity': final['severity'], 'teacher_action': final['moderation_action'],
                'teacher_record_file': str(path), 'teacher_record_line': line,
                'teacher_record_sha256': hashlib.sha256(data).hexdigest(),
                'reference_category': r['category'], 'reference_severity': r['severity'],
                'reference_boundary': r['severity_mode'] == 'boundary',
                'reference_role': 'old-frozen-Train-weak-target'}
            seed_offset = int(hashlib.sha256(args.selection_salt.encode()).hexdigest()[:8],16) * 100
            samples = [{'source': source, 'sample': i, 'seed': seed_offset+gi*100+i} for i in range(4)]
            selected.append({'group': gi, 'source': source, 'samples': samples,
                'rollout_dir': str(out/'student_rollouts'/f'group-{gi:02d}'),
                'category_stratum': category, 'severity_stratum': r['severity'],
                'historic_route_hint': r.get('tool_signature'), 'difficulty': '待验证',
                'difficulty_hypotheses': ['历史动作提示可能直接回答或需要搜索/补帧，未据此认定难度'],
                'requested_historic_tool_rounds': requested_tools,
                'reuse_existing_complete_group': False, 'selection_did_not_read_student_answers': True})
    # Reuse the exact already-bound SFT identity, not the former small training pool.
    previous = json.loads((OUTS/'grpo_teacher_category_validation_20261101_v1/paired_sft/PLAN.json').read_text())
    plan = {k: previous[k] for k in ['model_binding', 'model_path', 'checkpoint_path', 'policy_path',
                                    'policy_sha256', 'locator_model_path']}
    plan.update(status='FROZEN_LEGACY_TRAIN_PILOT_NO_UPDATE', source_manifest=str(SPLIT/'train.jsonl'),
        train_manifest_sha256=digest(SPLIT/'train.jsonl'), videos=20, samples_per_video=4, rollouts=80,
        initial_frames=4, max_frames=None, max_tool_calls=None, max_decisions=None,
        max_search_new_frames=None, source_labels_in_model_input=False, optimizer_steps=0,
        reward_version='outcome-cost-v1-draft', reward_enabled=False, training_authorized=False,
        compute_dtype='float32', head_precision='float32', request_seed_schedule='disjoint-request-v2',
        exclusion_files=exclusion_files, fixed120_used_for_identity_exclusion_only=True,
        dev_test_used_for_identity_exclusion_only=True, frozen_source_hashes=frozen,
        eligible_pool_size=len(eligible), quarantined_pool_size=len(quarantine),
        completion_scope='80 real rollouts, offline reward checks and traceable report; no optimizer')
    save(out/'PLAN.json', plan)
    save(out/'SELECTION.json', selected)
    save(out/'QUARANTINE.json', quarantine)
    save(out/'SOURCE_AUDIT.json', {'old_train_size': len(train), 'eligible': len(eligible),
        'quarantined': len(quarantine), 'reasons': dict(Counter(q['reason'] for q in quarantine)),
        'selected': len(selected), 'selected_overlap': 0, 'source_hashes': frozen,
        'coverage': dict(Counter(s['category_stratum']+'/'+s['severity_stratum'] for s in selected)),
        'severity_modes': dict(Counter(s['source']['severity_mode'] for s in selected)),
        'identity_limit': 'Recorded hashes, paths, source groups/components/parents and upstream IDs; unknown re-encodings not excluded with certainty'})
    print(json.dumps({'out': str(out), 'old_train': len(train), 'eligible': len(eligible),
                      'selected': len(selected), 'quarantined': len(quarantine)}, ensure_ascii=False))


if __name__ == '__main__': main()
