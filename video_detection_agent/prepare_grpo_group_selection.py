"""Freeze a Train-only selection pilot; evaluation records supply identities only."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTS = ROOT / 'video_detection_agent/outputs'
TRAIN = OUTS / 'agreed7077_sft_iteration_20261017_v2/train_manifest.jsonl'
PREVIOUS = OUTS / 'grpo_candidate_screen_20261029_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def identities(source):
    keys = set()
    for key in ['merged_id', 'sha256', 'source_group']:
        if source.get(key):
            keys.add((key, source[key]))
    if source.get('video_path'):
        p = Path(source['video_path']).resolve()
        keys.update([('path', str(p)), ('filename', p.name)])
    # Original annotation paths identify shared source media even across aliases.
    original = source.get('original_labels', [])
    if isinstance(original, str):
        original = json.loads(original)
    for row in original:
        annotation = row.get('annotation')
        if annotation:
            path = str((ROOT / annotation).resolve())
            if Path(annotation).suffix.lower() in {'.tsv', '.csv', '.jsonl', '.xlsx'}:
                if row.get('locator'):
                    keys.add(('annotation_record', path+'#'+str(row['locator'])))
            else:
                keys.add(('annotation', path))
        label = row.get('label')
        if isinstance(label, dict):
            for key in ['Video_ID', 'video_id', 'videoId']:
                if label.get(key):
                    keys.add(('upstream_video', str(row.get('source'))+':'+str(label[key])))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    excluded, files, excluded_counts = set(), [], {}
    names = [
        'grpo_rapid128_live_20261027_v1/shard-01_stratified32/MANIFEST128.json',
        'grpo_rapid128_live_20261027_v1/shard-00/MANIFEST128.json',
        'grpo32_initial_screened_20261023_v1/MANIFEST128.json',
        'next_iteration_train_screen128_20261026_v1/MANIFEST32.json',
        'grpo_rapid128_live_20261027_v1/holdout120_v1/SAMPLES120.json',
    ]
    for name in names:
        p = OUTS / name
        rows = json.loads(p.read_text())
        excluded_counts[name] = len({r['source']['merged_id'] for r in rows})
        for row in rows:
            excluded.update(identities(row['source']))
        files.append({'path': str(p), 'sha256': digest(p), 'purpose': 'EXCLUSION_IDENTITIES_ONLY'})
    train = [json.loads(line) for line in TRAIN.read_text().splitlines() if line]
    eligible, quarantine = [], []
    for source in train:
        reasons = []
        if source.get('frozen_split') != 'train': reasons.append('NOT_TRAIN')
        if not all(source.get(k) for k in ['merged_id', 'sha256', 'source_group', 'video_path']):
            reasons.append('INCOMPLETE_IDENTITY')
        overlaps = identities(source) & excluded
        if overlaps: reasons.append('OLD_SAMPLING_OR_FIXED120_IDENTITY_OVERLAP')
        if reasons:
            quarantine.append({'source_id': source.get('merged_id'), 'reasons': reasons,
                               'identity_matches': sorted(overlaps)})
        else:
            eligible.append(source)
    # Quotas and ordering are based only on frozen Train metadata, before review.
    # Prior 48-source ordering supplies duration/route strata, not Student outcomes.
    preference = {json.loads(line)['source']['merged_id']: i for i, line in enumerate(
        (PREVIOUS / 'SOURCE_SELECTION48.jsonl').read_text().splitlines())}
    categories = ['Harmless', 'Sexual Content', 'Violence & Dangerous', 'Sensitive Figures & Ideologies']
    selected, used = [], set()
    for cat in categories:
        pool = [s for s in eligible if s['mapped_category'] == cat]
        # Explicitly seek both weak severity strata and multiple historic route tiers.
        cells = [('L0', 'DIRECT'), ('L0', 'ONE_TOOL'), ('L0', 'DIRECT'),
                 ('L0', 'ONE_TOOL'), ('L0', 'DIRECT')] if cat == 'Harmless' else [
                 ('L1', 'DIRECT'), ('L1', 'ONE_TOOL'), ('L2', 'DIRECT'),
                 ('L2', 'ONE_TOOL'), ('L2', 'TWO_PLUS_TOOLS')]
        for severity, route in cells:
            options = [s for s in pool if s['merged_id'] not in used and s.get('teacher_severity') == severity
                       and not any(identities(s) & identities(r['source']) for r in selected)]
            options.sort(key=lambda s: (s.get('route_tier') != route,
                preference.get(s['merged_id'], 100000), hashlib.sha256(s['merged_id'].encode()).hexdigest()))
            if not options:
                raise RuntimeError('Insufficient eligible Train coverage: '+cat+'/'+severity)
            source = options[0]
            same_group = identities(source)
            if any(same_group & identities(r['source']) for r in selected):
                raise RuntimeError('Pilot source alias collision; preserve and diagnose')
            used.add(source['merged_id'])
            gi = len(selected)
            original_index = preference.get(source['merged_id'])
            prior = PREVIOUS / 'student_rollouts' / f'group-{original_index:02d}' if original_index is not None else None
            complete = prior is not None and (prior / 'summary.json').exists() and not (prior / 'FAILURE.json').exists()
            manifest = json.loads((prior / 'MANIFEST.json').read_text()) if complete else [
                {'source': source, 'sample': sample, 'seed': 202610300000+gi*100+sample}
                for sample in range(4)]
            if any(m['source'] != source for m in manifest):
                raise ValueError('Prior source differs from current frozen Train row')
            actual_sha = digest(source['video_path'])
            if actual_sha != source['sha256']:
                raise ValueError('Media changed: '+source['video_path'])
            selection = {'group': gi, 'source': source, 'category_stratum': cat,
                'severity_stratum': source.get('teacher_severity'), 'historic_route_hint': source.get('route_tier'),
                'difficulty': '待验证',
                'difficulty_hypotheses': ['初始画面足够' if route == 'DIRECT' else '可能需要补看',
                    '边界案例待核实' if severity == 'L1' else '清晰度与遮挡待核实'],
                'actual_video_sha256': actual_sha, 'samples': manifest,
                'rollout_dir': str(prior if complete else args.out.resolve() / 'student_rollouts' / f'group-{gi:02d}'),
                'reuse_existing_complete_group': complete,
                'selection_did_not_read_student_answers': True}
            selected.append(selection)
            save(args.out / f'group-{gi:02d}.json', manifest)
    old = json.loads((PREVIOUS / 'PLAN.json').read_text())
    plan = {**old, 'cohort': '20-source Qwen-only GRPO group selection and validation',
        'videos': 20, 'rollouts': 80, 'samples_per_video': 4,
        'category_severity_quotas': dict(Counter(s['category_stratum']+'/'+s['severity_stratum'] for s in selected)),
        'pilot_then_expansion_gate': True, 'training_authorized': False,
        'source_manifest': str(TRAIN), 'train_manifest_sha256': digest(TRAIN),
        'exclusion_files': files, 'exclusion_source_counts': excluded_counts,
        'eligible_pool_size': len(eligible), 'quarantined_pool_size': len(quarantine),
        'optimizer_steps': 0, 'online_image_review': False, 'test_used': False,
        'prior_partial_group_policy': 'preserve; no silent overwrite or repaired trajectory',
        'expansion_quality_gate': {'pilot_groups': 20, 'minimum_validated_candidates': 5,
            'candidate_repeat_pair_order_agreement': 1.0,
            'maximum_technical_hold_fraction': 0.1,
            'maximum_semantic_inconsistency_fraction': 0.2},
        'reward_rule': '0..3 trace-grounded ordinal utility, no call-count/length/citation bonus; uncertainty differs from contradiction',
        'no_fixed120_content_used_to_select_or_tune': True}
    for key in ['selection_sha256', 'rollout_manifest_sha256', 'excluded_old_grpo_source_ids', 'excluded_fixed_holdout_ids']:
        plan.pop(key, None)
    save(args.out/'PLAN.json', plan)
    save(args.out/'SELECTION.json', selected)
    save(args.out/'QUARANTINE.json', quarantine)
    save(args.out/'SOURCE_AUDIT.json', {'training_source_count': len(train),
        'train_manifest': str(TRAIN), 'train_manifest_sha256': digest(TRAIN),
        'identity_keys': ['merged_id', 'sha256', 'realpath', 'filename', 'source_group', 'original_annotation_path'],
        'exclusions': files, 'counts': excluded_counts, 'selected': len(selected),
        'eligible': len(eligible), 'quarantine': len(quarantine),
        'selected_overlap_count': 0, 'fuzzy_perceptual_identity_not_claimed': True,
        'sft_exposure': 'not used in previous GRPO source lists; SFT exposure is metadata, not guaranteed unseen by SFT',
        'fixed120_records_used_only_for_exclusion': True})
    print(json.dumps({'out': str(args.out), 'selected': len(selected), 'eligible': len(eligible),
        'reuse': sum(s['reuse_existing_complete_group'] for s in selected),
        'coverage': plan['category_severity_quotas']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
