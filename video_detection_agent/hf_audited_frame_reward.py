"""Narrow weak proxy: cite an existing supported source frame, not a gold fact.

No semantic comparison or reward for classification, identities, wording, or
Teacher route imitation. Missing references mean ineligible, not harmless.
"""
import hashlib
import json
from pathlib import Path

from request_policy_replay import sha
from unified_student_contract import POLICY_SHA

PROTOCOL = {'version': 'audited-source-frame-citation-proxy-v1',
    'frame_match_seconds': .000001, 'citation_reward': 1.,
    'tool_cost_only_after_hit': .03, 'new_frame_cost_only_after_hit': .003,
    'classification_reward': 0., 'semantic_description_reward': 0.,
    'independent_gold': False, 'reward_missing_reference': None,
    'missing_known_positive_hit_score': 0., 'score_zero_is_not_a_negative_annotation': True}


def reference(meta):
    path = Path(meta.get('candidate_file', ''))
    if not path.is_file():return None
    candidate = json.loads(path.read_text());original_path = path.parent/'original.json'
    if not original_path.is_file():return None
    original = json.loads(original_path.read_text())
    digest = hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':')).encode()).hexdigest()
    if (meta['frozen_split'] != 'train' or original['split'] != 'train'
            or original['id'] != meta['merged_id'] or original['video']['sha256'] != meta['sha256']
            or candidate['original_sha256'] != digest):
        raise ValueError('Weak reference identity/split/canonical-source binding failed')
    if (meta['mapped_category'] == 'Harmless' or candidate.get('technical_failures')
            or any(i.get('severity') != 'WARNING' for i in candidate.get('mechanical_issues', []))):
        return None
    frames = {f['id']: f for f in original['frames']};supported = []
    for check in candidate['checks']:
        subject = check['subject'];time = subject.get('timestamp')
        if (check['kind'] != 'claim' or check['judgment'] != 'SUPPORTED'
                or check.get('status') != 'COMPLETED' or type(time) not in {int, float}
                or check['provenance'].get('policy_sha256') != POLICY_SHA
                or '/evidence/' not in subject.get('target', {}).get('path', '')):
            continue
        for frame_id in subject.get('frame_ids', []):
            frame = frames.get(frame_id)
            if (frame is None or frame.get('video_sha256') != meta['sha256']
                    or abs(frame['timestamp']-time) > .001):continue
            manifest = next((im for im in check.get('image_manifest', []) if im['id']==frame_id), None)
            if not manifest or sha(manifest['path']) != manifest['sha256']:
                raise ValueError('Existing audit image unavailable or changed')
            supported.append({'source_frame_timestamp': frame['timestamp'], 'frame_id': frame_id,
                'audit_task_id': check['task_id'], 'claim_text': subject['text'],
                'reviewed_image_path': manifest['path'], 'reviewed_image_sha256': manifest['sha256'],
                'audit_model': check['provenance']['model']})
    if not supported:return None
    return {'source_id': meta['merged_id'], 'video_sha256': meta['sha256'],
        'candidate_path': str(path), 'candidate_sha256': sha(path),
        'original_path': str(original_path), 'original_sha256': sha(original_path),
        'supported_source_frames': supported, 'weak_reference_only': True,
        'new_rendering_pixel_identity_not_assumed': True}


def score(result, ref):
    if ref is None:return {'eligible': False, 'total': None, 'reason': 'NO_RELIABLE_FRAME_REFERENCE'}
    if result.get('source_id') != ref['source_id']:raise ValueError('Reward source identity mismatch')
    if result.get('status') != 'MODEL_ANSWER_UNREVIEWED':
        return {'eligible': False, 'total': None, 'reason': 'INCOMPLETE_OR_TECHNICAL_FAILURE'}
    last = Path(result['requests'][-1]['item_path'])
    item = json.loads(last.read_text());request = json.loads(Path(item['request_path']).read_text())
    if sha(item['request_path']) != item['request_sha256']:raise ValueError('Actual final input changed')
    visible = request['images'];matches = []
    for evidence in result['answer']['evidence']:
        shown = next((im for im in visible if abs(im['timestamp']-evidence['timestamp']) <= .000001), None)
        if shown is None:raise ValueError('Unseen citation cannot receive a weak frame reward')
        if sha(shown['path']) != shown['sha256']:raise ValueError('Rewarded actual pixels changed')
        for anchor in ref['supported_source_frames']:
            if abs(evidence['timestamp']-anchor['source_frame_timestamp']) <= PROTOCOL['frame_match_seconds']:
                matches.append({'actual_image_sha256': shown['sha256'], 'timestamp': evidence['timestamp'],
                    'audit_task_id': anchor['audit_task_id'], 'same_source_time_only_not_description_approval': True})
    for tool in result['actual_tools']:
        if not tool['actual_execution'] or tool['video_sha256'] != ref['video_sha256']:
            raise ValueError('Non-actual or wrong-video tool cannot be rewarded')
    hit = bool(matches)
    cost = (PROTOCOL['tool_cost_only_after_hit']*len(result['actual_tools'])
            + PROTOCOL['new_frame_cost_only_after_hit']*sum(len(t['returned_timestamps']) for t in result['actual_tools'])) if hit else 0.
    return {'eligible': True, 'total': 1.-cost if hit else 0., 'reference_hit': hit,
        'cost': cost, 'matches': matches, 'weak_proxy_only': True,
        'semantic_description_verified': False, 'missing_hit_is_unknown_not_wrong': not hit,
        'classification_reward': 0., 'reason': 'AUDITED_SOURCE_FRAME_CITATION_PROXY'}
