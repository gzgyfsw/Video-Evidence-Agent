"""Separate acquisition and citation of existing weakly supported source frames.

Exact source-time proxy only. It does not certify newly generated descriptions,
the equivalence of resized image renderings, or semantic risk discovery.
"""
import json
from pathlib import Path

from hf_audited_frame_reward import reference, score as citation_score
from request_policy_replay import sha

PROTOCOL = {'version':'audited-source-frame-acquisition-citation-proxy-v2',
    'frame_match_seconds':.000001, 'acquired_reference_frame_reward':1.,
    'cited_reference_frame_bonus':.25, 'tool_cost_only_after_acquisition':.03,
    'new_frame_cost_only_after_acquisition':.003, 'classification_reward':0.,
    'semantic_description_reward':0., 'independent_gold':False,
    'reward_missing_reference':None, 'missing_known_positive_hit_score':0.,
    'score_zero_is_not_a_negative_annotation':True, 'rendered_pixel_equivalence_assumed':False}


def score(result, ref):
    baseline=citation_score(result,ref)
    if not baseline['eligible']:return baseline
    acquired=[]
    for call in result['requests']:
        item=json.loads(Path(call['item_path']).read_text())
        if sha(item['request_path'])!=item['request_sha256']:raise ValueError('Actual request changed')
        request=json.loads(Path(item['request_path']).read_text())
        for image in request['images']:
            for anchor in ref['supported_source_frames']:
                if abs(image['timestamp']-anchor['source_frame_timestamp'])<=PROTOCOL['frame_match_seconds']:
                    if sha(image['path'])!=image['sha256']:raise ValueError('Acquired target pixels changed')
                    identity=(image['timestamp'],image['sha256'])
                    if any((m['timestamp'],m['actual_image_sha256'])==identity for m in acquired):continue
                    acquired.append({'timestamp':image['timestamp'],'actual_image_sha256':image['sha256'],
                        'audit_task_id':anchor['audit_task_id'],'first_visible_request':call['item_path'],
                        'same_source_time_only_not_description_approval':True})
    found=bool(acquired);cited=baseline['reference_hit']
    cost=(.03*len(result['actual_tools'])+.003*sum(len(t['returned_timestamps']) for t in result['actual_tools'])) if found else 0.
    return {'eligible':True,'total':(1.+(.25 if cited else 0.)-cost) if found else 0.,
        'reference_acquired':found,'reference_cited':cited,'acquisition_matches':acquired,
        'citation_matches':baseline['matches'],'cost':cost,'weak_proxy_only':True,
        'semantic_description_verified':False,'classification_reward':0.,
        'missing_hit_is_unknown_not_wrong':not found,
        'reason':'AUDITED_SOURCE_FRAME_ACQUISITION_AND_CITATION_PROXY'}
