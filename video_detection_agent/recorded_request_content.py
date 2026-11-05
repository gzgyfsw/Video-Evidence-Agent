"""Render optional exact recorded user-content order for single-state SFT.

Legacy rows retain their original text-then-images layout. Recorded rows must
bind an immutable actual request; image labels are never guessed or reordered.
"""
import hashlib
import json
from pathlib import Path


def render_state_content(row,images):
    if row.get('contract_version') == 'unified-student-next-step-v1':
        from unified_student_contract import content, prompt, validate_output
        if row['prompt'] != prompt(row['context'], row['policy_text']):
            raise ValueError('Unified prompt differs from shared inference contract')
        validate_output(json.loads(row['target']), row['context'], row['audit_images'])
        return content(row['context'], row['policy_text'], row['audit_images'], images)
    if 'request_content_order' not in row:
        return [{'type':'text','text':row['prompt']}]+[{'type':'image','image':im} for im in images]
    path=Path(row['request_manifest_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['request_manifest_sha256']:
        raise ValueError('Actual recorded user request changed')
    source=json.loads(path.read_text());order=row['request_content_order']
    if source['content_order']!=order or source['prompt']!=row['prompt']:
        raise ValueError('SFT content differs from the actual recorded request')
    if len(images)!=len(row['audit_images']):raise ValueError('Actual audited image count changed')
    content=[];cursor=0
    for block in order:
        if block['type']=='text':content.append({'type':'text','text':block['text']})
        elif block['type']=='image_reference':
            if cursor>=len(images):raise ValueError('Too many actual request images')
            image=row['audit_images'][cursor]
            if (Path(block['path']).resolve()!=Path(image['path']).resolve()
                    or block['sha256']!=image['sha256'] or abs(block['timestamp']-image['timestamp'])>1e-6):
                raise ValueError('Actual request image order or identity changed')
            content.append({'type':'image','image':images[cursor]});cursor+=1
        else:raise ValueError('Unknown actual request content block')
    if cursor!=len(images):raise ValueError('Actual request omitted an audited image')
    if not row.get('request_prompt_token_ids'):raise ValueError('Actual native prompt-token binding required')
    return content
