"""Locate top-level JSON fields without supervising unverified prose."""
import json


def validate_classification_context(payload, names):
    """Do not let an unsupervised generated answer reveal the label first.

    For category-only single-state tasks, action=ANSWER is a known task constant.
    Disposition, grade, prose or other generated fields are not given inputs.
    """
    obj=json.loads(payload)
    if not isinstance(obj,dict) or obj.get('action')!='ANSWER':
        raise ValueError('classification view requires the known ANSWER task')
    keys=list(obj)
    if not set(names)<=set(keys):raise ValueError('classification supervision fields missing')
    last=max(keys.index(name) for name in names)
    unknown=set(keys[:last+1])-set(names)-{'action'}
    if unknown:
        raise ValueError('Unsupervised generated fields precede classification labels: '+','.join(sorted(unknown)))


def field_ranges(payload, names, allow_integer_null=False):
    if not isinstance(json.loads(payload), dict):
        raise ValueError('field supervision requires a JSON object')
    decoder = json.JSONDecoder()
    def space(i):
        while i < len(payload) and payload[i].isspace():
            i += 1
        return i
    pos = space(0)
    if payload[pos:pos + 1] != '{':
        raise ValueError('field supervision requires a JSON object')
    pos += 1
    found, seen = {}, set()
    while True:
        pos = space(pos)
        if payload[pos:pos + 1] == '}':
            pos = space(pos + 1)
            break
        start = pos
        key, pos = decoder.raw_decode(payload, pos)
        if not isinstance(key, str) or key in seen:
            raise ValueError('JSON field keys must be unique strings')
        seen.add(key)
        pos = space(pos)
        if payload[pos:pos + 1] != ':':
            raise ValueError('missing field separator')
        pos = space(pos + 1)
        value, pos = decoder.raw_decode(payload, pos)
        if key in names:
            if not isinstance(value, str) and not (allow_integer_null and (value is None or type(value) is int)):
                raise ValueError('selected field must be a string (or an explicitly enabled integer/null)')
            found[key] = (start, pos)
        pos = space(pos)
        if payload[pos:pos + 1] == ',':
            pos += 1
        elif payload[pos:pos + 1] != '}':
            raise ValueError('invalid object separator')
    if pos != len(payload) or set(found) != set(names):
        raise ValueError('missing selected field or trailing content')
    return list(found.values())


def token_positions(tokenizer, payload, names, allow_integer_null=False):
    spans = field_ranges(payload, names, allow_integer_null)
    encoded = tokenizer(payload, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (start, end) in enumerate(encoded['offset_mapping'])
                 if end > start and any(a <= start < end <= b for a, b in spans)]
    if not positions:
        raise ValueError('selected JSON fields contain no complete tokens')
    return encoded['input_ids'], positions
