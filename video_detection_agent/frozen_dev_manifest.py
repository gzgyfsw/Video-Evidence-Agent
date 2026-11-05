"""Read the entire hash-bound Dev cohort; never infer its size from a subset."""
import json
from pathlib import Path
from request_policy_replay import sha


def load_frozen_dev(root):
    root=Path(root);plan=json.loads((root/'PLAN.json').read_text())
    name=plan.get('manifest_file','MANIFEST1679.json')
    path=root/name
    if path.resolve().parent!=root.resolve():raise ValueError('Manifest must be inside frozen cohort')
    expected=plan['videos']
    if type(expected) is not int or expected<1:raise ValueError('Invalid authoritative Dev count')
    if sha(path)!=plan['manifest_sha256']:raise ValueError('Frozen manifest changed')
    rows=json.loads(path.read_text())
    if (len(rows)!=expected or len({r['source']['merged_id'] for r in rows})!=expected
            or len({r['source']['sha256'] for r in rows})!=expected
            or any(r['source']['frozen_split']!='dev' for r in rows)):
        raise ValueError('Full Dev identity/partition coverage differs from frozen count')
    if 'source_full_dev_path' in plan:
        source=Path(plan['source_full_dev_path'])
        if sha(source)!=plan['source_full_dev_sha256']:raise ValueError('Full source Dev changed')
        original=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        if (len(original)!=expected or any(r['split']!='dev' for r in original)
                or {r['sha256'] for r in original}!={r['source']['sha256'] for r in rows}):
            raise ValueError('Frozen cohort is not the complete source Dev')
    return plan,rows,expected


def runtime_path(root):
    root=Path(root);plan=json.loads((root/'PLAN.json').read_text())
    return root/('RUNTIME_BINDINGS.json' if 'manifest_file' in plan else 'RUNTIME_BINDINGS_v3.json')
