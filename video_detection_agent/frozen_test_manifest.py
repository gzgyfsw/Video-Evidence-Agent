"""Read the complete hash-bound Test cohort without exposing weak labels."""
import json
from pathlib import Path

from request_policy_replay import sha


def load_frozen_test(root):
    root = Path(root)
    plan = json.loads((root / "PLAN.json").read_text())
    path = root / plan["manifest_file"]
    if path.resolve().parent != root.resolve():
        raise ValueError("Test manifest must be inside the frozen cohort")
    expected = plan["videos"]
    if type(expected) is not int or expected < 1 or sha(path) != plan["manifest_sha256"]:
        raise ValueError("Invalid frozen Test count or manifest hash")
    rows = json.loads(path.read_text())
    if (len(rows) != expected
            or len({r["source"]["merged_id"] for r in rows}) != expected
            or len({r["source"]["sha256"] for r in rows}) != expected
            or any(r["source"]["frozen_split"] != "test" for r in rows)):
        raise ValueError("Full Test identity or partition coverage differs")
    source = Path(plan["source_full_test_path"])
    if sha(source) != plan["source_full_test_sha256"]:
        raise ValueError("Source Test split changed")
    originals = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if (len(originals) != expected or any(r["split"] != "test" for r in originals)
            or {r["sha256"] for r in originals} != {r["source"]["sha256"] for r in rows}):
        raise ValueError("Frozen cohort is not the complete source Test split")
    return plan, rows, expected


def runtime_path(root):
    return Path(root) / "RUNTIME_BINDINGS.json"
