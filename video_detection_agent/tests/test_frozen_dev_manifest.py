"""Cohort identity and generation parity; no inference or visual review."""
import ast
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frozen_dev_manifest import load_frozen_dev
from frozen_dev_manifest_metrics import compare
from request_policy_replay import sha


@pytest.fixture
def cohort(tmp_path):
    rows=[{'source':{'merged_id':str(i),'sha256':str(i),'frozen_split':'dev'},
           'seed':i,'sample':0} for i in range(3)]
    source=tmp_path/'dev.jsonl';source.write_text(''.join(json.dumps({'sha256':str(i),'split':'dev'})+'\n' for i in range(3)))
    manifest=tmp_path/'MANIFEST.json';manifest.write_text(json.dumps(rows))
    refs=tmp_path/'WEAK_REFERENCES.json';refs.write_text('[]')
    plan={'videos':3,'manifest_file':'MANIFEST.json','manifest_sha256':sha(manifest),
          'source_full_dev_path':str(source),'source_full_dev_sha256':sha(source),
          'weak_reference_sha256':sha(refs)}
    (tmp_path/'PLAN.json').write_text(json.dumps(plan))
    return tmp_path,rows,plan


def write_revision(root,rows,plan):
    (root/'MANIFEST.json').write_text(json.dumps(rows));plan['manifest_sha256']=sha(root/'MANIFEST.json')
    (root/'PLAN.json').write_text(json.dumps(plan))


def test_size_comes_from_complete_frozen_source_not_1679(cohort):
    root,_,_=cohort
    assert load_frozen_dev(root)[2]==3


def test_cannot_present_a_consistently_rehashed_subset_as_full_source(cohort):
    root,rows,plan=cohort;plan['videos']=2;write_revision(root,rows[:2],plan)
    with pytest.raises(ValueError,match='complete source Dev'):load_frozen_dev(root)


def test_missing_or_non_dev_identity_is_rejected(cohort):
    root,rows,plan=cohort;rows[0]['source']['frozen_split']='train';write_revision(root,rows,plan)
    with pytest.raises(ValueError,match='identity/partition'):load_frozen_dev(root)


def test_changed_authoritative_source_is_rejected(cohort):
    root,_,_=cohort;(root/'dev.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='source Dev changed'):load_frozen_dev(root)


def test_actual_student_tool_loop_is_unchanged():
    base=Path(__file__).resolve().parents[1]
    def loop(path):
        tree=ast.parse(path.read_text())
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For)
                    and isinstance(n.iter,ast.Subscript) and isinstance(n.iter.value,ast.Name)
                    and n.iter.value.id=='pending')
    assert ast.dump(loop(base/'run_frozen_dev_manifest.py'))==ast.dump(loop(base/'run_full_dev_accounted.py'))


def test_old_role_contract_cannot_silently_count_toward_new_dev(cohort):
    root,_,_=cohort;role=root/'old_role';role.mkdir()
    (role/'PLAN.json').write_text('{"manifest_sha256":"old"}')
    with pytest.raises(ValueError,match='different frozen cohort'):
        compare(root,role,root/'absent_candidate')
