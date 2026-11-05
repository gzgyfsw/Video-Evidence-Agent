import copy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import student_action_outcomes as outcomes


@pytest.fixture
def duplicate(tmp_path):
    images=[]
    for i in range(2):
        path=tmp_path/f'image{i}.bin';path.write_bytes(bytes([i]))
        images.append({'id':f'frame-{i}','timestamp':float(i),'path':str(path),'sha256':outcomes.sha(path)})
    action={'action':'JOINT_SEMANTIC_SCAN','queries_by_risk':{'Violence & Dangerous':['person action']}}
    initial={'step_id':'initial','type':'observe','frame_ids':['frame-0']}
    events=[initial,{'step_id':'turn-1','type':'decision','action':action['action'],'parameters':{'queries_by_risk':action['queries_by_risk']}},
            {'step_id':'turn-1-result','type':'tool_result','call_step_id':'turn-1','result':{'frame_ids':['frame-1']}}]
    records={};calls=[]
    for i,history in enumerate([[initial],events],1):
        folder=tmp_path/f'turn-{i}';folder.mkdir();item=folder/'item.json';item.write_text('{}')
        context={'duration_seconds':2.,'budget':{'max_frames':12,'max_tool_calls':3},
                 'events':history,'visible_frame_ids':[im['id'] for im in images[:i]]}
        request={'model':'fixed-student','context':context,'images':images[:i]}
        records[str(item)]=({'model_binding':{},'source_trajectory_id':'source'},request,
            {'decoded_text_for_inspection_only':json.dumps(action),'finish_reason':'stop'})
        calls.append({'item_path':str(item)})
        (tmp_path/f'turn-{i}-context.json').write_text(json.dumps({'context':context,'images':images[:i]}))
        (tmp_path/f'turn-{i}-output.json').write_text(json.dumps(action))
    (tmp_path/'turn-1-tool-return.json').write_text('{}')
    failure={'error':repr(ValueError('Duplicate tool call; no silent replacement')),'events':events}
    (tmp_path/'failure.json').write_text(json.dumps(failure))
    return tmp_path,calls,lambda p:records[str(p)],failure


def test_recorded_duplicate_is_unresolved_and_not_resampled(duplicate):
    loop,calls,reader,_=duplicate
    result=outcomes.classify_recorded_failure(loop,calls,reader)
    assert result['status']=='INVALID_STUDENT_ACTION' and result['failure_reason']=='DUPLICATE_TOOL_CALL'
    assert result['answer'] is None and result['tool_calls']==1 and result['decisions']==2
    assert not result['resampled_or_repaired'] and not result['business_disposition_generated']


def test_tool_error_is_not_relabelled_student_failure(duplicate):
    loop,calls,reader,failure=duplicate
    failure['error']=repr(ValueError('Tool technical failure: timeout'))
    (loop/'failure.json').write_text(json.dumps(failure))
    assert outcomes.classify_recorded_failure(loop,calls,reader) is None


def test_same_error_text_after_tool_attempt_still_is_not_student_duplicate(duplicate):
    loop,calls,reader,failure=duplicate
    altered=copy.deepcopy(failure);altered['events'].append({'step_id':'turn-2','type':'decision','action':'JOINT_SEMANTIC_SCAN','parameters':{}})
    (loop/'failure.json').write_text(json.dumps(altered))
    assert outcomes.classify_recorded_failure(loop,calls,reader) is None


def test_changed_pixels_remain_integrity_error(duplicate):
    loop,calls,reader,_=duplicate
    (loop/'image0.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='pixels changed'):
        outcomes.classify_recorded_failure(loop,calls,reader)


def test_valid_run_is_passed_through_once_without_changes(monkeypatch):
    sentinel={'status':'MODEL_ANSWER_UNREVIEWED'};seen=[]
    monkeypatch.setattr(outcomes,'run',lambda *a,**k:seen.append(1) or sentinel)
    assert outcomes.run_accounted(None,2,{},[],None,'unused',reader=None) is sentinel
    assert seen==[1]


def test_invalid_parameters_are_confirmed_from_actual_output_not_error_text(duplicate):
    loop,calls,reader,failure=duplicate
    response=reader(Path(calls[-1]['item_path']))[2]
    response['decoded_text_for_inspection_only']=json.dumps({'action':'JOINT_SEMANTIC_SCAN','queries_by_risk':{}})
    (loop/'turn-2-output.json').unlink()
    failure['error']=repr(ValueError('Invalid risk map'))
    (loop/'failure.json').write_text(json.dumps(failure))
    assert outcomes.classify_recorded_failure(loop,calls,reader)['failure_reason']=='INVALID_ACTION_PARAMETERS'
    failure['error']=repr(ValueError('Invalid window'))
    (loop/'failure.json').write_text(json.dumps(failure))
    assert outcomes.classify_recorded_failure(loop,calls,reader) is None


def test_truncated_output_is_not_repaired_or_called_a_final_answer(duplicate):
    loop,calls,reader,failure=duplicate
    response=reader(Path(calls[-1]['item_path']))[2]
    response.update(decoded_text_for_inspection_only='{"action":',finish_reason='length')
    (loop/'turn-2-output.json').unlink()
    failure['error']=repr(ValueError('Incomplete model JSON; saved output is not repaired'))
    (loop/'failure.json').write_text(json.dumps(failure))
    result=outcomes.classify_recorded_failure(loop,calls,reader)
    assert result['failure_reason']=='INCOMPLETE_MODEL_OUTPUT'
    assert result['failed_output']=='{"action":' and result['answer'] is None
