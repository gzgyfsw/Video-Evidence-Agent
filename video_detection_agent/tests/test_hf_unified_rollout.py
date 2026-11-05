"""Real-tool binding and causal schema checks; no semantic quality claims."""
import copy
from types import SimpleNamespace
from pathlib import Path
import sys

import jsonschema
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_unified_rollout as runtime
from unified_student_contract import public_context, prompt, POLICY_SHA
from agent.qwen_backend import SAFETY_POLICY


def state():
    return {'duration_seconds': 30., 'budget': {'max_frames': 12, 'max_tool_calls': 3},
            'events': [{'type': 'observe', 'frame_ids': ['initial']}], 'visible_frame_ids': ['initial']}


def test_schema_preserves_all_three_model_choices_and_full_parameters():
    context = state();images = [{'timestamp': 1.}];schema = runtime.next_step_schema(context, images)
    for value in [
        {'action':'JOINT_SEMANTIC_SCAN','queries_by_risk':{'Violence & Dangerous':['visible knife']}},
        {'action':'TEMPORAL_EXPAND','timestamps':[1.],'window':2.,'num_frames':3},
        {'action':'ANSWER','category':None,'severity':None,'moderation_action':'REVIEW',
         'review_reason':'EVIDENCE_INSUFFICIENT','evidence':[{'timestamp':1.,'description':'Unclear visible action'}]}]:
        jsonschema.validate(value, schema)


def test_budget_removes_unexecutable_tools_without_forcing_block_or_allow():
    context = state();context['budget']['max_frames'] = 1
    schema = runtime.next_step_schema(context, [{'timestamp':1.}])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'action':'TEMPORAL_EXPAND','timestamps':[1.],'window':2.,'num_frames':1},schema)
    jsonschema.validate({'action':'ANSWER','category':None,'severity':None,'moderation_action':'REVIEW',
        'review_reason':'EVIDENCE_INSUFFICIENT','evidence':[{'timestamp':1.,'description':'Unclear'}]},schema)


def test_citation_excludes_candidate_hints_but_expansion_can_follow_them():
    context = state();context['events'].append({'type':'tool_result','result':{'candidates':[{'timestamp':12.}]}})
    schema = runtime.next_step_schema(context,[{'timestamp':1.}])
    jsonschema.validate({'action':'TEMPORAL_EXPAND','timestamps':[12.],'window':2.,'num_frames':1},schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'action':'ANSWER','category':'Harmless','severity':'L0','moderation_action':'ALLOW',
            'review_reason':None,'evidence':[{'timestamp':12.,'description':'Unseen candidate'}]},schema)


def test_schema_does_not_mutate_historical_context():
    context=state();before=copy.deepcopy(context)
    runtime.next_step_schema(context,[{'timestamp':1.}]);assert context==before


def test_shared_sft_inference_prompt_excludes_category_path_and_audit_provenance():
    context=state();context['events'].append({'type':'tool_result','result':{
        'frame_ids':['new-image'], 'candidates':[{'timestamp':8.,'score':.7,
            'matched_query':'visible fight','risk_type':'Violence & Dangerous','path':'SECRET_FILE'}],
        'original_tool_record':{'video_path':'data/merged_video_safety_v1/LEAKED_GOLD/video.mp4',
                               'video_sha256':'PRIVATE_HASH','elapsed_seconds':1.}}})
    before=copy.deepcopy(context);clean=public_context(context)
    text=prompt(context,SAFETY_POLICY)
    for forbidden in ['video_path','LEAKED_GOLD','SECRET_FILE','PRIVATE_HASH','original_tool_record']:
        assert forbidden not in text
    assert 'visible fight' in text and clean['events'][-1]['result']['frame_ids']==['new-image']
    assert context==before


def test_unified_loop_cannot_reintroduce_raw_tool_receipt_into_public_prompt():
    context=state();context['events'].append({'type':'tool_result','result':{
        'frame_ids':[], 'candidates':[], 'original_tool_record':{'dataset_label':'secret'}}})
    assert public_context(public_context(context))==public_context(context)
    assert 'dataset_label' not in prompt(context,SAFETY_POLICY)


def test_search_records_only_delivered_pixels_as_seen(tmp_path,monkeypatch):
    context=state();context['budget']['max_frames']=2
    images=[{'id':'initial','timestamp':1.}];calls=[]
    def locator(path, queries, top_k_per_risk):
        calls.append((path,queries,top_k_per_risk))
        return [SimpleNamespace(timestamp=t,score=.7,matched_query='knife',risk_type='Violence & Dangerous') for t in [4.,8.]]
    def decode(path,times):
        assert times==[4.]
        return [SimpleNamespace(timestamp=4.,image=Image.new('RGB',(8,8)))]
    monkeypatch.setattr(runtime,'extract_frames',decode)
    executor=runtime.RealVideoTools({'video_path':'original.mp4','sha256':'frozen-video'},locator,tmp_path)
    record,new=executor({'action':'JOINT_SEMANTIC_SCAN','queries_by_risk':{'Violence & Dangerous':['knife']}},context,images)
    assert calls==[('original.mp4',{'Violence & Dangerous':['knife']},2)]
    assert [im['timestamp'] for im in new]==[4.]
    assert [c['timestamp'] for c in record['candidates']]==[4.,8.]
    assert record['frame_ids']==[new[0]['id']]
    assert record['original_tool_record']['video_sha256']=='frozen-video'


def test_search_delivery_cap_keeps_candidates_and_temporal_headroom(tmp_path,monkeypatch):
    context=state();images=[{'id':str(i),'timestamp':float(i)} for i in range(4)]
    def locate(*args,**kw):
        return [SimpleNamespace(timestamp=float(t),score=.7,matched_query='event',risk_type='Violence & Dangerous') for t in range(5,11)]
    monkeypatch.setattr(runtime,'extract_frames',lambda path,times:[SimpleNamespace(timestamp=t,image=Image.new('RGB',(8,8))) for t in times])
    executor=runtime.RealVideoTools({'video_path':'source','sha256':'hash'},locate,tmp_path,max_search_new_frames=4)
    record,new=executor({'action':'JOINT_SEMANTIC_SCAN','queries_by_risk':{'Violence & Dangerous':['event']}},context,images)
    assert len(new)==4 and len(record['candidates'])==6
    assert context['budget']['max_frames']-len(images)-len(new)==4


def test_temporal_forwards_real_model_parameters_and_deduplicates(tmp_path,monkeypatch):
    got=[]
    class Expander:
        def __call__(self,path,timestamps,window,num_frames):
            got.append((path,timestamps,window,num_frames))
            return [SimpleNamespace(timestamp=t,image=Image.new('RGB',(8,8))) for t in [1.,2.,2.]]
    monkeypatch.setattr(runtime,'TemporalEvidenceExpander',Expander)
    executor=runtime.RealVideoTools({'video_path':'source.mp4','sha256':'hash'},None,tmp_path)
    record,new=executor({'action':'TEMPORAL_EXPAND','timestamps':[1.],'window':1.5,'num_frames':3},
        state(),[{'id':'initial','timestamp':1.}])
    assert got==[('source.mp4',[1.],1.5,3)]
    assert [im['timestamp'] for im in new]==[2.]
    assert record['original_tool_record']['actual_execution'] is True
