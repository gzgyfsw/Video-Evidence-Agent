"""Exact context/token contract tests; these do not verify model quality."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from request_policy_replay import pack,replay_batch,validate_sampling,score_request
from train_grouped_video_grpo import make_trajectory


def fixture(tmp_path):
    settings={'version':1,'generation_config':'vllm','logprobs_mode':'processed_logprobs',
        'top_p':1.,'top_k':-1,'min_p':0.,'repetition_penalty':1.,'presence_penalty':0.,
        'frequency_penalty':0.,'min_tokens':0,'ignore_eos':False,'logit_bias':{},'stop':[],'bad_words':[]}
    request={'model':'student','stage':'tool_payload','temperature':.7,'effective_sampling_contract':settings,
        'chat_template_kwargs':{'enable_thinking':False},'content_order':[{'type':'text','text':'ONLY_THIS_REQUEST'}], 'images':[]}
    file=tmp_path/'request.json';file.write_text(json.dumps(request));digest=hashlib.sha256(file.read_bytes()).hexdigest()
    call={'model':'student','stage':'tool_payload','request_input_manifest':str(file),'request_input_sha256':digest,
        'raw':{'model':'student','prompt_token_ids':[101,102],'choices':[{'token_ids':[11,12,13],
        'message':{'content':'never reencode this text'},'finish_reason':'stop',
        'logprobs':{'content':[{'logprob':-.1},{'logprob':-.2},{'logprob':-.3}]}}]}}
    response=tmp_path/'response.json';response.write_text(json.dumps(call['raw']));call['response_path']=str(response)
    return call,{'source_partition':'train','source_trajectory_id':'train-video'}, {'model':'student','weight_sha256':'a'*64}


class Processor:
    def apply_chat_template(self,messages,**kwargs):
        assert messages==[{'role':'user','content':[{'type':'text','text':'ONLY_THIS_REQUEST'}]}]
        assert kwargs['enable_thinking'] is False
        return 'literal input'
    def __call__(self,**kwargs):return {'input_ids':torch.tensor([[101,102]]),'attention_mask':torch.ones((1,2),dtype=torch.long)}


def test_independent_request_prefix_and_causal_positions(tmp_path):
    item=pack(*fixture(tmp_path));batch,pos,targets=replay_batch(item,Processor())
    assert batch['input_ids'].tolist()==[[101,102,11,12]]
    assert pos.tolist()==[1,2,3] and targets.tolist()==[11,12,13]
    assert len(batch['attention_mask'][0])==4


def test_actual_completion_ids_are_used_without_reencoding(tmp_path):
    item=pack(*fixture(tmp_path))
    batch,_,targets=replay_batch(item,Processor());assert targets.tolist()==[11,12,13]
    item['completion_token_ids']=[99,98,97]
    with pytest.raises(ValueError,match='actual native response'):replay_batch(item,Processor())


def test_packed_temperature_cannot_override_actual_request(tmp_path):
    item=pack(*fixture(tmp_path));item['temperature']=1.
    with pytest.raises(ValueError,match='actual request'):replay_batch(item,Processor())


def test_extra_history_is_detected_by_native_prefix_check(tmp_path):
    item=pack(*fixture(tmp_path))
    class Wrong(Processor):
        def __call__(self,**kwargs):return {'input_ids':torch.tensor([[999,101,102]])}
    with pytest.raises(ValueError,match='prefix'):replay_batch(item,Wrong())


def test_dev_test_cannot_be_packed(tmp_path):
    call,source,binding=fixture(tmp_path)
    for split in ['dev','test']:
        with pytest.raises(ValueError,match='Train'):pack(call,{**source,'source_partition':split},binding)


def test_unknown_sampling_defaults_fail_closed(tmp_path):
    call,_,_=fixture(tmp_path);r=json.loads(Path(call['request_input_manifest']).read_text())
    r['effective_sampling_contract'].pop('top_k')
    with pytest.raises(ValueError,match='missing or unsupported'):validate_sampling(r)
    r['effective_sampling_contract']['top_k']=20
    with pytest.raises(ValueError,match='missing or unsupported'):validate_sampling(r)


def test_greedy_is_not_treated_as_stochastic_behavior(tmp_path):
    call,_,_=fixture(tmp_path);r=json.loads(Path(call['request_input_manifest']).read_text());r['temperature']=0
    with pytest.raises(ValueError,match='Positive'):validate_sampling(r)


def test_changed_request_or_missing_native_logprobs_fail(tmp_path):
    call,source,binding=fixture(tmp_path);call['raw']['choices'][0]['logprobs']=None
    Path(call['response_path']).write_text(json.dumps(call['raw']))
    with pytest.raises(ValueError,match='logprobs missing'):pack(call,source,binding)
    Path(call['request_input_manifest']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):pack(call,source,binding)


def test_frozen_other_model_cannot_receive_this_policy_credit(tmp_path):
    call,source,binding=fixture(tmp_path)
    with pytest.raises(ValueError,match='model/checkpoint'):pack(call,source,{**binding,'model':'other'})


def test_legacy_builder_refuses_independent_request_linearization():
    record={'id':'x','context_layout':'independent_requests','model_turns':[{'user_prompt':'x','frame_timestamps':[], 'assistant_raw':'{}'}]}
    with pytest.raises(ValueError,match='Independent requests'):make_trajectory({},record,0.)


def test_full_and_cached_modes_use_same_known_causal_prefix_and_gradient(tmp_path):
    import numpy as np
    item=pack(*fixture(tmp_path));mask=tmp_path/'mask.npz'
    np.savez(mask,bitmask=np.full((3,4),-1,dtype=np.int32),indices=np.arange(3),token_ids=np.array([11,12,13]))
    grammar={'request_sha256':item['request_sha256'],'masks_path':str(mask),
        'masks_sha256':hashlib.sha256(mask.read_bytes()).hexdigest(),'vocab_size':128}
    class Causal(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.arange(128,dtype=torch.float32)/128)
        def forward(self,input_ids,logits_to_keep,attention_mask=None,mm_token_type_ids=None,past_key_values=None,use_cache=False):
            cumulative=input_ids.cumsum(-1)+(0 if past_key_values is None else past_key_values.total)
            logits=(cumulative.float()/100)[...,None]*self.w
            logits=logits[:,-logits_to_keep:] if isinstance(logits_to_keep,int) else logits[:,logits_to_keep]
            return SimpleNamespace(logits=logits,past_key_values=SimpleNamespace(total=cumulative[:,-1:]))
    model=Causal();first,_=score_request(model,Processor(),item,grammar,'cpu','full_causal')
    first.sum().backward();grad=model.w.grad.clone();model.zero_grad()
    second,_=score_request(model,Processor(),item,grammar,'cpu','cached_teacher_forcing')
    second.sum().backward();torch.testing.assert_close(first,second);torch.testing.assert_close(grad,model.w.grad)


def test_legacy_optimizer_refuses_implicit_conversation_before_loading_weights():
    script=Path(__file__).resolve().parents[1]/'train_grouped_video_grpo.py'
    command=[sys.executable,str(script)]
    for key in ['--model','--policy-adapter','--reference-adapter','--manifest','--rollouts','--output']:
        command += [key,'unused']
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and 'disabled by default' in result.stderr
