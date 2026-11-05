"""Credit actual native tokens in fresh requests, including noncanonical BPE splits."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from actor_request_credit import build_actor_credit,bytes_to_unicode


class FakeTokenizer:
    all_special_ids=[0]
    backend_tokenizer=SimpleNamespace(decoder=SimpleNamespace(__getstate__=lambda:b'{"type":"ByteLevel"}'))
    def __init__(self,text):
        mapping=bytes_to_unicode();self.pieces=[mapping[b] for b in text.encode('utf-8')]
    def convert_ids_to_tokens(self,index):return self.pieces[index-1]
    def encode(self,*args,**kwargs):raise AssertionError('Actual generated IDs must never be re-encoded')


class Processor:
    def __init__(self,text,prefix=(101,102)):self.tokenizer=FakeTokenizer(text);self.prefix=prefix
    def apply_chat_template(self,messages,**kwargs):
        assert len(messages)==1 and messages[0]['role']=='user'
        assert kwargs['add_generation_prompt'] and not kwargs['enable_thinking']
        return 'single actual user request'
    def __call__(self,**kwargs):return {'input_ids':torch.tensor([self.prefix])}


def fixture(tmp_path,text,stage='route'):
    request={'model':'initial','stage':stage,'content_order':[{'type':'text','text':'actual prefix'}]}
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    processor=Processor(text);completion=list(range(1,len(processor.tokenizer.pieces)+1))+[0]
    call={'model':'initial','stage':stage,'request_input_manifest':str(path),
        'request_input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw':{'model':'initial','prompt_token_ids':[101,102],
            'choices':[{'token_ids':completion,'message':{'content':text}}]}}
    return call,processor


def credit(call,processor,partition='train'):
    return build_actor_credit(call,partition,'initial',processor,{'model':'initial','weight_sha256':'a'*64})


def test_route_credits_actual_action_not_unused_later_output(tmp_path):
    text='{"action":"ANSWER","unused":"not trained"}'
    call,processor=fixture(tmp_path,text);result=credit(call,processor)
    selected=bytes(text.encode()[i] for i in result['credited_completion_indices']).decode()
    assert selected=='"action":"ANSWER"'
    assert result['completion_token_ids']==call['raw']['choices'][0]['token_ids']
    assert result['fresh_single_user_context'] and result['frozen_helper_tokens_credited']==0


def test_tool_container_and_split_utf8_tokens_are_credited_without_action_constant(tmp_path):
    text='{"action":"JOINT_SEMANTIC_SCAN","queries_by_risk":{"Sexual Content":["café scene"]}}'
    call,processor=fixture(tmp_path,text,'tool_payload');result=credit(call,processor)
    selected=bytes(text.encode('utf-8')[i] for i in result['credited_completion_indices']).decode('utf-8')
    assert selected.startswith('"queries_by_risk":') and 'café scene' in selected
    assert 'JOINT_SEMANTIC_SCAN' not in selected


def test_frozen_stages_and_other_actor_role_receive_no_credit(tmp_path):
    call,processor=fixture(tmp_path,'{"action":"ANSWER"}')
    call['stage']='policy_payload';assert credit(call,processor) is None
    call['stage']='route';call['model']='post_candidate';assert credit(call,processor) is None
    with pytest.raises(ValueError,match='Dev/Test'):credit(call,processor,'dev')


def test_changed_request_or_wrong_conditioning_fails_closed(tmp_path):
    call,processor=fixture(tmp_path,'{"action":"ANSWER"}')
    processor.prefix=(999,101,102)
    with pytest.raises(ValueError,match='differs'):credit(call,processor)
    Path(call['request_input_manifest']).write_text('{}')
    with pytest.raises(ValueError,match='request changed'):credit(call,processor)


def test_missing_actual_ids_is_not_reconstructed_by_guessing(tmp_path):
    call,processor=fixture(tmp_path,'{"action":"ANSWER"}')
    call['raw']['prompt_token_ids']=None
    with pytest.raises(ValueError,match='Actual native'):credit(call,processor)


def test_bytelevel_alphabet_matches_known_whitespace_and_utf8_byte_pieces():
    mapping=bytes_to_unicode()
    assert len(mapping)==256 and len(set(mapping.values()))==256
    assert mapping[32]=='Ġ' and mapping[10]=='Ċ' and mapping[0]=='Ā'
    assert ''.join(mapping[b] for b in 'é'.encode('utf-8'))=='Ã©'
