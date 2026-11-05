from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from actor_policy_probabilities import allowed_from_bitmask,constrained_token_logprobs
from export_actor_grammar_masks import token_allowed


def test_masked_tokens_cannot_affect_probability_or_receive_gradients():
    logits=torch.tensor([[1.,2.,100.]],requires_grad=True)
    allowed=torch.tensor([[True,True,False]])
    score=constrained_token_logprobs(logits,torch.tensor([1]),allowed,.7)
    assert torch.allclose(score,torch.log_softmax(torch.tensor([[1.,2.]])/.7,-1)[:,1])
    (-score.sum()).backward()
    assert logits.grad[0,2]==0
    fixed=constrained_token_logprobs(logits,torch.tensor([0]),torch.tensor([[True,False,False]]),1.)
    assert fixed.item()==0


def test_actual_disallowed_token_or_missing_stochastic_contract_fails_closed():
    logits=torch.zeros((1,3));token=torch.tensor([2])
    with pytest.raises(ValueError,match='allowed'):constrained_token_logprobs(logits,token,torch.tensor([[True,False,False]]),1.)
    with pytest.raises(ValueError,match='positive'):constrained_token_logprobs(logits,token,torch.ones_like(logits,dtype=torch.bool),0)
    with pytest.raises(ValueError,match='no next'):constrained_token_logprobs(logits,token,torch.zeros_like(logits,dtype=torch.bool),1.)


def test_signed_packed_masks_preserve_token_indices():
    masks=np.array([[1,-2147483648],[-1,1]],dtype=np.int32)
    unpacked=allowed_from_bitmask(masks,64,'cpu')
    for row in range(2):
        for token in range(64):assert bool(unpacked[row,token])==token_allowed(masks[row],token)
    assert unpacked[0,0] and unpacked[0,63] and not unpacked[0,32]
