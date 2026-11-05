"""Score Actor tokens under the same explicit grammar/temperature convention.

Does not equate an arbitrary rescoring distribution with a recorded behavior
policy. Collection must separately bind its effective sampling configuration.
"""
import torch


def allowed_from_bitmask(bitmask, vocab_size, device):
    words=torch.as_tensor(bitmask.astype('int64'),device=device)
    indices=torch.arange(vocab_size,device=device)
    return ((words[:,indices//32] >> (indices%32)) & 1).bool()


def constrained_token_logprobs(logits, token_ids, allowed, temperature):
    if not isinstance(temperature,(int,float)) or not 0<temperature<=2:
        raise ValueError('A stochastic policy needs an explicit positive temperature')
    if logits.ndim!=2 or logits.shape!=allowed.shape or len(token_ids)!=len(logits):
        raise ValueError('Actor score/mask/token dimensions disagree')
    if not bool(allowed.any(dim=-1).all()):raise ValueError('Grammar permits no next token')
    if not bool(allowed.gather(-1,token_ids[:,None]).all()):
        raise ValueError('Actual Actor token is not allowed by its recorded grammar prefix')
    scores=logits.float().masked_fill(~allowed,float('-inf'))/temperature
    result=scores.log_softmax(-1).gather(-1,token_ids[:,None]).squeeze(-1)
    if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite Actor policy score')
    return result
