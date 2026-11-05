"""Explicit disjoint per-request RNG streams; no model or dataset access."""
STRIDE=1_000_000_000_000
SCHEDULE='disjoint-request-v2'
LEGACY='legacy-plus-call'


def request_seed(base_seed,call_index,schedule=SCHEDULE):
    if type(base_seed) is not int or type(call_index) is not int or base_seed<0 or call_index<0:
        raise ValueError('Nonnegative integer base seed/call index required')
    if schedule==LEGACY:return base_seed+call_index
    if schedule!=SCHEDULE or base_seed>=STRIDE:raise ValueError('Unsupported seed schedule/base range')
    value=base_seed+STRIDE*call_index
    if value>=2**63:raise ValueError('Request seed outside signed64 range')
    return value


def validate_seeds(results,manifest,reader,schedule=SCHEDULE):
    expected={(m['source']['merged_id'],m['sample']):m['seed'] for m in manifest};used={}
    for result in results:
        ident=(result['source_id'],result['sample']);base=expected[ident]
        for index,call in enumerate(result['requests']):
            _,request,_=reader(call['item_path']);actual=request['seed']
            if actual!=request_seed(base,index,schedule):raise ValueError('Actual request RNG seed does not match frozen schedule')
            if actual in used:raise ValueError('Actual request RNG seed overlaps another independent sample')
            used[actual]=(ident,index)
    return used
