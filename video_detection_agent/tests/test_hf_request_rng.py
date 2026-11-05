"""Request streams must not overlap across independently sampled trajectories."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hf_request_rng import request_seed,validate_seeds,STRIDE


def test_consecutive_start_seeds_cannot_collide_across_four_rounds():
    bases=[202610194100+i for i in range(4)]
    seeds=[request_seed(base,turn) for base in bases for turn in range(4)]
    assert len(set(seeds))==16
    assert [request_seed(base,0) for base in bases]==bases
    assert request_seed(bases[0],1)!=request_seed(bases[1],0)


def test_actual_legacy_seed_overlap_is_rejected_without_relabelling_requests():
    manifest=[{'source':{'merged_id':'v'},'sample':i,'seed':100+i} for i in range(4)]
    results=[{'source_id':'v','sample':i,'requests':[{'item_path':f'{i}-0'},{'item_path':f'{i}-1'}]} for i in range(4)]
    reader=lambda p:({}, {'seed':100+int(p[0])+int(p[-1])},{})
    with pytest.raises(ValueError,match='frozen schedule'):validate_seeds(results,manifest,reader)
    reader=lambda p:({}, {'seed':request_seed(100+int(p[0]),int(p[-1]))},{})
    assert len(validate_seeds(results,manifest,reader))==8


def test_only_unchanged_direct_requests_can_reuse_original_first_seeds():
    manifest=[{'source':{'merged_id':'v'},'sample':i,'seed':100+i} for i in range(4)]
    direct=[{'source_id':'v','sample':i,'requests':[{'item_path':str(i)}]} for i in range(4)]
    assert len(validate_seeds(direct,manifest,lambda p:({}, {'seed':100+int(p)},{})))==4
    with pytest.raises(ValueError):request_seed(STRIDE,0)
