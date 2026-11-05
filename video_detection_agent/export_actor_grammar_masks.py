#!/usr/bin/env python3
"""CPU replay of actual Actor tokens through the installed native grammar FSM.

No model generation, optimizer, rewards, or behavior-policy equivalence claim.
Run using the same environment as the recorded vLLM/XGrammar backend.
"""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np
from prepare_postroute_online import sha


def token_allowed(mask,token):
    if token<0 or token>=len(mask)*32:return False
    return bool((int(mask[token//32]) >> (token%32)) & 1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--credits',nargs='+',required=True,type=Path)
    ap.add_argument('--server-launch',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Preserve previous exact-mask replay')
    args.out.mkdir(parents=True);start=time.time()
    try:
        import xgrammar as xgr
        from transformers import AutoTokenizer
        launch=json.loads(args.server_launch.read_text());command=launch['command']
        if '--structured-outputs-config' not in command:
            raise ValueError('Bind the explicit native grammar configuration')
        config=json.loads(command[command.index('--structured-outputs-config')+1])
        if config!={'backend':'xgrammar','disable_any_whitespace':True}:
            raise ValueError('Only the inspected exact native compiler settings are supported')
        root=Path(__file__).resolve().parents[1]/'Qwen3.5-4B'
        cfg=json.loads((root/'config.json').read_text());vocab_size=cfg['text_config']['vocab_size']
        tokenizer=AutoTokenizer.from_pretrained(root,trust_remote_code=True,local_files_only=True)
        info=xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=vocab_size)
        compiler=xgr.GrammarCompiler(info,max_threads=4,cache_enabled=True)
        sources=[Path(inspect.getfile(xgr.GrammarCompiler)),Path(inspect.getfile(xgr.GrammarMatcher)),
                 Path(inspect.getfile(xgr.TokenizerInfo))]
        results=[]
        for number,path in enumerate(args.credits):
            credit=json.loads(path.read_text())
            if credit['source_partition']!='train':raise ValueError('Train-only policy math preflight')
            request_path=Path(credit['request_path'])
            if sha(request_path)!=credit['request_sha256']:raise ValueError('Actual request changed')
            request=json.loads(request_path.read_text())
            for im in request['images']:
                if sha(Path(im['path']))!=im['sha256']:raise ValueError('Actual input pixels changed')
            output=request['response_format'];matcher=None;schema=None
            if output:
                if output['type']!='json_schema':raise ValueError('Unsupported actual grammar type')
                schema=output['json_schema']['schema']
                compiled=compiler.compile_json_schema(json.dumps(schema),any_whitespace=False)
                matcher=xgr.GrammarMatcher(compiled,max_rollback_tokens=0)
            indices=credit['credited_completion_indices'];wanted=set(indices);masks=[];counts=[]
            workspace=xgr.allocate_token_bitmask(1,vocab_size)
            all_ids=credit['completion_token_ids']
            for i,token in enumerate(all_ids):
                workspace.fill_(-1)
                if matcher:matcher.fill_next_token_bitmask(workspace)
                mask=workspace[0].numpy().copy()
                if not token_allowed(mask,token):
                    raise ValueError(f'Native actual token {i}/{token} is outside the replayed grammar')
                if i in wanted:
                    masks.append(mask)
                    unpacked=np.unpackbits(mask.view(np.uint8),bitorder='little')[:vocab_size]
                    counts.append(int(unpacked.sum()))
                if matcher and not matcher.accept_token(token):
                    raise ValueError('Native actual token failed grammar-state transition')
            if len(masks)!=len(indices):raise ValueError('Credited mask coverage incomplete')
            maskpath=args.out/f'{number:03d}_masks.npz'
            np.savez_compressed(maskpath,bitmask=np.stack(masks),
                                indices=np.array(indices,dtype=np.int64),
                                token_ids=np.array([all_ids[i] for i in indices],dtype=np.int64))
            results.append(dict(credit_path=str(path.resolve()),credit_sha256=sha(path),
                request_path=str(request_path),request_sha256=sha(request_path),
                actor_model=credit['actor_model'],stage=credit['stage'],vocab_size=vocab_size,
                masks_path=str(maskpath.resolve()),masks_sha256=sha(maskpath),
                grammar_applied=bool(matcher),actual_completion_tokens_replayed=len(all_ids),
                credited_tokens=len(indices),allowed_counts=counts,
                recorded_request_temperature=request['temperature'],
                schema_terminated=matcher.is_terminated() if matcher else None,
                probability_behavior_equivalence='UNPROVEN_EFFECTIVE_SAMPLING_CONTRACT_REQUIRED'))
        report=dict(status='PASS',rows=len(results),results=results,
            structured_output_configuration=config,native_server_launch_sha256=sha(args.server_launch),
            tokenizer_config_sha256=sha(root/'tokenizer_config.json'),tokenizer_json_sha256=sha(root/'tokenizer.json'),
            native_compiler_version=importlib.metadata.version('xgrammar'),
            compiler_sources={str(f):sha(f) for f in sources},
            actual_native_tokens_all_accepted=True,model_generation_executed=False,
            semantic_support_validated=False,reward_computed=False,optimizer_steps=0,test_used=False,
            scope='Exact FSM-mask replay only; not old/new gradient or behavior-distribution equivalence',
            elapsed_seconds=time.time()-start,exporter_sha256=sha(Path(__file__)))
        (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.out/'exporter_snapshot.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(report,indent=2))
    except Exception as exc:
        (args.out/'FAILURE.json').write_text(json.dumps(dict(error=repr(exc),automatic_retry=False,optimizer_steps=0),indent=2)+'\n')
        raise


if __name__=='__main__':main()
