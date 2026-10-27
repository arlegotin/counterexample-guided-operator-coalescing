"""Task 8 floating qualification; verified public local teacher, synthetic token prompts only.

Run explicitly outside the filesystem sandbox for native Metal and nested OS
offline denial. No model/data download and no protected corpus access.
"""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tests'))
from poc.bank import BankDecoder, canonicalize_partition, owner_census
from poc.bundle import archive_bundle, load_bundle, measure_bundle, measure_resident, write_bundle
from poc.contracts import ContinuationRequest, RuntimeSpec
from poc.teacher import AssetManifest, load_teacher, load_tokenizer, sha256_file
from test_model_parity import batch
from test_native_parity import native_batch
from test_offline_bundle import run_offline_bundle
from test_export import ordinary_artifact_inference

PRIOR_ATTEMPTS = (Path('/private/tmp/cgoc-task8-floating-attempt1/summary.json'),Path('/private/tmp/cgoc-task8-floating-attempt2/summary.json'),Path('/private/tmp/cgoc-task8-floating-attempt3/summary.json'),Path('/private/tmp/cgoc-task8-floating-attempt4/summary.json'))
CAPS = {'wall_seconds':1200,'process_rss_bytes':6*1024**3,'unified_compute_bytes':6*1024**3,'generated_disk_bytes':6*1024**3}
ASSET_HASH = '9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083'


def main(output):
    out = Path(output).resolve(); out.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); available = psutil.virtual_memory().available
    prior_cost = sum(json.loads(p.read_text())['resource_observations']['wall_seconds'] for p in PRIOR_ATTEMPTS if p.exists())
    remaining_wall = CAPS['wall_seconds']-prior_cost
    prior_disk = sum(sum(f.stat().st_size for f in p.parent.rglob('*') if f.is_file()) for p in PRIOR_ATTEMPTS if p.exists())
    if remaining_wall <= 0: raise RuntimeError('aggregate actual qualification wall budget exhausted')
    if available < CAPS['unified_compute_bytes']: raise RuntimeError('less available memory than declared unified ceiling; wait for headroom')
    if psutil.disk_usage(out).free < CAPS['generated_disk_bytes']: raise RuntimeError('insufficient declared disk headroom')
    print(json.dumps({'preflight':CAPS,'available_memory_bytes':available,'disk_free_bytes':psutil.disk_usage(out).free,'prior_actual_seconds':prior_cost,'remaining_actual_seconds':remaining_wall}),flush=True)
    report = {'schema_version':1,'status':'running','caps':CAPS,'available_memory_before_bytes':available,
              'disk_free_before_bytes':psutil.disk_usage(out).free,'source_asset_sha256':ASSET_HASH,
              'source_commit':os.popen('git rev-parse HEAD').read().strip(),
              'scope':'floating identity/format only; synthetic tokenizer prompts, no quality advantage or B*',
              'prior_attempts':[{'summary':str(p),'sha256':sha256_file(p),'status':json.loads(p.read_text())['status'],'cost':json.loads(p.read_text())['resource_observations']} for p in PRIOR_ATTEMPTS if p.exists()],
              'prior_actual_seconds':prior_cost,'rows':[],'artifacts':{},'offline':[],'resource_observations':{}}
    done = threading.Event(); observed = {'parent_peak_rss_bytes':0,'largest_child_rss_bytes':0,'generated_disk_peak_bytes':0}
    def budget_signal(signum,frame): raise RuntimeError('measured resource/time budget exceeded')
    signal.signal(signal.SIGTERM,budget_signal)
    def monitor():
        while not done.wait(.2):
            try:
                rss = psutil.Process().memory_info().rss; observed['parent_peak_rss_bytes'] = max(observed['parent_peak_rss_bytes'],rss)
                children = psutil.Process().children(recursive=True)
                child = max([p.memory_info().rss for p in children if p.is_running()] or [0])
                observed['largest_child_rss_bytes'] = max(observed['largest_child_rss_bytes'],child)
                if max(rss,child)>CAPS['process_rss_bytes'] or time.monotonic()-start>remaining_wall:
                    os.kill(os.getpid(),signal.SIGTERM); return
            except psutil.Error: pass
    thread = threading.Thread(target=monitor,daemon=True); thread.start()
    try:
        torch.set_num_threads(4)
        asset = AssetManifest.load(ROOT/'data/prepared-135m-v2/teacher-assets.json',expected_sha256=ASSET_HASH)
        tokenizer = load_tokenizer(asset); source = load_teacher(asset)
        assert source.head_dim == 64 and source.rope_theta == 100000
        files = json.loads(asset.path.read_text())['assets']
        tokenizer_files = {Path(f['local_path']).name:str(asset.path.parent/f['local_path']) for f in files
                           if Path(f['local_path']).name not in ('config.json','model.safetensors','generation_config.json')}
        apache = Path(importlib.metadata.distribution('transformers')._path)/'licenses/LICENSE'
        notice = out/'teacher-NOTICE.txt'
        notice.write_text('HuggingFaceTB/SmolLM2-135M model and tokenizer; revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2.\n'
                          'Apache-2.0 per verified teacher-assets.json source records. Full standard license text from local transformers distribution.\n'
                          'Candidate-specific runtime source is this project\'s code; common packages remain separate installation dependencies.\n')
        installed_bytes = sum(p.stat().st_size for p in (ROOT/'.venv').rglob('*') if p.is_file() and not p.is_symlink())
        runtime = RuntimeSpec(tokenizer_files,{'Apache-2.0.txt':str(apache),'teacher-NOTICE.txt':str(notice)},
                              str(ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
                              install_bytes=installed_bytes,install_unknown_reason=None)
        report['install'] = {'bytes':installed_bytes,'method':'sum all regular non-symlink .venv files, including common runtime packages and caches; conservative entire environment boundary',
                             'lock_sha256':sha256_file(Path(runtime.dependency_lock)),'python':sys.version}
        prompt = 'The small model reads a simple sentence. '
        token_ids = tokenizer.encode(prompt*100,add_special_tokens=False)
        assert len(token_ids)>=512
        policy = hashlib.sha256(b'synthetic-task-v1; exact token IDs; raw summed conditional log probabilities; no length normalization').hexdigest()
        pids = tuple(tokenizer.encode('The small model reads',add_special_tokens=False))
        requests = [ContinuationRequest(pids,tuple(tokenizer.encode(text,add_special_tokens=False)),policy,f'option-{i}')
                    for i,text in enumerate((' a sentence.',' a tree.'))]
        report['task_bridge'] = {'policy_sha256':policy,'prompt_sha256':hashlib.sha256(b'The small model reads').hexdigest(),
                                'prompt_ids':pids,'continuation_ids':[r.continuation_ids for r in requests],
                                'normalization':'raw sum, FP64 aggregation of FP32 token scores','scored_mask':'all continuation tokens, identical across backends',
                                'boundary_policy':'frozen separate synthetic prompt/continuation IDs; no benchmark outcome or protected data'}
        for kind in ('dense','compact'):
            begin = time.monotonic()
            if kind == 'dense': candidate = source
            else:
                candidate = BankDecoder(source,canonicalize_partition([list(range(i,i+2)) for i in range(0,30,2)],30),8,8,initialize=False)
                # Exercise nonzero private factors without SVD, fitting or training.
                with torch.no_grad():
                    for layer in candidate.layers:
                        for factors in layer.mlp.residuals.values():
                            factors.A.fill_(.0001); factors.B.fill_(.0002)
            candidate.requires_grad_(False)
            ref = write_bundle(candidate,out/kind,runtime)
            loaded = load_bundle(Path(ref.directory),backend='torch')
            import mlx.core as mx
            assert mx.metal.is_available(), 'real MLX Metal required'
            native = load_bundle(Path(ref.directory),backend='mlx')
            census = owner_census(candidate)
            assert owner_census(loaded) == census
            assert measure_resident(native).owner_count == len(census)
            assert measure_resident(loaded).owner_count == len(census)
            assert native.owner('lm_head.weight') is native.owner('embed_tokens.weight')
            artifact = {'manifest_sha256':ref.manifest_sha256,'payload_bytes':measure_bundle(Path(ref.directory)).payload_bytes,
                        'resident_bytes':measure_resident(native).resident_bytes,'owners':len(census),
                        'conversion_load_seconds':time.monotonic()-begin,'source_checkpoint_digest':candidate.manifest().checkpoint_digest,
                        'source_owner_numel':sum(p.numel() for p in candidate.parameters()),
                        'norm_epsilon_preserved':all(loaded.norm.variance_epsilon == candidate.norm.variance_epsilon for _ in [0]),
                        'rank':getattr(candidate,'rank',0),'groups':len(getattr(candidate,'bank',candidate.layers))}
            print(json.dumps({'converted':kind,**artifact}),flush=True)
            for length in (16,128,512):
                b = batch([token_ids[:length]],offset=7)
                nb = native_batch(b)
                with torch.no_grad():
                    expected = candidate.logits(b).detach().numpy()
                    scientific = loaded.logits(b).detach().numpy()
                    scientific_cache = loaded.new_cache(start_position=7)
                    cut = min(8,length-1)
                    pieces = [loaded.logits(batch([token_ids[a:z]],offset=7+a),cache=scientific_cache)
                              for a,z in ((0,cut),(cut,length))]
                    cached_scientific = torch.cat(pieces,dim=1).detach().numpy()
                actual = native.logits(nb); mx.eval(actual)
                cache = native.new_cache(start_position=7)
                pieces = [native.logits(native_batch(batch([token_ids[a:z]],offset=7+a)),cache=cache)
                          for a,z in ((0,cut),(cut,length))]
                cached = mx.concatenate(pieces,axis=1); mx.eval(cached)
                errors = {'scientific_export_max_abs':float(np.max(np.abs(scientific-expected))),
                          'scientific_cached_max_abs':float(np.max(np.abs(cached_scientific-expected))),
                          'native_full_max_abs':float(np.max(np.abs(np.array(actual)-expected))),
                          'native_cached_max_abs':float(np.max(np.abs(np.array(cached)-expected)))}
                # Bounds declared before observing real-model results. Exact tensor
                # serialization is tested separately; floating backend errors accumulate.
                assert errors['scientific_export_max_abs'] <= 1e-6,errors
                assert errors['scientific_cached_max_abs'] <= .002,errors
                assert errors['native_full_max_abs'] <= .003,errors
                assert errors['native_cached_max_abs'] <= .003,errors
                row = {'kind':kind,'context_tokens':length,'offset':7,'token_ids_sha256':hashlib.sha256(np.array(token_ids[:length],dtype='<i8').tobytes()).hexdigest(),
                       'mask':'all valid; targets all adjacent positions','max_abs_tolerance':{'scientific_export':1e-6,'scientific_cache':.002,'native':.003},**errors}
                report['rows'].append(row); print(json.dumps(row),flush=True)
                del expected,scientific,cached_scientific,actual,cached,pieces,scientific_cache,cache
                gc.collect(); mx.clear_cache()
            scientific_scores = candidate.score_continuations(requests)
            native_scores = native.score_continuations(requests)
            score_error = max(abs(a.log_likelihood-b.log_likelihood) for a,b in zip(scientific_scores,native_scores,strict=True))
            assert score_error<=.002
            assert np.argmax([s.log_likelihood for s in scientific_scores]) == np.argmax([s.log_likelihood for s in native_scores])
            artifact['task'] = {'scientific_scores':[s.log_likelihood for s in scientific_scores],
                                'native_scores':[s.log_likelihood for s in native_scores], 'max_abs':score_error,
                                'prediction':int(np.argmax([s.log_likelihood for s in native_scores]))}
            artifact['mlx_memory'] = {'active_bytes':mx.get_active_memory(),'cache_bytes':mx.get_cache_memory(),'peak_bytes':mx.get_peak_memory(),
                                      'counters_nonadditive_with_rss':True}
            assert mx.get_peak_memory()<=CAPS['unified_compute_bytes']
            # Drop model allocations before a fresh child process; one compute job.
            del native,loaded
            if kind != 'dense': del candidate
            gc.collect(); mx.clear_cache()
            for backend in ('torch','mlx'):
                offline_work = out/(kind+'-'+backend+'-offline'); offline_work.mkdir()
                proof = run_offline_bundle(ref.directory,offline_work,backend,os_denial=True)
                ordinary_artifact_inference(ref.directory,offline_work,backend=backend)
                proof['ordinary_import_payload_unchanged'] = True
                report['offline'].append({'kind':kind,**proof})
                print(json.dumps({'offline':kind,'backend':backend,'os_source_denied':proof['os_source_denied']}),flush=True)
            artifact['archive'] = {'bytes':None,'sha256':None,'unpack_seconds':None,
                                   'status':'not_run_budget','unknown_reason':'level9 gzip produced only 84.8 MB partial compressed output in interrupted 192 s pre-ZIP attempt; full-model archive repetition exceeds remaining aggregate cap. Complete tiny dense/rank0/rank2 archives verify deterministic equality and safe extraction accounting.'}
            report['artifacts'][kind] = artifact
            disk = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            observed['generated_disk_peak_bytes'] = max(observed['generated_disk_peak_bytes'],disk)
            assert disk+prior_disk<=CAPS['generated_disk_bytes']
            (out/'partial.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'; report['failure'] = repr(exc)
        raise
    finally:
        done.set(); thread.join()
        observed['generated_disk_peak_bytes'] = max(observed['generated_disk_peak_bytes'],sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        report['resource_observations'] = observed | {'wall_seconds':time.monotonic()-start,'available_memory_after_bytes':psutil.virtual_memory().available,
                                                    'ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                    'rss_and_mlx_are_overlapping_not_added':True,
                                                    'aggregate_actual_seconds':prior_cost+time.monotonic()-start,
                                                    'prior_generated_disk_bytes':prior_disk,
                                                    'aggregate_generated_disk_bytes':prior_disk+observed['generated_disk_peak_bytes']}
        report['source_sha256'] = {str(p.relative_to(ROOT)):sha256_file(p) for p in [ROOT/'provenance/measure_bundle_deployment.py',ROOT/'tests/test_offline_bundle.py',ROOT/'tests/test_native_parity.py',*[ROOT/'poc'/f for f in __import__('poc.bundle',fromlist=['RUNTIME_FILES']).RUNTIME_FILES]]}
        (out/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':report['status'],'cost':report['resource_observations'],'summary':str(out/'summary.json')}),flush=True)


if __name__ == '__main__': main(sys.argv[1])
