"""The co-reported late window is fixed, and mixed signs stay mixed."""
import pytest
from experiments.representation_study.analysis import history_metrics,paired_summary


def history(value):return [{'update':u,'equal_connection_force_mae_eV_per_A':value+u/1e8} for u in range(0,200001,500)]


def test_exact_scheduled_late20_window_and_complete_support():
    h=history(.03);m=history_metrics(h)
    assert m['late20_updates']==list(range(190500,200001,500))
    assert m['late20_equal_connection_force_mae_eV_per_A']==pytest.approx(sum(x['equal_connection_force_mae_eV_per_A'] for x in h[-20:])/20)
    assert m['best_update']==0
    with pytest.raises(ValueError,match='schedule'):history_metrics(h+[h[-1]])
    with pytest.raises(ValueError,match='schedule'):history_metrics(h[1:])


def test_paired_delta_sign_and_mixed_interpretation():
    rows=[]
    for seed in (0,1):
        rows.extend([{'candidate':f'E64-{seed}','best':.03,'late20':.05}, {'candidate':f'SP-{seed}','best':.04,'late20':.04}])
    out=paired_summary(rows)['SP']
    assert out['best_mean_delta']==pytest.approx(.01) and out['late20_mean_delta']==pytest.approx(-.01)
    assert out['interpretation']=='mixed'
    with pytest.raises(ValueError):paired_summary(rows[:-1])


def test_interrupted_scoring_uses_fresh_attempt_and_keeps_old_files(tmp_path,monkeypatch):
    import experiments.representation_study.analysis as analysis
    root=tmp_path/'analysis';old=root/'000001';old.mkdir(parents=True);(old/'best-validation.npz').write_bytes(b'retained partial evidence')
    called=[]
    def score(run,out):called.append(out);out.mkdir(parents=True);return {'status':'done'}
    monkeypatch.setattr(analysis,'score_run',score)
    result=analysis.score_fit('unused',root)
    assert result['status']=='done' and called[0].name=='000002'
    assert (old/'best-validation.npz').read_bytes()==b'retained partial evidence'


def test_cached_score_metric_edits_are_detected_without_model_loading(tmp_path,monkeypatch):
    from dataclasses import asdict
    import hashlib,json
    import experiments.representation_study.analysis as analysis
    from experiments.representation_study.protocol import make_configs
    from force_consistency.data.splits import canonical_hash
    run=tmp_path/'run';run.mkdir();out=tmp_path/'analysis';out.mkdir()
    cfg=make_configs('unused','unused',88)['SP-0'];(run/'config.json').write_text(json.dumps(cfg))
    source={k:'fixed' for k in ('code_fingerprint','code_files','lock_sha256','python','dependencies')}
    prov={**source,'config_computation_hash':canonical_hash({k:v for k,v in cfg.items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')})}
    (run/'provenance.json').write_text(json.dumps(prov));monkeypatch.setattr(analysis,'code_provenance',lambda root:source)
    for role in ('best','last'):(run/(role+'.msgpack')).write_bytes(b'not deserialized in this check')
    (out/'score.json').write_text('{"edited_metric": 0.0}')
    (out/'score.sha256').write_text(hashlib.sha256(b'original score').hexdigest())
    with pytest.raises(ValueError,match='Cached score fields changed'):analysis.score_run(run,out)


def test_corrupt_old_timing_is_unknown_without_blocking_scoring(tmp_path):
    from experiments.representation_study.analysis import segment_timings
    old=tmp_path/'segments/000001';new=tmp_path/'segments/000002';old.mkdir(parents=True);new.mkdir()
    (old/'summary.json').write_text('{truncated')
    (new/'summary.json').write_text('{"elapsed_seconds": 3.0, "steady_update_seconds_mean": 0.01}')
    rows=segment_timings(new)
    assert rows[0]['elapsed_seconds'] is None and rows[0]['timing_status']=='unknown'
    assert rows[1]['elapsed_seconds']==3.0 and (old/'summary.json').read_text()=='{truncated'


def test_nonfinite_optional_old_timing_becomes_unknown(tmp_path):
    import json
    from experiments.representation_study.analysis import segment_timings
    folder=tmp_path/'segments/000001';folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps({'elapsed_seconds':3.,'validation_seconds_total':float('nan')}))
    assert segment_timings(folder)[0]['timing_status']=='unknown'
