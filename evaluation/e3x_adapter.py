"""Authenticated E3x BEST supplier for the existing force-only evaluator."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from e3x_phase_common import (assert_imports,read_json,digest,source_pin,
                              require_equal_pin,dataset_pin)
from benchmark_e3x_budget import cpu_context
from evaluation.adapter import (callable_arm,_validated_positions,
                                ArmConfigurationError,_require_finite_values)
from force_consistency.models.energy import make_batch


def e3x_checkpoint_arm(run_directory,launch_record,score_directory,*,acceptance):
    """Restore one score-bound BEST. Acceptance is a frozen per-model binding.

    The reviewed driver derives the binding from the accepted C or matched
    comparison receipt; this supplier restores exactly those bound bytes.
    Forces are physical predictor outputs without projection or rescaling.
    """
    assert_imports();run=Path(run_directory);score_dir=Path(score_directory)
    require_equal_pin(source_pin(),launch_record['source_pin'])
    score=read_json(score_dir/'score.json');checkpoint=run/'best.msgpack'
    if (digest(checkpoint.read_bytes())!=acceptance['checkpoint_sha256']
            or digest((score_dir/'score.json').read_bytes())!=acceptance['score_sha256']
            or type(acceptance['best_update']) is not int
            or score['best_update']!=acceptance['best_update']
            or score['best_checkpoint_sha256']!=acceptance['checkpoint_sha256']
            or score['status']!='verified' or score['source_pin']!=launch_record['source_pin']
            or score['candidate']!=launch_record['candidate']
            or any(score[k] is not True for k in ('source_matches','best_reload_verified'))):
        raise ValueError('Accepted BEST/score binding mismatch')
    if digest((score_dir/'best-validation.npz').read_bytes())!=score['prediction_sha256']:
        raise ValueError('Accepted predictions changed')
    with cpu_context():return _restore(run,launch_record,score,acceptance)


def _restore(run,launch,score,acceptance):
    import jax
    from ase.data import chemical_symbols
    from force_consistency.train import (RunConfig,model_and_predictor,initialize_state,
                                         initialization_identity)
    from force_consistency.data.prepare import load_training_bundle
    from force_consistency.metrics import partition_batch
    from force_consistency.runs import load_checkpoint
    from force_consistency.data.splits import canonical_hash
    from evaluate_e3x_adequacy import verify_embedded_best
    cfg=RunConfig(**read_json(run/'config.json'));prov=read_json(run/'provenance.json')
    summary=read_json(run/'summary.json')
    if asdict(cfg)!=launch['config'] or prov!=summary['provenance']:raise ValueError('Run configuration/provenance mismatch')
    if (summary['status']!='passed' or summary['updates']!=cfg.max_updates
            or summary['stop_reason']!='max_updates' or score['updates']!=cfg.max_updates
            or summary['best_update']!=score['best_update']):raise ValueError('Incomplete accepted fit')
    computation=canonical_hash({k:v for k,v in asdict(cfg).items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')})
    if (prov['config_computation_hash']!=computation or prov['arm']!=cfg.arm
            or prov['loss_weights']!=cfg.loss_weights or prov['consistency_strength']!=cfg.consistency_strength):raise ValueError('Objective/configuration mismatch')
    for key in ('code_fingerprint','code_files','lock_sha256','python','dependencies'):
        if prov[key]!=launch['source_pin'][key]:raise ValueError('Numerical source mismatch')
    require_equal_pin(dataset_pin(cfg.data_directory),launch['dataset_pin'])
    bundle=load_training_bundle(cfg.data_directory,expected_split_hash=cfg.expected_split_hash)
    if prov['dataset_identity']!=bundle.manifest['identity'] or prov['normalization']!=asdict(bundle.normalization):raise ValueError('Normalization/dataset mismatch')
    model,predictor=model_and_predictor(cfg)
    template=initialize_state(model,partition_batch(bundle.train,np.arange(cfg.batch_size)),bundle.normalization,cfg.seed,cfg.learning_rate)
    initial=initialization_identity(template)
    if initial!=launch['initial_identity'] or any(prov[k]!=v for k,v in initial.items()):raise ValueError('Initial state mismatch')
    embedded=verify_embedded_best(run/'best.msgpack',prov)
    state=load_checkpoint(run/'best.msgpack',prov,template=template)
    if int(state['update'])!=acceptance['best_update'] or state['normalization_json']!=json.dumps(asdict(bundle.normalization),sort_keys=True):raise ValueError('Restored BEST state mismatch')
    proof=read_json(run/'best-checkpoint-verification.json')
    if proof['status']!='passed' or proof['role']!='best' or proof['update']!=int(state['update']):raise ValueError('BEST fresh-process proof missing')
    z=np.asarray(bundle.train.atomic_numbers[0],dtype=np.int64)
    if not np.all(bundle.train.atomic_numbers==z) or not np.all(bundle.validation.atomic_numbers==z):raise ValueError('Species order varies')
    species=tuple(chemical_symbols[int(value)] for value in z)
    predict=jax.jit(lambda batch:predictor(state['params'],batch,model=model))

    def evaluated(positions):
        array=np.asarray(positions,dtype=np.float64)
        if array.ndim==2:batch_positions=array[None]
        elif array.ndim==3:batch_positions=array
        else:raise ArmConfigurationError('E3x positions require (N,3) or (B,N,3)')
        if (not len(batch_positions) or batch_positions.shape[1:]!=(len(z),3)
                or not np.isfinite(batch_positions).all()):raise ArmConfigurationError('E3x positions must be finite, nonempty and species-aligned')
        with cpu_context():
            batch=make_batch(batch_positions,np.broadcast_to(z,batch_positions.shape[:2]))
            energy,forces=jax.device_get(predict(batch))
        forces=np.asarray(forces,dtype=np.float64).reshape(batch_positions.shape)
        return np.asarray(energy,dtype=np.float64),forces[0] if array.ndim==2 else forces

    arm_id=f'e3x-{cfg.arm}-seed{cfg.seed}-best{int(state["update"])}-{acceptance["checkpoint_sha256"]}'
    arm=callable_arm(lambda positions:evaluated(positions)[1],species,arm_id)
    arm.update(kind='checkpoint',checkpoint_identity={**acceptance,'arm':cfg.arm,'seed':cfg.seed,
        'lambda':cfg.consistency_strength,'source_fingerprint':launch['source_pin']['code_fingerprint'],
        'dataset_identity':bundle.manifest['identity'],'precision':'float64','backend':'cpu',
        'energy_convention':'training-offset residual eV','energy_offset_eV':bundle.normalization.energy_offset_eV,
        'forces_are_energy_gradient':cfg.arm=='energy','embedded_best_identity':embedded})
    if cfg.arm=='energy':
        def energy(positions,species=None,record_id=None):
            array=_validated_positions(positions,arm['species'],species,arm_id)
            if array.ndim!=2:raise ArmConfigurationError('Scalar molecular energy requires one unbatched molecule')
            values,_=evaluated(array)
            _require_finite_values(values,record_id,"energy")
            return float(values[0]),{'status':'ok','record_id':record_id,'energy_convention':'training-offset residual eV','energy_offset_eV':bundle.normalization.energy_offset_eV}
        arm.update(provides_energy=True,energy=energy)
    return arm
