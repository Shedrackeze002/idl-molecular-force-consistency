"""Frozen recovery provenance. No numerical imports or array loading."""
from pathlib import Path
from .width_rate_contract import canonical_hash,digest,read,safe_path,resolve_commit,verify_width_rate_sources,expected_updates

CONFIG_ID='55c63996e555fad2c65edb38525dcd27f022188650ee255353b24e20a83c487c'
PINS_ID='f7fff5b24a9e2eae365a97924d720d996d9c60321c39fc00711d307a03e5a4d4'
ROLE='copied_checkpoint_verification_only'


def validate(config):
    if canonical_hash(config)!=CONFIG_ID:raise ValueError('Unreviewed recovery configuration')


def verify_recovery_sources(config):
    validate(config);repo=Path.cwd().resolve();manifest=safe_path(repo,config['recovery_pin_manifest'])
    if digest(manifest)!=PINS_ID:raise ValueError('Recovery pin map changed')
    pins=read(manifest)['files_sha256']
    if len(pins)!=163:raise ValueError('Recovery pin count mismatch')
    for name,want in pins.items():
        path=safe_path(repo,name)
        if Path(name).is_symlink() or digest(path)!=want:raise ValueError('Recovery source changed: '+name)
    original=read(config['original_width_rate_config']);verify_width_rate_sources(original)
    if canonical_hash(original)!=config['original_width_rate_config_identity']:raise ValueError('Original config changed')
    source=safe_path(repo,config['failed_run_directory'])/config['source_cell'];receipt,history=resolve_commit(source)
    if (receipt['checkpoint_sha256']!=config['expected_source_checkpoint_sha256'] or receipt['history_sha256']!=config['expected_source_history_sha256']
        or receipt['state_hashes']!=config['expected_source_state_hashes'] or receipt['update']!=6000
        or [r['update'] for r in history]!=expected_updates()):raise ValueError('Original endpoint changed')
    prov=read(source/'provenance.json');contract=read(source/'checkpoint-contract.json');case=read(source/'case.json')
    if (contract['commit']!=receipt or contract['state_hashes']!=receipt['state_hashes'] or contract['inputs']!=prov['inputs'] or contract['case']!=case
        or contract['normalization']!=prov['normalization'] or contract['config_identity']!=canonical_hash(original) or read(source/'config.json')!=original
        or contract['input_files_sha256']!=prov['input_files_sha256'] or contract['reference_sha256']!=digest(source/'reference.npz')
        or any(digest(source/f'{r}-inputs.npz')!=prov['input_files_sha256'][r] for r in ('fitting','anchor'))
        or [len(prov['inputs'][r]['record_ids']) for r in ('fitting','anchor')]!=[128,16]):raise ValueError('Source input/reference contract changed')
    report=read('reports/width-rate-diagnostics.json')
    if report['status']!='failed' or report['execution_valid'] or report['interpretation_valid'] or report['full_pilot']!='pending':raise ValueError('Original failure status changed')
    destination=Path(config['run_directory']);resolved=safe_path(repo,str(destination))
    if destination.is_symlink() or resolved==source or resolved.is_relative_to(source.parent) or resolved.is_relative_to(repo/'reports/execution'):raise ValueError('Unsafe recovery root')
    return {'config':config,'source_root':source,'source_receipt':receipt,'source_contract':contract,'provenance':prov,'old_report':report,'source_files_sha256':{str(Path(name).relative_to(source.relative_to(repo))):want for name,want in pins.items() if Path(name).is_relative_to(source.relative_to(repo))}}
