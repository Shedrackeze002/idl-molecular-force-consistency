"""Reviewed replacement controls and ancestry. No numerical imports or array reads."""
import json
import os
from pathlib import Path
from .width_rate_contract import canonical_hash, digest, read, safe_path, verify_width_rate_sources
from .width_rate_recovery_contract import verify_recovery_sources

CONFIG_ID='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36'
PINS_ID='557daf51680f48a9c852bde83a133ee6bac0796bb36103bc2ae22c833ca66cb6'
ROLE='fresh_four_cell_width_rate_diagnostic'
ROOT=Path('runs/aspirin-width-rate-replacement-2026-10-01')
ADMIN_KEYS={'run_directory','source_pin_manifest','source_pin_manifest_sha256','source_pin_count',
            'original_width_rate_config','original_failed_run_directory','verified_copy_directory',
            'replacement_role','maximum_matrix_invocations','maximum_probe_attempts_per_cell'}


def validate_replacement(config):
    if canonical_hash(config)!=CONFIG_ID:
        raise ValueError('Unreviewed replacement configuration')


def no_symlinks(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Symlinked artifact path')
    return path


def verify_replacement_sources(config):
    validate_replacement(config)
    repo=Path.cwd().resolve();manifest=safe_path(repo,config['source_pin_manifest'])
    no_symlinks(manifest)
    if digest(manifest)!=PINS_ID:raise ValueError('Replacement source pin map changed')
    pins=read(manifest)['files_sha256']
    if len(pins)!=518:raise ValueError('Replacement source pin count changed')
    for name,want in pins.items():
        path=safe_path(repo,name);no_symlinks(repo/name)
        if digest(path)!=want:raise ValueError('Replacement source changed: '+name)
    original=read(config['original_width_rate_config'])
    if {k:v for k,v in config.items() if k not in ADMIN_KEYS}!={k:v for k,v in original.items() if k not in ADMIN_KEYS}:
        raise ValueError('Replacement scientific controls differ')
    old=verify_width_rate_sources(original)
    recovery=verify_recovery_sources(read('configs/aspirin-width-rate-recovery.json'))
    copy=read(Path(config['verified_copy_directory'])/'summary.json')
    if copy['status']!='passed' or not copy['verification_valid']:
        raise ValueError('Verified copy ancestry changed')
    no_symlinks(repo/ROOT)
    return {**old,'config_identity':CONFIG_ID,'pin_manifest_identity':PINS_ID,'pin_count':518,'pins':pins,
            'original_config':original,'original_sources':old,
            'ancestry':{'original_root':config['original_failed_run_directory'],'original_status':'failed',
                        'original_execution_valid':False,'copy_root':config['verified_copy_directory'],
                        'copy_status':'passed','copy_training_updates':0,'original_effort_complete':False,
                        'copy_checkpoint_sha256':copy['child_receipt']['checkpoint_sha256']}}


def claim_root(root):
    """Exclusive creation; the caller owns nothing if creation fails."""
    root=no_symlinks(root)
    root.mkdir(parents=True,exist_ok=False)
    return root
