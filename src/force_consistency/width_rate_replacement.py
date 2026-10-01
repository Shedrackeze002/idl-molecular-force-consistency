"""Guarded replacement entry points. Numerical engine is imported only after pure guards."""
from pathlib import Path
from .width_rate_contract import read
from .width_rate_replacement_contract import validate_replacement, verify_replacement_sources


def _engine():
    from . import width_rate_runs
    return width_rate_runs


def replacement_preflight(config,context,save_partial):
    sources=verify_replacement_sources(config)
    prepared=_engine().width_rate_preflight(sources['original_config'],context,save_partial)
    if prepared['sources']!=sources['original_sources']:
        raise ValueError('Original numerical preflight source facts changed')
    prepared['sources']=verify_replacement_sources(config)
    return prepared


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(args.config);validate_replacement(config)
    raise RuntimeError('Replacement execution not implemented')
