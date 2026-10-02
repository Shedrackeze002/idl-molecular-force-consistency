"""Saved-only report, archive and standard figure; never executes a model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile


def main():
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location('capacity_spikes_saved', 'scripts/diagnose_capacity_spikes.py')
    analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)
    config = analysis.read('configs/aspirin-capacity-spike.json')
    root = Path(config['run_directory']); records = Path('reports/execution/spike-diagnostic-2026-10-02')
    from force_consistency.capacity_contract import no_symlinks
    inventory = {str(p.relative_to(root)): analysis.digest(no_symlinks(p)) for p in root.rglob('*') if p.is_file()}
    result = analysis.render(root, 'reports')
    with tempfile.TemporaryDirectory(prefix='spike-report-check-') as temporary:
        again = analysis.render(root, temporary)
        assert result == again
        for name in ('capacity-spike-diagnostics.json', 'capacity-spike-diagnostics.md'):
            assert (Path('reports') / name).read_bytes() == (Path(temporary) / name).read_bytes()
    inventory_path = records / 'numerical-run-inventory.json'
    with inventory_path.open('x') as file: json.dump(inventory, file, indent=2, sort_keys=True); file.write('\n')
    archive = records / 'numerical-run.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(inventory): zipped.write(root / name, name)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert len(zipped.namelist()) == len(set(zipped.namelist())) == len(inventory)
        assert set(zipped.namelist()) == set(inventory)
        for name in inventory: assert hashlib.sha256(zipped.read(name)).hexdigest() == inventory[name]
    figures = {}
    if result['summary']['execution_valid']:
        os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='idl-spike-matplotlib-'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for row, (width, updates) in enumerate([('width32', [3600, 3700, 3800, 6000]), ('width64', [4900, 5000, 5100, 6000])]):
            for column, role in enumerate(('fitting', 'anchor')):
                ax = axes[row, column]; x = np.arange(4)
                for offset, key, label in [(-.25, 'energy_mse_eV2', 'Raw MSE'), (0, 'mean_energy_error_squared_eV2', 'Mean error squared'),
                                           (.25, 'centered_energy_error_variance_eV2', 'Centered variance')]:
                    values = [result['cases'][f'{width}-u{u}']['observations']['residuals'][role][key] for u in updates]
                    ax.bar(x + offset, values, .23, label=label)
                ax.set_yscale('log'); ax.set_xticks(x, updates); ax.set_xlabel('Selected saved update')
                ax.set_ylabel('Energy error component (eV²)'); ax.set_title(width + ' / ' + role)
                ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
        fig.suptitle('Raw energy error decomposition at selected saved states (all structures fitted)')
        fig.tight_layout(); figure = records / 'energy-residual-components.png'; fig.savefig(figure, dpi=180); plt.close(fig)
        figures[str(figure)] = analysis.digest(figure)
    after = {str(p.relative_to(root)): analysis.digest(no_symlinks(p)) for p in root.rglob('*') if p.is_file()}
    assert inventory == after
    analysis.verify_execution_integrity(config, expected_gate=analysis.read(root / 'prelaunch-gate.json'))
    record = {'status': 'saved_only_verified', 'execution_valid': result['summary']['execution_valid'],
              'canonical_json_markdown_match': True, 'raw_file_count': len(inventory), 'raw_files_unchanged': True,
              'archive_members': len(inventory), 'archive_crc': 'passed', 'every_member_hash_matches': True,
              'archive_sha256': analysis.digest(archive), 'inventory_sha256': analysis.digest(inventory_path),
              'report_sha256': {str(Path('reports') / name): analysis.digest(Path('reports') / name)
                                for name in ('capacity-spike-diagnostics.json', 'capacity-spike-diagnostics.md')},
              'figures': figures, 'independent_result_review': 'pending',
              'new_training_updates': 0, 'new_predictions': 0, 'new_random_parameter_initializations': 0,
              'elapsed_seconds': time.perf_counter() - started,
              'cap_scope': 'Saved-only work after numerical closure; separately timed'}
    with (records / 'saved-closeout-verification.json').open('x') as file: json.dump(record, file, indent=2, sort_keys=True); file.write('\n')
    analysis.evidence(config, 'SPIKE-SAVED-CLOSEOUT-001', 'saved-only canonical report/archive verification', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
