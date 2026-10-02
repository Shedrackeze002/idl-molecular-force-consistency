"""Saved-only figures/archive verification. Never trains or initializes a model."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

from force_consistency import capacity_contract as cc
from force_consistency import capacity_report as cr


def main():
    start=time.perf_counter();c=cc.read('configs/aspirin-capacity.json');root=Path(c['run_directory']);out=Path('reports/execution/capacity-2026-10-02')
    report=cr.build_report(root)
    primary_missing=not cr.REPORT_JSON.exists() or not cr.REPORT_MD.exists()
    if primary_missing:
        if report['execution_valid']:raise ValueError('Cannot publish a valid comparison without original closure')
        cr.write_report(report)
    cr.verify_saved_report(report)
    observed={}
    for name,a in report['cases'].items():
        observed[name]={'execution_valid':a['execution_valid'],'updates':a.get('updates'),
                        'persisted_updates':a.get('verified_persisted_updates'),'child_work':a.get('child_work'),
                        'learning_accepted':a.get('learning_accepted'),'endpoint_metrics':a.get('endpoint_metrics'),
                        'sampled_peaks':a.get('sampled_peaks'),'terminal_window':a.get('terminal_window')}
    # Retain every numerical file, including partial/error artifacts and code snapshot.
    files={str(p.relative_to(root)):cc.digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    inventory_path=out/'numerical-run-inventory.json'
    with inventory_path.open('x') as f:json.dump(files,f,sort_keys=True,indent=2);f.write('\n')
    archive=out/'numerical-run.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in files:z.write(root/name,name)
    with zipfile.ZipFile(archive) as z:
        crc=z.testzip()
        if crc is not None or set(z.namelist())!=set(files):raise ValueError('Archive CRC/member inventory differs')
        if any(hashlib.sha256(z.read(name)).hexdigest()!=sha for name,sha in files.items()):raise ValueError('Archived member bytes differ')
    if any(cc.digest(root/name)!=sha for name,sha in files.items()):raise ValueError('Raw numerical artifacts changed during closeout')
    figures={}
    if all(a['execution_valid'] for a in report['cases'].values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
        for row,role in enumerate(('fitting','anchor')):
            for col,key in enumerate(cc.MSE_KEYS):
                ax=axes[row,col];values=[]
                for name,a in report['cases'].items():
                    x=[h['update'] for h in a['history']];y=[h[role+'_gate']['fraction_remaining'][key] for h in a['history']]
                    values+=y;ax.plot(x,y,label=name,linewidth=1.5,marker='.',markersize=3)
                ax.axhline(.1,color='black',linestyle='--',linewidth=1,label='10% raw gate')
                if all(v>0 for v in values):
                    ax.set_yscale('log');ax.set_ylim(min(min(values)*.8,.05),max(max(values)*1.2,1.2))
                else:ax.set_ylim(min(0,min(values)),max(max(values)*1.2,1.2))
                ax.set_title(role.capitalize()+(' energy' if col==0 else ' force'));ax.grid(True,alpha=.2)
                ax.set_ylabel('Fraction of original update0 raw MSE');ax.set_xlabel('Committed update')
                ax.set_xlim(0,c['max_updates']);ax.legend(fontsize=8)
        fig.suptitle('Fresh capacity diagnostic · fitted structures · sampled observations' if report['execution_valid'] else 'Invalid global execution · descriptive saved case observations only')
        for extension in ('png','svg'):
            path=out/f'capacity-trajectories.{extension}';fig.savefig(path,dpi=180);figures[str(path)]=cc.digest(path)
        plt.close(fig)
    summary={'status':'saved_only_verified','execution_valid':report['execution_valid'],
             'canonical_json_markdown_match':True,'primary_report_missing_before_closeout':primary_missing,'raw_file_count':len(files),'archive_members':len(files),
             'archive_crc':'passed','every_archive_member_hash_matches':True,'raw_files_unchanged':True,
             'archive_sha256':cc.digest(archive),'inventory_sha256':cc.digest(inventory_path),'figures':figures,
             'report_sha256':cc.digest(cr.REPORT_JSON),'report_md_sha256':cc.digest(cr.REPORT_MD),
             'observed_cases':observed,'verification_errors':report['verification_errors'],'limits':report['limits'],
             'elapsed_seconds':time.perf_counter()-start,
             'cap_scope':'Saved-only archive/figures after immutable scientific cap closure, no new numerical work',
             'new_training_updates':0,'new_model_initializations':0,'new_predictions':0,'independent_result_review':'pending'}
    with (out/'saved-closeout-verification.json').open('x') as f:json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
    cr.evidence(c,'CAPACITY-SAVED-CLOSEOUT-001','canonical saved reconstruction and verified numerical archive/figures',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('observed_cases','limits')},indent=2))


if __name__=='__main__':main()
