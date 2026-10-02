"""Separately timed saved-only rendering/archive/plot; no model calls."""
import hashlib,json,os
from pathlib import Path
import sys,tempfile,time,zipfile
sys.path.insert(0,str(Path('scripts').absolute()))
from report_offset_profile import render,matching_rows,cc,verify_inputs,verify_gate
start=time.perf_counter();records=Path(__file__).parent;config=cc.read('configs/aspirin-offset-profile.json');root=Path(config['run_directory'])
def inventory():return {str(p.relative_to(root)):cc.digest(cc.no_symlinks(p)) for p in root.rglob('*') if p.is_file()}
before=inventory();report=render(root,'reports')
with tempfile.TemporaryDirectory(prefix='offset-saved-regeneration-') as temporary:
    again=render(root,Path(temporary).resolve())
    assert report==again
    for name in ('offset-profile-diagnostics.json','offset-profile-diagnostics.md'):
        assert (Path('reports')/name).read_bytes()==(Path(temporary)/name).read_bytes()
with (records/'numerical-run-inventory.json').open('x') as f:json.dump(before,f,sort_keys=True,indent=2);f.write('\n')
archive=records/'numerical-run.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name in sorted(before):z.write(root/name,name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))==len(before)
    assert set(z.namelist())==set(before)
    for name,sha in before.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
figures={}
if report['execution_valid']:
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='offset-plot-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref=Path(config['source_root']);commit=cc.read(ref/'commit.json');reference=cc.read(ref/commit['history'])
    pairs=matching_rows(reference,report['audit']['history']);assert len(pairs)==62
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,role in enumerate(('fitting','anchor')):
        for col,key in enumerate(cc.MSE_KEYS):
            ax=axes[row,col];baseline=report['contrasts'][role][key]['original_baseline']
            ax.plot([u for u,a,b in pairs],[a['metrics'][role][key]/baseline for u,a,b in pairs],label='Historical width32',linewidth=1.3)
            ax.plot([u for u,a,b in pairs],[b['metrics'][role][key]/baseline for u,a,b in pairs],label='Offset profiled',linewidth=1.3)
            ax.axhline(.1,color='black',linestyle=':',label='Strict joint gate threshold')
            ax.set_yscale('log');ax.set_title(role+' / '+('Energy' if col==0 else 'Forces'));ax.set_ylabel('MSE / original zero-state MSE');ax.set_xlabel('Accepted Adam update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('62 matching postzero observations; all structures fitted')
    fig.tight_layout();path=records/'matched-raw-mse.png';fig.savefig(path,dpi=180);plt.close(fig);figures[str(path)]=cc.digest(path)
    (records/'matched-raw-mse-caption.md').write_text('Raw energy and force MSE fractions on62 shared postzero observations from1 through6000, relative to original unprojected zero-state baselines. Historical width32 and once-only analytical fitting-bias profiling use the same128 fitted examples and16 fitted anchors. Lines join sparse observations and do not establish continuous stability. Fitting mean-zero bias is enforced; raw force/centered-energy behavior remains empirical. One seed/rate and historical reference limit causal and generalization claims. Dense projection corrections are excluded from this matched-grid comparison.\n')
assert before==inventory();verify_inputs(config);verify_gate(config,cc.read(root/'prelaunch-gate.json'))
record={'status':'saved_only_verified','execution_valid':report['execution_valid'],'canonical_json_markdown_match':True,
 'raw_file_count':len(before),'raw_files_unchanged':True,'archive_members':len(before),'archive_crc':'passed','every_member_hash_matches':True,
 'archive_sha256':cc.digest(archive),'inventory_sha256':cc.digest(records/'numerical-run-inventory.json'),
 'report_sha256':{str(Path('reports')/name):cc.digest(Path('reports')/name) for name in ('offset-profile-diagnostics.json','offset-profile-diagnostics.md')},
 'figures':figures,'independent_saved_result_review':'pending','new_training_updates':0,'new_predictions':0,'new_random_parameter_initializations':0,
 'elapsed_seconds':time.perf_counter()-start,'cap_scope':'saved-only work separately timed after numerical closure'}
with (records/'saved-closeout-verification.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(record,indent=2))
